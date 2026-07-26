from __future__ import annotations

import html
import json
import logging
import signal
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import __version__
from .delivery import TelegramDelivery
from .models import AlertEvent, Assessment, NewsItem, RenderedMessage
from .policy import assess, is_fresh, is_quiet_window, quiet_window_end
from .render import render_p0, render_p1
from .sources import (
    FmpCollector,
    GdeltCollector,
    GoogleNewsCollector,
    HttpClient,
    RssCollector,
)
from .store import RadarStore, is_infra_event
from .summarizer import LlmSummarizer
from .util import (
    SGT,
    atomic_write,
    canonicalize_url,
    hamming_distance,
    json_dumps,
    normalize_title,
    redact_secrets,
    register_secrets,
    simhash64,
    stable_hash,
    strip_html,
    utc_now,
    visible_length,
)

LOGGER = logging.getLogger(__name__)


class RadarService:
    def __init__(
        self,
        *,
        root: Path,
        config: dict[str, Any],
        env: dict[str, str],
        send_enabled: bool,
    ):
        self.root = root
        self.config = config
        self.env = env
        register_secrets(
            env.get(name, "")
            for name in (
                "FMP_API_KEY",
                "TELEGRAM_BOT_TOKEN",
                "LLM_API_KEY",
            )
        )
        self.news_thread_id = env.get(
            "TELEGRAM_NEWS_THREAD_ID",
            str(config["telegram"]["news_thread_id"]),
        )
        self.monitor_thread_id = env.get(
            "TELEGRAM_MONITOR_THREAD_ID",
            str(config["telegram"]["monitor_thread_id"]),
        )
        self.store = RadarStore(root / "state" / "radar.sqlite3")
        self.client = HttpClient(
            user_agent=env.get(
                "RADAR_USER_AGENT",
                f"global-news-radar/{__version__} "
                "(+https://github.com/EricEEEEEEE/global-news-radar)",
            ),
            attempts=int(config["runtime"]["source_retry_attempts"]),
        )
        self.rss = RssCollector(self.client, self.store)
        self.gnews = GoogleNewsCollector(
            self.rss, int(config["discovery"]["max_queries"])
        )
        self.gdelt = GdeltCollector(self.client)
        self.fmp = FmpCollector(
            self.client,
            env.get("FMP_API_KEY", ""),
            config["structured"],
        )
        self.summarizer = LlmSummarizer(
            enabled=bool(config["llm"]["enabled"]),
            base_url=env.get("LLM_API_BASE", ""),
            api_key=env.get("LLM_API_KEY", ""),
            model=env.get("LLM_MODEL", config["llm"]["model"]),
            timeout=int(config["llm"]["timeout_seconds"]),
            max_output_tokens=int(config["llm"]["max_output_tokens"]),
        )
        self.delivery = TelegramDelivery(
            token=env.get("TELEGRAM_BOT_TOKEN", ""),
            chat_id=env.get("TELEGRAM_CHAT_ID", str(config["telegram"]["chat_id"])),
            outbox_dir=root / "outbox",
            max_visible_chars=int(config["telegram"]["max_visible_chars"]),
            send_enabled=send_enabled,
        )
        self._cycle = int(self.store.get_meta("cycle_count", "0") or "0")
        self._stopping = False
        self.readonly = False
        self.last_cycle_stats: dict[str, Any] = {}

    def close(self) -> None:
        self.store.close()

    def request_stop(self, *_: object) -> None:
        self._stopping = True

    def _source_call(
        self,
        source_id: str,
        callback: Any,
        now: datetime,
        errors: list[dict[str, Any]],
        recoveries: list[dict[str, Any]] | None = None,
        tier: str = "secondary",
    ) -> list[NewsItem]:
        window_hours = float(self.config["runtime"]["source_health_window_hours"])
        try:
            items = callback()
            if not self.readonly:
                recovery = self.store.source_success(source_id, now, window_hours)
                if recovery and recoveries is not None:
                    recoveries.append(recovery)
            return items
        except Exception as exc:  # noqa: BLE001
            # Upstream error text carries the full request URL, and the FMP URL
            # carries the API key. Redact before it reaches SQLite, the log file
            # or a Telegram alert.
            reason = redact_secrets(exc)
            if self.readonly:
                errors.append(
                    {
                        "source_id": source_id,
                        "error": reason[:300],
                        "failures": 0,
                        "tier": tier,
                    }
                )
                LOGGER.error(
                    "source_failed source=%s error=%s", source_id, reason[:300]
                )
                return []
            count = self.store.source_failure(source_id, reason, now, window_hours)
            attempts, failures = self.store.source_window(source_id)
            errors.append(
                {
                    "source_id": source_id,
                    "error": reason[:300],
                    "failures": count,
                    "window_attempts": attempts,
                    "window_failures": failures,
                    "tier": tier,
                }
            )
            LOGGER.error(
                "source_failed source=%s failures=%s error=%s",
                source_id,
                count,
                reason[:300],
            )
            return []

    def _collect_feeds(
        self,
        feeds: list[dict[str, Any]],
        source_tier: str,
        now: datetime,
        errors: list[dict[str, Any]],
        recoveries: list[dict[str, Any]],
    ) -> list[NewsItem]:
        items: list[NewsItem] = []
        for feed in feeds:
            items.extend(
                self._source_call(
                    str(feed["id"]),
                    lambda feed=feed: self.rss.collect(
                        source_id=str(feed["id"]),
                        source_name=str(feed["name"]),
                        url=str(feed["url"]),
                        source_tier=source_tier,
                        region=str(feed["region"]),
                        category_hint=str(feed["category_hint"]),
                        now=now,
                    ),
                    now,
                    errors,
                    recoveries,
                    source_tier,
                )
            )
        return items

    def collect(
        self, now: datetime
    ) -> tuple[list[NewsItem], list[dict[str, Any]], list[dict[str, Any]]]:
        items: list[NewsItem] = []
        errors: list[dict[str, Any]] = []
        recoveries: list[dict[str, Any]] = []
        first_collection = not self.store.baseline_complete()
        items.extend(
            self._collect_feeds(
                list(self.config["official_feeds"]),
                "primary",
                now,
                errors,
                recoveries,
            )
        )

        discovery_due = (
            first_collection
            or self._cycle % int(self.config["discovery"]["interval_cycles"]) == 0
        )
        if bool(self.config["discovery"]["enabled"]) and discovery_due:
            discovery_items = self._collect_feeds(
                list(self.config["discovery"].get("feeds") or []),
                "secondary",
                now,
                errors,
                recoveries,
            )
            discovery_items.extend(
                self._source_call(
                    "gnews",
                    lambda: self.gnews.collect(now, now.astimezone(SGT)),
                    now,
                    errors,
                    recoveries,
                    "secondary",
                )
            )
            items.extend(discovery_items)
            # A thin batch is the symptom of a throttled or reshaped feed, and it
            # is exactly when the fallback is needed; waiting for a fully empty
            # batch meant the fallback almost never ran.
            floor = int(self.config["discovery"].get("gdelt_min_items") or 0)
            if len(discovery_items) < floor and bool(
                self.config["discovery"]["gdelt_fallback"]
            ):
                items.extend(
                    self._source_call(
                        "gdelt",
                        lambda: self.gdelt.collect(now, now.astimezone(SGT)),
                        now,
                        errors,
                        recoveries,
                        "secondary",
                    )
                )

        structured_due = (
            first_collection
            or self._cycle % int(self.config["structured"]["interval_cycles"]) == 0
        )
        if bool(self.config["structured"]["fmp_enabled"]) and structured_due:
            # FMP is the only source of scheduled macro and earnings releases:
            # losing it is closer to losing an official feed than to losing one
            # of eleven discovery feeds, so it alerts on the strict threshold.
            items.extend(
                self._source_call(
                    "fmp_macro",
                    lambda: self.fmp.collect_macro(now),
                    now,
                    errors,
                    recoveries,
                    "primary",
                )
            )
            items.extend(
                self._source_call(
                    "fmp_earnings",
                    lambda: self.fmp.collect_earnings(now),
                    now,
                    errors,
                    recoveries,
                    "primary",
                )
            )
        return items, errors, recoveries

    def _is_near_duplicate(self, item: NewsItem, now: datetime) -> bool:
        if self.store.item_seen(item.identity):
            return True
        fingerprint = simhash64(item.title)
        since = now - timedelta(hours=float(self.config["topic_cooldown_hours"]))
        for _, previous in self.store.recent_simhashes(since):
            if hamming_distance(fingerprint, previous) <= 5:
                return True
        return False

    def _mark_item(self, item: NewsItem, now: datetime) -> None:
        canonical = canonicalize_url(item.url)
        normalized = normalize_title(item.title, item.source)
        self.store.mark_item(
            identity=item.identity,
            canonical_url=canonical,
            norm_title=normalized,
            simhash=simhash64(item.title),
            source=item.source,
            now=now,
        )

    def _corroborating_items(
        self, item: NewsItem, assessment: Assessment, now: datetime
    ) -> list[NewsItem]:
        since = now - timedelta(minutes=int(self.config["corroboration_minutes"]))
        found: dict[str, NewsItem] = {
            observation.identity: observation
            for observation in self.store.recent_observations(
                assessment.event_key, since
            )
        }
        fingerprint = simhash64(item.title)
        for event_key, previous, observation in self.store.observation_cluster(
            since, assessment.category
        ):
            if event_key == assessment.event_key:
                continue
            if hamming_distance(fingerprint, previous) <= 5:
                found.setdefault(observation.identity, observation)
        return list(found.values())

    def _qualify_event(
        self, item: NewsItem, assessment: Assessment, now: datetime
    ) -> AlertEvent | None:
        self.store.add_observation(
            assessment.event_key, item, now, simhash64(item.title)
        )
        observations = self._corroborating_items(item, assessment, now)
        unique_sources = {observation.source.lower() for observation in observations}
        if assessment.requires_corroboration and len(unique_sources) < 2:
            LOGGER.info(
                "p0_waiting_corroboration event=%s sources=%s",
                assessment.event_key,
                sorted(unique_sources),
            )
            return None
        if assessment.requires_corroboration:
            major_sources = {
                value.lower() for value in self.config["discovery"]["major_sources"]
            }
            reputable = {
                source
                for source in unique_sources
                if any(major in source for major in major_sources)
            }
            if len(reputable) < 2:
                return None
        return AlertEvent(assessment=assessment, items=observations or [item])

    def _record_success(
        self, message: RenderedMessage, result: dict[str, Any], now: datetime
    ) -> None:
        if result.get("dry_run"):
            return
        self.store.record_delivery(
            message=message,
            message_id=str(result["message_id"]),
            visible_chars=visible_length(message.html),
            now=now,
        )
        for event_key in message.event_keys:
            # Data-source alerts share the delivery path but are not market
            # events: they must not enter the topic ledger or the export.
            if is_infra_event(event_key):
                continue
            observation_rows = self.store.recent_observations(
                event_key, now - timedelta(hours=6)
            )
            source_count = len({item.source for item in observation_rows}) or 1
            # Prefer the deterministic material hash carried by the evidence.
            row = self.store.connection.execute(
                "SELECT assessment_json FROM p1_buffer WHERE event_key=?",
                (event_key,),
            ).fetchone()
            material_hash = message.content_hash[:20]
            topic_anchor = ""
            lineage_day = 1
            for evidence in message.evidence:
                if evidence.get("event_key") == event_key and evidence.get(
                    "material_hash"
                ):
                    material_hash = str(evidence["material_hash"])
                    topic_anchor = str(evidence.get("topic_anchor") or "")
                    lineage_day = int(evidence.get("lineage_day") or 1)
                    break
            if row:
                data = json.loads(str(row["assessment_json"]))
                material_hash = str(data["material_hash"])
            self.store.record_topic_sent(
                event_key=event_key,
                material_hash=material_hash,
                level=message.level,
                source_count=source_count,
                summary=message.plain_text,
                now=now,
            )
            self.store.record_lineage_sent(
                topic_anchor=topic_anchor,
                event_key=event_key,
                material_hash=material_hash,
                day_number=lineage_day,
                now=now,
            )
        self.store.remove_p1(message.event_keys)
        self.store.export_deliveries(self.root / "state" / "radar_events.jsonl")

    def _deliver_or_queue(
        self, message: RenderedMessage, thread_id: str, now: datetime
    ) -> bool:
        if self.store.delivery_exists(message.content_hash):
            LOGGER.info(
                "delivery_already_recorded content_hash=%s", message.content_hash
            )
            return True
        try:
            result = self.delivery.send(message, thread_id)
            self._record_success(message, result, now)
            LOGGER.info(
                "delivery_ok level=%s message_id=%s dry_run=%s",
                message.level,
                result.get("message_id"),
                result.get("dry_run"),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            reason = redact_secrets(exc)
            LOGGER.error("delivery_failed level=%s error=%s", message.level, reason)
            self.store.queue_delivery(
                message,
                thread_id=thread_id,
                next_retry_at=now
                + timedelta(
                    minutes=int(self.config["runtime"]["delivery_retry_minutes"])
                ),
                error=reason,
            )
            return False

    def retry_due_deliveries(self, now: datetime) -> int:
        if is_quiet_window(
            now,
            str(self.config["quiet_window"]["start"]),
            str(self.config["quiet_window"]["end"]),
        ):
            return 0
        completed = 0
        for row in self.store.due_deliveries(now):
            message = RenderedMessage.from_dict(json.loads(str(row["message_json"])))
            if self.store.delivery_exists(message.content_hash):
                self.store.connection.execute(
                    "DELETE FROM pending_delivery WHERE content_hash=?",
                    (message.content_hash,),
                )
                self.store.connection.commit()
                continue
            if now - message.created_at > timedelta(minutes=90):
                LOGGER.warning("delivery_expired content_hash=%s", message.content_hash)
                self.store.connection.execute(
                    "DELETE FROM pending_delivery WHERE content_hash=?",
                    (message.content_hash,),
                )
                self.store.connection.commit()
                continue
            attempts = int(row["attempts"])
            if attempts >= int(self.config["runtime"]["delivery_max_attempts"]):
                continue
            try:
                result = self.delivery.send(message, str(row["thread_id"]))
                self._record_success(message, result, now)
                completed += 1
            except Exception as exc:  # noqa: BLE001
                self.store.mark_delivery_retry(
                    message.content_hash,
                    now
                    + timedelta(
                        minutes=int(self.config["runtime"]["delivery_retry_minutes"])
                    ),
                    redact_secrets(exc),
                )
        return completed

    def _alert_threshold(self, tier: str) -> int:
        """Consecutive failures needed before a source is worth interrupting for.

        Thresholds are counts, but the wall-clock meaning depends on the cadence
        of the source. Official feeds run every cycle; discovery feeds run once
        per ``discovery.interval_cycles``, so the same count is ten times longer
        there and a routine CDN hiccup used to page at thirty minutes.
        """
        runtime = self.config["runtime"]
        if tier == "primary":
            return int(runtime["source_failure_alert_after"])
        return int(
            runtime.get("source_failure_alert_after_discovery")
            or runtime["source_failure_alert_after"]
        )

    def _source_error_alerts(self, errors: list[dict[str, Any]], now: datetime) -> None:
        cooldown = float(self.config["runtime"]["source_error_cooldown_hours"])
        realert = float(self.config["runtime"].get("source_realert_minutes") or 0)
        rate_threshold = float(self.config["runtime"]["source_failure_alert_rate"])
        rate_min_attempts = int(self.config["runtime"]["source_failure_rate_min_calls"])
        for error in errors:
            threshold = self._alert_threshold(str(error.get("tier") or "secondary"))
            attempts = int(error.get("window_attempts") or 0)
            failures = int(error.get("window_failures") or 0)
            rate = failures / attempts if attempts else 0.0
            # A source that fails one call in three never reaches a consecutive
            # failure count of 3, so the rate is a second, independent trigger.
            # It still has to clear the tier threshold in total failures, or a
            # source that broke right after a window reset would alert on the
            # rate before its tier's consecutive bar was ever reached.
            flapping = (
                attempts >= rate_min_attempts
                and failures >= threshold
                and rate >= rate_threshold
            )
            if int(error["failures"]) < threshold and not flapping:
                continue
            source_id = str(error["source_id"])
            if not self.store.source_alert_due(source_id, now, cooldown, realert):
                continue
            # Name the trigger that actually fired. A source down six cycles in
            # a row is also a 100% failure rate, and reporting it as a rate hid
            # the simpler and more actionable fact.
            detail = (
                f"连续失败 {int(error['failures'])} 次"
                if int(error["failures"]) >= threshold
                else f"{attempts} 次调用失败 {failures} 次（{rate:.0%}）"
            )
            text = (
                "⚠️ <b>新闻雷达数据源异常</b>\n"
                f"<blockquote>{html.escape(source_id)} {detail}\n"
                f"{html.escape(redact_secrets(error['error'])[:180])}</blockquote>\n"
                f"<i>{now.astimezone(SGT).strftime('%Y-%m-%d %H:%M')} SGT · "
                f"global-news-radar/{__version__}</i>"
            )
            message = RenderedMessage(
                level="P1",
                html=text,
                plain_text=strip_html(text),
                event_keys=[f"source-error:{source_id}"],
                content_hash=stable_hash(text, 32),
                evidence=[error],
                created_at=now,
            )
            if self._deliver_or_queue(message, self.monitor_thread_id, now):
                self.store.mark_source_alerted(source_id, now)

    def _source_recovery_alerts(
        self, recoveries: list[dict[str, Any]], now: datetime
    ) -> None:
        """Close every outage that was announced, so no alert stays open.

        Without this the operator holds a warning that says a source is down and
        has no way to learn it came back ten minutes later.
        """
        for recovery in recoveries:
            source_id = str(recovery["source_id"])
            minutes = recovery.get("outage_minutes")
            duration = f"中断 {int(minutes)} 分钟 · " if minutes is not None else ""
            text = (
                "✅ <b>新闻雷达数据源恢复</b>\n"
                f"<blockquote>{html.escape(source_id)} 已恢复\n"
                f"{duration}连续失败 {int(recovery['failures'])} 次</blockquote>\n"
                f"<i>{now.astimezone(SGT).strftime('%Y-%m-%d %H:%M')} SGT · "
                f"global-news-radar/{__version__}</i>"
            )
            message = RenderedMessage(
                level="P1",
                html=text,
                plain_text=strip_html(text),
                event_keys=[f"source-recovery:{source_id}"],
                content_hash=stable_hash(text, 32),
                evidence=[recovery],
                created_at=now,
            )
            if self._deliver_or_queue(message, self.monitor_thread_id, now):
                self.store.mark_source_recovered(source_id, now)

    def _freshness_filter(
        self, items: list[NewsItem], now: datetime
    ) -> tuple[list[NewsItem], dict[str, int]]:
        fresh: list[NewsItem] = []
        rejected: dict[str, int] = {}
        for item in items:
            ok, reason = is_fresh(
                item,
                now,
                freshness_minutes=int(self.config["freshness_minutes"]),
                future_tolerance_minutes=int(self.config["future_tolerance_minutes"]),
            )
            if not ok:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            fresh.append(item)
        return fresh, rejected

    def inspect(self, now: datetime | None = None) -> dict[str, Any]:
        """Report what the radar sees right now without consuming any state.

        A plain `once` marks every fresh headline seen, so running it by hand
        costs the daemon that batch of news. This path answers the same
        question and writes nothing: no seen_items, no cycle counter, no ETag
        cache, no heartbeat, no delivery.
        """
        now = now or utc_now()
        self.readonly = True
        self.rss.readonly = True
        try:
            # Readonly never records a success, so it can never observe a
            # recovery and must never announce one.
            items, errors, _ = self.collect(now)
            fresh, rejected = self._freshness_filter(items, now)
            preview: list[dict[str, Any]] = []
            for item in fresh:
                assessment = assess(item, self.config)
                if assessment is None:
                    continue
                preview.append(
                    {
                        "level": assessment.level,
                        "category": assessment.category,
                        "event_key": assessment.event_key,
                        "source": item.source,
                        "title": item.title[:160],
                        "requires_corroboration": assessment.requires_corroboration,
                    }
                )
        finally:
            self.readonly = False
            self.rss.readonly = False
        stats = {
            "status": "readonly",
            "fetched": len(items),
            "fresh": len(fresh),
            "eligible": len(preview),
            "rejected": rejected,
            "source_errors": errors,
            "preview": preview[:25],
        }
        LOGGER.info("readonly_cycle %s", json_dumps({**stats, "preview": len(preview)}))
        return stats

    def run_once(
        self, now: datetime | None = None, *, readonly: bool = False
    ) -> dict[str, Any]:
        if readonly:
            return self.inspect(now)
        now = now or utc_now()
        self._cycle += 1
        self.store.set_meta("cycle_count", str(self._cycle), now)
        retries = self.retry_due_deliveries(now)
        items, errors, recoveries = self.collect(now)
        fresh, rejected = self._freshness_filter(items, now)

        first_run = not self.store.baseline_complete()
        if first_run and bool(self.config["runtime"]["baseline_on_first_run"]):
            for item in fresh:
                self._mark_item(item, now)
            self.store.mark_baseline_complete(now)
            stats = {
                "status": "baseline",
                "fetched": len(items),
                "baselined": len(fresh),
                "rejected": rejected,
                "source_errors": errors,
                "sent": 0,
                "retries": retries,
            }
            self._finish_cycle(stats, now)
            return stats

        p0_events: list[AlertEvent] = []
        new_items = 0
        for item in fresh:
            try:
                new_items += self._process_item(item, now, p0_events)
            except Exception:  # noqa: BLE001
                # One malformed headline must not abort the cycle: every other
                # item in this batch has already been marked seen and would be
                # lost for good.
                LOGGER.exception("item_failed identity=%s", item.identity)

        quiet = is_quiet_window(
            now,
            str(self.config["quiet_window"]["start"]),
            str(self.config["quiet_window"]["end"]),
        )
        sent = 0
        unique_p0: dict[str, AlertEvent] = {}
        for event in p0_events:
            unique_p0[event.assessment.event_key] = event
        p0_events = list(unique_p0.values())
        for event in p0_events:
            try:
                sent += self._publish_p0(event, now, quiet)
            except Exception:  # noqa: BLE001
                LOGGER.exception(
                    "p0_render_failed event=%s", event.assessment.event_key
                )

        if not quiet:
            p1_rows = self.store.ready_p1(
                now,
                now - timedelta(minutes=int(self.config["p1_window_minutes"])),
            )
            if len(p1_rows) >= 2:
                entries: list[tuple[AlertEvent, Any]] = []
                for row in p1_rows:
                    item = NewsItem.from_dict(json.loads(str(row["item_json"])))
                    data = json.loads(str(row["assessment_json"]))
                    data["entities"] = tuple(data["entities"])
                    assessment = Assessment(**data)
                    event = AlertEvent(assessment=assessment, items=[item])
                    entries.append((event, self.summarizer.summarize(event)))
                message = render_p1(
                    entries,
                    now,
                    int(self.config["telegram"]["max_visible_chars"]),
                )
                if self._deliver_or_queue(message, self.news_thread_id, now):
                    sent += 1

        self._source_recovery_alerts(recoveries, now)
        self._source_error_alerts(errors, now)
        self.store.prune(
            now,
            int(self.config["topic_retention_days"]),
            item_retention_days=int(self.config["item_retention_days"]),
            delivery_retention_days=int(self.config["delivery_retention_days"]),
        )
        pending_count, deadletter_count = self.store.pending_counts(
            int(self.config["runtime"]["delivery_max_attempts"])
        )
        stats = {
            "status": "ok",
            "fetched": len(items),
            "fresh": len(fresh),
            "new_eligible": new_items,
            "p0_qualified": len(p0_events),
            "p1_buffered": len(
                self.store.ready_p1(
                    now,
                    now - timedelta(minutes=int(self.config["p1_window_minutes"])),
                )
            ),
            "quiet_window": quiet,
            "sent": sent,
            "retries": retries,
            "rejected": rejected,
            "source_errors": errors,
            "source_recoveries": [str(r["source_id"]) for r in recoveries],
            "pending_deliveries": pending_count,
            "deadletter_deliveries": deadletter_count,
        }
        self._finish_cycle(stats, now)
        return stats

    def _process_item(
        self, item: NewsItem, now: datetime, p0_events: list[AlertEvent]
    ) -> int:
        assessment = assess(item, self.config)
        if assessment is None:
            if not self.store.item_seen(item.identity):
                self._mark_item(item, now)
            return 0
        # Add an observation before near-duplicate suppression so a second
        # independent outlet can corroborate a secondary-source P0.
        event = self._qualify_event(item, assessment, now)
        near_duplicate = self._is_near_duplicate(item, now)
        if near_duplicate and not (
            event is not None and assessment.requires_corroboration
        ):
            return 0
        self._mark_item(item, now)
        allowed, reason = self.store.topic_decision(
            event_key=assessment.event_key,
            material_hash=assessment.material_hash,
            level=assessment.level,
            now=now,
            cooldown_hours=float(self.config["topic_cooldown_hours"]),
        )
        if not allowed:
            LOGGER.info(
                "topic_suppressed event=%s reason=%s", assessment.event_key, reason
            )
            return 1
        if event is None:
            return 1
        lineage_allowed, lineage_reason, lineage_day = self.store.lineage_decision(
            topic_anchor=assessment.topic_anchor,
            material_hash=assessment.material_hash,
            now=now,
            max_gap_days=float(self.config["lineage_max_gap_days"]),
            min_interval_minutes=float(self.config["anchor_min_interval_minutes"]),
            daily_cap=int(self.config["anchor_daily_cap"]),
        )
        if not lineage_allowed:
            LOGGER.info(
                "lineage_suppressed anchor=%s reason=%s",
                assessment.topic_anchor,
                lineage_reason,
            )
            return 1
        assessment.lineage_day = lineage_day
        if assessment.level == "P0":
            p0_events.append(event)
        else:
            self.store.buffer_p1(
                event_key=assessment.event_key,
                item=event.primary,
                assessment=assessment.to_dict(),
                now=now,
                expires_at=now
                + timedelta(minutes=int(self.config["p1_expiry_minutes"])),
            )
        return 1

    def _publish_p0(self, event: AlertEvent, now: datetime, quiet: bool) -> int:
        summary = self.summarizer.summarize(event)
        message = render_p0(
            event,
            summary,
            now,
            int(self.config["telegram"]["max_visible_chars"]),
        )
        if quiet:
            quiet_end = quiet_window_end(now, str(self.config["quiet_window"]["end"]))
            self.store.queue_delivery(
                message,
                thread_id=self.news_thread_id,
                next_retry_at=quiet_end.astimezone(now.tzinfo),
                error="quiet_window",
            )
            return 0
        if self._deliver_or_queue(message, self.news_thread_id, now):
            return 1
        return 0

    def _finish_cycle(self, stats: dict[str, Any], now: datetime) -> None:
        self.last_cycle_stats = stats
        heartbeat = {
            "service": "global-news-radar",
            "version": __version__,
            "timestamp": now.isoformat(),
            "timestamp_sgt": now.astimezone(SGT).isoformat(),
            "pid": __import__("os").getpid(),
            "stats": stats,
        }
        atomic_write(self.root / "state" / "heartbeat.json", json_dumps(heartbeat))
        LOGGER.info("cycle_complete %s", json_dumps(stats))

    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        LOGGER.info(
            "service_start poll_seconds=%s send_enabled=%s",
            self.config["poll_seconds"],
            self.delivery.send_enabled,
        )
        while not self._stopping:
            started = time.monotonic()
            try:
                self.run_once()
            except Exception:  # noqa: BLE001
                LOGGER.exception("cycle_crashed")
            elapsed = time.monotonic() - started
            remaining = max(1.0, float(self.config["poll_seconds"]) - elapsed)
            deadline = time.monotonic() + remaining
            while not self._stopping and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))
        LOGGER.info("service_stop")


def check_health(root: Path, max_age_seconds: int = 600) -> tuple[bool, dict[str, Any]]:
    path = root / "state" / "heartbeat.json"
    if not path.exists():
        return False, {"status": "missing_heartbeat", "path": str(path)}
    # The health check is what the watchdog trusts, so a half-written or
    # corrupt heartbeat has to read as unhealthy, never as a crash.
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        timestamp = datetime.fromisoformat(str(payload["timestamp"]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, {
            "status": "unreadable_heartbeat",
            "path": str(path),
            "error": str(exc)[:200],
        }
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    age = (utc_now() - timestamp).total_seconds()
    deadletters = int(payload.get("stats", {}).get("deadletter_deliveries", 0))
    healthy = age <= max_age_seconds and deadletters == 0
    if age > max_age_seconds:
        status = "stale"
    elif deadletters:
        status = "deadletter"
    else:
        status = "ok"
    return healthy, {
        "status": status,
        "age_seconds": round(age, 1),
        "deadletter_deliveries": deadletters,
        "heartbeat": payload,
    }
