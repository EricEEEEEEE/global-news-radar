from __future__ import annotations

import json
import logging
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import __version__
from .delivery import TelegramDelivery
from .models import AlertEvent, Assessment, NewsItem, RenderedMessage
from .policy import assess, is_fresh, is_quiet_window
from .render import render_p0, render_p1
from .sources import (
    FmpCollector,
    GdeltCollector,
    GoogleNewsCollector,
    HttpClient,
    RssCollector,
)
from .store import RadarStore
from .summarizer import LlmSummarizer
from .util import (
    SGT,
    atomic_write,
    canonicalize_url,
    hamming_distance,
    json_dumps,
    normalize_title,
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
        self.store = RadarStore(root / "state" / "radar.sqlite3")
        self.client = HttpClient(
            user_agent=env.get(
                "RADAR_USER_AGENT",
                f"global-news-radar/{__version__} "
                "(+https://github.com/EricEEEEEEE/global-news-radar)",
            )
        )
        self.rss = RssCollector(self.client, self.store)
        self.gnews = GoogleNewsCollector(self.rss)
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
        self.news_thread_id = env.get(
            "TELEGRAM_NEWS_THREAD_ID",
            str(config["telegram"]["news_thread_id"]),
        )
        self.monitor_thread_id = env.get(
            "TELEGRAM_MONITOR_THREAD_ID",
            str(config["telegram"]["monitor_thread_id"]),
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
    ) -> list[NewsItem]:
        try:
            items = callback()
            self.store.source_success(source_id, now)
            return items
        except Exception as exc:  # noqa: BLE001
            count = self.store.source_failure(source_id, str(exc), now)
            errors.append(
                {"source_id": source_id, "error": str(exc)[:300], "failures": count}
            )
            LOGGER.exception("source_failed source=%s failures=%s", source_id, count)
            return []

    def collect(self, now: datetime) -> tuple[list[NewsItem], list[dict[str, Any]]]:
        items: list[NewsItem] = []
        errors: list[dict[str, Any]] = []
        first_collection = not self.store.baseline_complete()
        for feed in self.config["official_feeds"]:
            items.extend(
                self._source_call(
                    str(feed["id"]),
                    lambda feed=feed: self.rss.collect(
                        source_id=str(feed["id"]),
                        source_name=str(feed["name"]),
                        url=str(feed["url"]),
                        source_tier="primary",
                        region=str(feed["region"]),
                        category_hint=str(feed["category_hint"]),
                        now=now,
                    ),
                    now,
                    errors,
                )
            )

        discovery_due = (
            first_collection
            or self._cycle % int(self.config["discovery"]["interval_cycles"]) == 0
        )
        if bool(self.config["discovery"]["enabled"]) and discovery_due:
            discovery_items = self._source_call(
                "gnews",
                lambda: self.gnews.collect(now, now.astimezone(SGT)),
                now,
                errors,
            )
            items.extend(discovery_items)
            if not discovery_items and bool(self.config["discovery"]["gdelt_fallback"]):
                items.extend(
                    self._source_call(
                        "gdelt",
                        lambda: self.gdelt.collect(now, now.astimezone(SGT)),
                        now,
                        errors,
                    )
                )

        structured_due = (
            first_collection
            or self._cycle % int(self.config["structured"]["interval_cycles"]) == 0
        )
        if bool(self.config["structured"]["fmp_enabled"]) and structured_due:
            items.extend(
                self._source_call(
                    "fmp_macro", lambda: self.fmp.collect_macro(now), now, errors
                )
            )
            items.extend(
                self._source_call(
                    "fmp_earnings",
                    lambda: self.fmp.collect_earnings(now),
                    now,
                    errors,
                )
            )
        return items, errors

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

    def _qualify_event(
        self, item: NewsItem, assessment: Assessment, now: datetime
    ) -> AlertEvent | None:
        self.store.add_observation(assessment.event_key, item, now)
        observations = self.store.recent_observations(
            assessment.event_key,
            now - timedelta(minutes=int(self.config["corroboration_minutes"])),
        )
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
            LOGGER.exception("delivery_failed level=%s", message.level)
            self.store.queue_delivery(
                message,
                thread_id=thread_id,
                next_retry_at=now
                + timedelta(
                    minutes=int(self.config["runtime"]["delivery_retry_minutes"])
                ),
                error=str(exc),
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
                    str(exc),
                )
        return completed

    def _source_error_alerts(self, errors: list[dict[str, Any]], now: datetime) -> None:
        threshold = int(self.config["runtime"]["source_failure_alert_after"])
        cooldown = float(self.config["runtime"]["source_error_cooldown_hours"])
        for error in errors:
            if int(error["failures"]) < threshold:
                continue
            source_id = str(error["source_id"])
            if not self.store.source_alert_due(source_id, now, cooldown):
                continue
            text = (
                "⚠️ <b>新闻雷达数据源异常</b>\n"
                f"<blockquote>{source_id} 连续失败 {error['failures']} 次\n"
                f"{str(error['error'])[:180]}</blockquote>\n"
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

    def run_once(self, now: datetime | None = None) -> dict[str, Any]:
        now = now or utc_now()
        self._cycle += 1
        self.store.set_meta("cycle_count", str(self._cycle), now)
        retries = self.retry_due_deliveries(now)
        items, errors = self.collect(now)
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
            assessment = assess(item, self.config)
            if assessment is None:
                if not self.store.item_seen(item.identity):
                    self._mark_item(item, now)
                continue
            # Add an observation before near-duplicate suppression so a second
            # independent outlet can corroborate a secondary-source P0.
            event = self._qualify_event(item, assessment, now)
            near_duplicate = self._is_near_duplicate(item, now)
            if near_duplicate and not (
                event is not None and assessment.requires_corroboration
            ):
                continue
            self._mark_item(item, now)
            new_items += 1
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
                continue
            if event is None:
                continue
            lineage_allowed, lineage_reason, lineage_day = self.store.lineage_decision(
                topic_anchor=assessment.topic_anchor,
                material_hash=assessment.material_hash,
                now=now,
            )
            if not lineage_allowed:
                LOGGER.info(
                    "lineage_suppressed anchor=%s reason=%s",
                    assessment.topic_anchor,
                    lineage_reason,
                )
                continue
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
            summary = self.summarizer.summarize(event)
            message = render_p0(
                event,
                summary,
                now,
                int(self.config["telegram"]["max_visible_chars"]),
            )
            if quiet:
                local = now.astimezone(SGT)
                quiet_end = local.replace(hour=21, minute=30, second=0, microsecond=0)
                self.store.queue_delivery(
                    message,
                    thread_id=self.news_thread_id,
                    next_retry_at=quiet_end.astimezone(now.tzinfo),
                    error="quiet_window",
                )
                continue
            if self._deliver_or_queue(message, self.news_thread_id, now):
                sent += 1

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

        self._source_error_alerts(errors, now)
        self.store.prune(now, int(self.config["topic_retention_days"]))
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
            "pending_deliveries": pending_count,
            "deadletter_deliveries": deadletter_count,
        }
        self._finish_cycle(stats, now)
        return stats

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
    payload = json.loads(path.read_text(encoding="utf-8"))
    timestamp = datetime.fromisoformat(str(payload["timestamp"]))
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
