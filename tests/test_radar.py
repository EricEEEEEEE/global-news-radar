from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from radar.config import load_config
from radar.delivery import TelegramDelivery
from radar.models import AlertEvent, NewsItem
from radar.policy import assess, is_fresh, is_quiet_window
from radar.render import outbox_manifest, render_p0, render_p1, validate_html
from radar.service import RadarService, check_health
from radar.sources import FmpCollector
from radar.store import RadarStore
from radar.summarizer import Summary
from radar.util import canonicalize_url, hamming_distance, simhash64, visible_length

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "radar.yaml")
NOW = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)


def item(
    title: str,
    *,
    identity: str = "item-1",
    source: str = "Federal Reserve",
    source_tier: str = "primary",
    category_hint: str = "",
    minutes_old: int = 5,
    region: str = "us",
    actual: float | None = None,
    estimate: float | None = None,
    symbol: str = "",
) -> NewsItem:
    return NewsItem(
        identity=identity,
        title=title,
        url=f"https://example.com/{identity}?utm_source=x",
        source=source,
        source_id=source.lower().replace(" ", "_"),
        source_tier=source_tier,
        published_at=NOW - timedelta(minutes=minutes_old),
        fetched_at=NOW,
        region=region,
        category_hint=category_hint,
        actual=actual,
        estimate=estimate,
        symbol=symbol,
    )


class PolicyTests(unittest.TestCase):
    def test_official_fomc_statement_is_p0(self) -> None:
        result = assess(
            item(
                "Federal Reserve issues FOMC statement",
                category_hint="central_bank",
            ),
            CONFIG,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.level, "P0")
        self.assertFalse(result.requires_corroboration)

    def test_secondary_rate_decision_requires_corroboration(self) -> None:
        result = assess(
            item(
                "Fed cuts rates in surprise decision",
                source="Reuters",
                source_tier="secondary",
            ),
            CONFIG,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.requires_corroboration)

    def test_forecast_is_silent(self) -> None:
        result = assess(
            item(
                "Analyst forecast: Fed could cut rates next week",
                source="TipRanks",
                source_tier="secondary",
            ),
            CONFIG,
        )
        self.assertIsNone(result)

    def test_macro_surprise_from_structured_data(self) -> None:
        result = assess(
            item(
                "US Consumer Price Index: actual 3.8, estimate 3.2",
                category_hint="macro",
                source="FMP Economic Calendar",
                source_tier="structured",
                actual=3.8,
                estimate=3.2,
            ),
            CONFIG,
        )
        self.assertIsNotNone(result)
        self.assertEqual((result.level, result.category), ("P0", "macro"))

    def test_macro_small_surprise_is_silent(self) -> None:
        result = assess(
            item(
                "US CPI actual 3.3, estimate 3.2",
                category_hint="macro",
                source="FMP Economic Calendar",
                source_tier="structured",
                actual=3.3,
                estimate=3.2,
            ),
            CONFIG,
        )
        self.assertIsNone(result)

    def test_mega_cap_earnings_surprise(self) -> None:
        result = assess(
            item(
                "TSLA EPS $0.33 vs $0.53 expected",
                source="Reuters",
                source_tier="secondary",
                symbol="TSLA",
            ),
            CONFIG,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "earnings")

    def test_single_merger_is_p1(self) -> None:
        result = assess(
            item(
                "Acme agrees to buy Beta in $12 billion deal",
                source="Reuters",
                source_tier="secondary",
            ),
            CONFIG,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.level, "P1")

    def test_stale_and_future_rejected(self) -> None:
        stale = item("Fed cuts rates", minutes_old=76)
        future = item("Fed cuts rates", minutes_old=-6)
        self.assertEqual(
            is_fresh(
                stale,
                NOW,
                freshness_minutes=75,
                future_tolerance_minutes=5,
            )[1],
            "stale",
        )
        self.assertEqual(
            is_fresh(
                future,
                NOW,
                freshness_minutes=75,
                future_tolerance_minutes=5,
            )[1],
            "future_timestamp",
        )

    def test_quiet_window(self) -> None:
        quiet = datetime(2026, 7, 25, 13, 3, tzinfo=UTC)  # 21:03 SGT
        open_time = datetime(2026, 7, 25, 13, 31, tzinfo=UTC)
        self.assertTrue(is_quiet_window(quiet, "21:00", "21:30"))
        self.assertFalse(is_quiet_window(open_time, "21:00", "21:30"))


class DedupeTests(unittest.TestCase):
    def test_canonical_url_strips_tracking(self) -> None:
        self.assertEqual(
            canonicalize_url("http://www.example.com/a/?utm_source=x&b=2"),
            "https://example.com/a?b=2",
        )

    def test_simhash_is_stable_and_near(self) -> None:
        first = simhash64("Federal Reserve cuts interest rates by 25 basis points")
        second = simhash64("Federal Reserve cuts interest rate by 25 basis points")
        self.assertEqual(
            first, simhash64("Federal Reserve cuts interest rates by 25 basis points")
        )
        self.assertLessEqual(hamming_distance(first, second), 12)

    def test_topic_upgrade_and_material_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RadarStore(Path(directory) / "radar.sqlite3")
            store.record_topic_sent(
                event_key="event:1",
                material_hash="old",
                level="P1",
                source_count=2,
                summary="old",
                now=NOW,
            )
            allowed, reason = store.topic_decision(
                event_key="event:1",
                material_hash="old",
                level="P0",
                now=NOW + timedelta(minutes=5),
                cooldown_hours=6,
            )
            self.assertTrue(allowed)
            self.assertEqual(reason, "severity_upgrade")
            allowed, reason = store.topic_decision(
                event_key="event:1",
                material_hash="new",
                level="P1",
                now=NOW + timedelta(minutes=5),
                cooldown_hours=6,
            )
            self.assertTrue(allowed)
            self.assertEqual(reason, "material_update")
            store.close()

    def test_cross_day_lineage_requires_material_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RadarStore(Path(directory) / "radar.sqlite3")
            store.record_lineage_sent(
                topic_anchor="fed-policy",
                event_key="day-1",
                material_hash="old",
                day_number=1,
                now=NOW,
            )
            allowed, reason, day = store.lineage_decision(
                topic_anchor="fed-policy",
                material_hash="old",
                now=NOW + timedelta(hours=20),
            )
            self.assertFalse(allowed)
            self.assertEqual(reason, "same_fact_24h")
            allowed, reason, day = store.lineage_decision(
                topic_anchor="fed-policy",
                material_hash="new",
                now=NOW + timedelta(days=1, minutes=5),
            )
            self.assertTrue(allowed)
            self.assertEqual(reason, "cross_day_update")
            self.assertEqual(day, 2)
            store.close()


class SourceTests(unittest.TestCase):
    def test_fmp_error_object_is_a_source_failure(self) -> None:
        class Response:
            def json(self):
                return {"Error Message": "invalid key"}

        class Client:
            def get(self, *args, **kwargs):
                return Response()

        collector = FmpCollector(Client(), "bad-key", CONFIG["structured"])
        with self.assertRaisesRegex(RuntimeError, "unexpected payload"):
            collector._get("economic-calendar", NOW)


class RenderTests(unittest.TestCase):
    def _event(self, identity: str, title: str, region: str = "us") -> AlertEvent:
        news_item = item(
            title,
            identity=identity,
            region=region,
            category_hint="central_bank",
        )
        assessment = assess(
            NewsItem(
                **{
                    **news_item.to_dict(),
                    "published_at": news_item.published_at,
                    "fetched_at": news_item.fetched_at,
                }
            ),
            CONFIG,
        )
        self.assertIsNotNone(assessment)
        return AlertEvent(assessment=assessment, items=[news_item])

    def test_p0_mobile_text_contract(self) -> None:
        event = self._event(
            "fed",
            "Federal Reserve issues FOMC statement",
        )
        message = render_p0(
            event,
            Summary(
                "美联储发布决议",
                "美联储发布最新 FOMC 利率决议。",
                "利率与风险资产定价将立即重估。",
            ),
            NOW,
            400,
        )
        validate_html(message.html, 400)
        self.assertLessEqual(visible_length(message.html), 400)
        self.assertIn("18:00 SGT", message.html)
        self.assertIn("发生 17:55", message.html)
        self.assertIn("Federal Reserve", message.html)
        self.assertTrue(message.html.startswith("🚨 <b>美联储发布决议</b>"))
        self.assertNotIn("<b>速报</b>", message.html)
        self.assertIn("<b>即时影响</b>", message.html)
        self.assertIn('<a href="https://example.com/fed?utm_source=x">', message.html)
        self.assertEqual(message.visual_spec["grammar"], "verdict-key-values")
        self.assertTrue(
            all(item["source_path"] for item in message.visual_spec["evidence"])
        )
        self.assertIn("即时影响：利率与风险资产定价将立即重估。", message.plain_text)
        self.assertIn("发生 17:55 · 雷达 18:00 SGT", message.plain_text)
        self.assertNotIn("<", message.plain_text)

    def test_cross_day_p0_shows_day_number(self) -> None:
        event = self._event("fed-day2", "Federal Reserve issues FOMC statement")
        event.assessment.lineage_day = 2
        message = render_p0(
            event,
            Summary("美联储更新决议", "美联储发布新增政策事实。", "定价可能调整。"),
            NOW,
            400,
        )
        self.assertIn("【Day 2】", message.html)
        self.assertEqual(message.visual_spec["headline"], "【Day 2】美联储更新决议")

    def test_p1_requires_two_entries_and_fits(self) -> None:
        one = item(
            "Acme agrees to buy Beta in $12 billion deal",
            identity="m1",
            source="Reuters",
            source_tier="secondary",
            region="us",
        )
        two = item(
            "Gamma raises $2 billion in funding round",
            identity="m2",
            source="Bloomberg",
            source_tier="secondary",
            region="eu",
        )
        assessment_one = assess(one, CONFIG)
        assessment_two = assess(two, CONFIG)
        self.assertIsNotNone(assessment_one)
        self.assertIsNotNone(assessment_two)
        message = render_p1(
            [
                (
                    AlertEvent(assessment_one, [one]),
                    Summary("并购", one.title, "相关资产可能重新定价。"),
                ),
                (
                    AlertEvent(assessment_two, [two]),
                    Summary("融资", two.title, "相关资产可能重新定价。"),
                ),
            ],
            NOW,
            400,
        )
        self.assertEqual(len(message.event_keys), 2)
        self.assertLessEqual(visible_length(message.html), 400)
        self.assertIn("<b>2 条需关注的新动态</b>", message.html)
        self.assertEqual(message.html.count("<blockquote>"), 2)
        self.assertEqual(message.html.count("<a href="), 2)
        self.assertIn("发生 17:55", message.html)
        self.assertIn("雷达 18:00 SGT", message.html)
        self.assertEqual(message.visual_spec["grammar"], "html-digest")
        self.assertIn("Reuters", message.plain_text)

    def test_p1_five_long_cjk_entries_stay_traceable_and_fit(self) -> None:
        entries = []
        for index in range(5):
            current = item(
                "Acme agrees to buy Beta in $12 billion deal " + "超长事实" * 20,
                identity=f"long-{index}",
                source=f"Source {index}",
                source_tier="secondary",
                region="asia",
            )
            assessment = assess(current, CONFIG)
            self.assertIsNotNone(assessment)
            entries.append(
                (
                    AlertEvent(assessment, [current]),
                    Summary("行业动态", current.title, "相关资产可能重新定价。"),
                )
            )
        message = render_p1(entries, NOW, 400)
        self.assertEqual(len(message.event_keys), 5)
        self.assertEqual(len(message.evidence), 5)
        self.assertLessEqual(visible_length(message.html), 400)
        self.assertIn("…", message.html)
        self.assertEqual(
            len(message.visual_spec["evidence"]),
            2 + 4 * len(message.event_keys),
        )

    def test_outbox_manifest_embeds_source_bound_visual_spec(self) -> None:
        event = self._event("manifest", "Federal Reserve issues FOMC statement")
        message = render_p0(
            event,
            Summary("美联储发布决议", "美联储公布最新决定。", "利率预期可能调整。"),
            NOW,
            400,
        )
        manifest = json.loads(
            outbox_manifest(message, {"status": "prepared", "thread_id": "412"})
        )
        self.assertEqual(manifest["version"], "global-news-radar/1.1.0")
        self.assertEqual(manifest["visual_spec"]["selected_modality"], "text")
        self.assertEqual(manifest["visual_spec"]["fallback_chain"], ["text"])


class FakeDelivery:
    def __init__(self, *, fail: bool = False, dry_run: bool = False):
        self.fail = fail
        self.dry_run = dry_run
        self.messages = []
        self.send_enabled = not dry_run

    def send(self, message, thread_id):
        self.messages.append((message, thread_id))
        if self.fail:
            raise RuntimeError("simulated Telegram failure")
        return {
            "ok": True,
            "dry_run": self.dry_run,
            "message_id": "9001" if not self.dry_run else "dry-run",
        }


class TelegramFallbackTests(unittest.TestCase):
    def test_topic_zero_omits_message_thread_id(self) -> None:
        delivery = TelegramDelivery(
            token="test-token",
            chat_id="-1001",
            outbox_dir=Path("/tmp"),
            max_visible_chars=400,
            send_enabled=False,
        )
        without_topic = delivery._payload(thread_id="0", text="test")
        with_topic = delivery._payload(thread_id="412", text="test")
        self.assertNotIn("message_thread_id", without_topic)
        self.assertEqual(with_topic["message_thread_id"], 412)

    def test_html_rejection_falls_back_to_complete_plain_text(self) -> None:
        event = RenderTests()._event(
            "fallback",
            "Federal Reserve issues FOMC statement",
        )
        message = render_p0(
            event,
            Summary("美联储发布决议", "美联储公布最新决定。", "利率预期可能调整。"),
            NOW,
            400,
        )
        rejected = mock.Mock(
            ok=False,
            status_code=400,
            text="Bad Request: can't parse entities",
        )
        rejected.json.return_value = {
            "ok": False,
            "description": "Bad Request: can't parse entities",
        }
        accepted = mock.Mock(ok=True, status_code=200, text="ok")
        accepted.json.return_value = {
            "ok": True,
            "result": {"message_id": 9002},
        }
        with tempfile.TemporaryDirectory() as directory:
            delivery = TelegramDelivery(
                token="test-token",
                chat_id="-1001",
                outbox_dir=Path(directory),
                max_visible_chars=400,
                send_enabled=True,
            )
            with mock.patch(
                "radar.delivery.requests.post",
                side_effect=[rejected, accepted],
            ) as post:
                result = delivery.send(message, "412")
            fallback_payload = post.call_args_list[1].kwargs["json"]
            manifest = json.loads(Path(directory, "latest.json").read_text())
        self.assertEqual(result["fallback"], "plain_text")
        self.assertEqual(fallback_payload["text"], message.plain_text)
        self.assertNotIn("parse_mode", fallback_payload)
        self.assertEqual(manifest["delivery"]["status"], "sent_plain_fallback")
        self.assertEqual(manifest["visible_chars"], visible_length(message.html))


class ServiceIntegrationTests(unittest.TestCase):
    def make_service(self, directory: str, *, baseline: bool = False) -> RadarService:
        config = copy.deepcopy(CONFIG)
        config["official_feeds"] = []
        config["discovery"]["enabled"] = False
        config["structured"]["fmp_enabled"] = False
        config["runtime"]["baseline_on_first_run"] = baseline
        root = Path(directory)
        for name in ("state", "logs", "outbox"):
            root.joinpath(name).mkdir(parents=True, exist_ok=True)
        service = RadarService(root=root, config=config, env={}, send_enabled=False)
        service.delivery = FakeDelivery()
        return service

    def test_first_run_baselines_without_send(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory, baseline=True)
            service.collect = lambda now: (
                [
                    item(
                        "Federal Reserve issues FOMC statement",
                        identity="baseline-fed",
                        category_hint="central_bank",
                    )
                ],
                [],
            )
            stats = service.run_once(NOW)
            self.assertEqual(stats["status"], "baseline")
            self.assertEqual(stats["sent"], 0)
            self.assertTrue(service.store.item_seen("baseline-fed"))
            self.assertEqual(len(service.delivery.messages), 0)
            service.close()

    def test_environment_overrides_public_telegram_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(CONFIG)
            root = Path(directory)
            for name in ("state", "logs", "outbox"):
                root.joinpath(name).mkdir(parents=True, exist_ok=True)
            service = RadarService(
                root=root,
                config=config,
                env={
                    "TELEGRAM_NEWS_THREAD_ID": "77",
                    "TELEGRAM_MONITOR_THREAD_ID": "88",
                },
                send_enabled=False,
            )
            self.assertEqual(service.news_thread_id, "77")
            self.assertEqual(service.monitor_thread_id, "88")
            service.close()

    def test_secondary_p0_waits_for_two_independent_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            first = item(
                "Fed cuts rates in surprise decision",
                identity="fed-reuters",
                source="Reuters",
                source_tier="secondary",
            )
            second = item(
                "Federal Reserve cuts rates in surprise decision",
                identity="fed-bloomberg",
                source="Bloomberg",
                source_tier="secondary",
            )
            service.collect = lambda now: ([first], [])
            self.assertEqual(service.run_once(NOW)["sent"], 0)
            service.collect = lambda now: ([second], [])
            stats = service.run_once(NOW + timedelta(minutes=2))
            self.assertEqual(stats["sent"], 1)
            self.assertEqual(len(service.delivery.messages), 1)
            service.close()

    def test_p1_sends_only_after_second_unique_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            merger = item(
                "Acme agrees to buy Beta in $12 billion deal",
                identity="p1-merger",
                source="Reuters",
                source_tier="secondary",
            )
            financing = item(
                "Gamma raises $2 billion in funding round",
                identity="p1-financing",
                source="Bloomberg",
                source_tier="secondary",
                region="eu",
            )
            service.collect = lambda now: ([merger], [])
            self.assertEqual(service.run_once(NOW)["sent"], 0)
            service.collect = lambda now: ([financing], [])
            stats = service.run_once(NOW + timedelta(minutes=2))
            self.assertEqual(stats["sent"], 1)
            message = service.delivery.messages[0][0]
            self.assertEqual(message.level, "P1")
            self.assertEqual(len(message.event_keys), 2)
            service.close()

    def test_quiet_window_queues_then_delivers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            quiet_now = datetime(2026, 7, 25, 13, 3, tzinfo=UTC)
            service.store.mark_baseline_complete(quiet_now - timedelta(minutes=20))
            fed = item(
                "Federal Reserve issues FOMC statement",
                identity="quiet-fed",
                category_hint="central_bank",
            )
            fed.published_at = quiet_now - timedelta(minutes=2)
            fed.fetched_at = quiet_now
            service.collect = lambda now: ([fed], [])
            stats = service.run_once(quiet_now)
            self.assertTrue(stats["quiet_window"])
            self.assertEqual(stats["sent"], 0)
            self.assertEqual(len(service.delivery.messages), 0)
            service.collect = lambda now: ([], [])
            after = datetime(2026, 7, 25, 13, 31, tzinfo=UTC)
            stats = service.run_once(after)
            self.assertEqual(stats["retries"], 1)
            self.assertEqual(len(service.delivery.messages), 1)
            service.close()

    def test_send_failure_does_not_record_topic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            service.delivery = FakeDelivery(fail=True)
            fed = item(
                "Federal Reserve issues FOMC statement",
                identity="failed-fed",
                category_hint="central_bank",
            )
            service.collect = lambda now: ([fed], [])
            stats = service.run_once(NOW)
            self.assertEqual(stats["sent"], 0)
            topic_count = service.store.connection.execute(
                "SELECT count(*) FROM seen_topics"
            ).fetchone()[0]
            pending_count = service.store.connection.execute(
                "SELECT count(*) FROM pending_delivery"
            ).fetchone()[0]
            self.assertEqual(topic_count, 0)
            self.assertEqual(pending_count, 1)
            service.close()

    def test_health_fails_when_deadletter_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("state").mkdir()
            root.joinpath("state", "heartbeat.json").write_text(
                json.dumps(
                    {
                        "timestamp": NOW.isoformat(),
                        "stats": {"deadletter_deliveries": 1},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("radar.service.utc_now", return_value=NOW):
                healthy, payload = check_health(root, 600)
            self.assertFalse(healthy)
            self.assertEqual(payload["status"], "deadletter")


if __name__ == "__main__":
    unittest.main()
