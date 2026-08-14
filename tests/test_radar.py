from __future__ import annotations

import base64
import copy
import json
import logging
import sys
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from radar import __version__
from radar.briefs import BriefComposer
from radar.clusters import ClusterEngine, tokenize
from radar.comic import CATEGORY_SCENES, HARD_BANS, ComicArtist
from radar.config import load_config
from radar.delivery import TelegramDelivery
from radar.models import AlertEvent, Assessment, NewsItem, RenderedMessage
from radar.policy import assess, is_fresh, is_quiet_window, trending_assessment
from radar.render import outbox_manifest, render_p0, render_p1, validate_html
from radar.service import RadarService, check_health
from radar.sources import FmpCollector, GdeltCollector, discovery_queries
from radar.store import RadarStore
from radar.summarizer import LlmSummarizer, Summary, bare_english_spans
from radar.trending import TrendingJudge
from radar.util import (
    SGT,
    RedactingFormatter,
    canonicalize_url,
    echoable_numbers,
    hamming_distance,
    numeric_values,
    redact_secrets,
    register_secrets,
    simhash64,
    visible_length,
)
from radar.watchdog import run_watchdog

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config" / "radar.yaml")
NOW = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
# Routing assertions read the configured threads: the open-source mirror ships
# placeholder IDs, so hardcoding this deployment's 412/61 fails there.
NEWS_THREAD = str(CONFIG["telegram"]["news_thread_id"])
MONITOR_THREAD = str(CONFIG["telegram"]["monitor_thread_id"])


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


class VersionTests(unittest.TestCase):
    def test_config_and_package_versions_stay_in_sync(self) -> None:
        # The heartbeat, the outbox manifest and the outbound User-Agent all
        # report __version__. When only the config was bumped, a freshly
        # deployed build told the watchdog it was still the old one.
        self.assertEqual(CONFIG["version"], __version__)


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

    def test_month_dated_data_release_is_not_a_forecast(self) -> None:
        result = assess(
            item(
                "US May CPI rises 4.1% vs 3.2% expected, biggest surprise in years",
                source="Reuters",
                source_tier="secondary",
            ),
            CONFIG,
        )
        self.assertIsNotNone(result)
        self.assertEqual((result.level, result.category), ("P0", "macro"))

    def test_modal_speculation_is_still_silent(self) -> None:
        speculative = (
            "Beta may be acquired as Acme agrees to buy rival",
            "Gamma agrees to buy Delta in a deal that could reshape the sector",
            "Acme agrees to buy Beta, though regulators might block it",
        )
        for headline in speculative:
            with self.subTest(headline=headline):
                self.assertIsNone(
                    assess(
                        item(headline, source="Reuters", source_tier="secondary"),
                        CONFIG,
                    )
                )

    def test_macro_anchor_separates_series(self) -> None:
        cpi = assess(
            item(
                "United States Consumer Price Index: actual 3.8, estimate 3.2",
                identity="macro-cpi",
                category_hint="macro",
                source="FMP Economic Calendar",
                source_tier="structured",
                actual=3.8,
                estimate=3.2,
            ),
            CONFIG,
        )
        nfp = assess(
            item(
                "United States Nonfarm Payrolls: actual 120000, estimate 200000",
                identity="macro-nfp",
                category_hint="macro",
                source="FMP Economic Calendar",
                source_tier="structured",
                actual=120000.0,
                estimate=200000.0,
            ),
            CONFIG,
        )
        self.assertIsNotNone(cpi)
        self.assertIsNotNone(nfp)
        self.assertEqual(cpi.entities, nfp.entities)
        self.assertNotEqual(cpi.topic_anchor, nfp.topic_anchor)
        self.assertNotEqual(cpi.event_key, nfp.event_key)

    def test_macro_anchor_is_stable_across_outlet_wording(self) -> None:
        first = assess(
            item(
                "US Consumer Price Index rises 4.1% vs 3.2% expected",
                identity="cpi-reuters",
                source="Reuters",
                source_tier="secondary",
            ),
            CONFIG,
        )
        second = assess(
            item(
                "US Consumer Price Index climbs 4.1% versus 3.2% consensus",
                identity="cpi-bloomberg",
                source="Bloomberg",
                source_tier="secondary",
            ),
            CONFIG,
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.event_key, second.event_key)

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

    def test_company_news_is_rejected(self) -> None:
        # 公司类整体退役：财报、并购、融资都不再是雷达的事。
        for title in (
            "TSLA EPS $0.33 vs $0.53 expected",
            "Acme agrees to buy Beta in $12 billion deal",
            "Gamma raises $2 billion in funding round",
        ):
            self.assertIsNone(
                assess(
                    item(title, source="Reuters", source_tier="secondary"),
                    CONFIG,
                ),
                title,
            )

    def test_minor_economy_macro_is_rejected(self) -> None:
        minor = item(
            "PK Inflation Rate YoY (Jul): actual 9.2, estimate 10.2",
            source="FMP Economic Calendar",
            source_tier="structured",
            category_hint="macro",
            actual=9.2,
            estimate=10.2,
        )
        self.assertIsNone(assess(minor, CONFIG))
        major = item(
            "US Inflation Rate YoY (Jul): actual 9.2, estimate 10.2",
            source="FMP Economic Calendar",
            source_tier="structured",
            category_hint="macro",
            actual=9.2,
            estimate=10.2,
        )
        result = assess(major, CONFIG)
        self.assertIsNotNone(result)
        self.assertEqual(result.category, "macro")

    def test_minor_economy_headline_macro_is_rejected(self) -> None:
        self.assertIsNone(
            assess(
                item(
                    "Pakistan CPI rises 9.2% vs 10.2% expected",
                    source="Reuters",
                    source_tier="secondary",
                ),
                CONFIG,
            )
        )

    def test_world_events_are_p0_with_corroboration(self) -> None:
        cases = {
            "Magnitude 7.8 earthquake strikes off the coast": "disaster",
            "Army seizes power in coup, leader detained": "political_crisis",
            "WHO declares public health emergency over new virus": ("health_emergency"),
        }
        for title, category in cases.items():
            result = assess(
                item(title, source="BBC", source_tier="secondary"),
                CONFIG,
            )
            self.assertIsNotNone(result, title)
            self.assertEqual(result.category, category)
            self.assertEqual(result.level, "P0")
            self.assertTrue(result.requires_corroboration)

    def test_recoup_is_not_a_coup(self) -> None:
        self.assertIsNone(
            assess(
                item(
                    "Investors recoup losses as markets stabilise",
                    source="Reuters",
                    source_tier="secondary",
                ),
                CONFIG,
            )
        )

    def test_trending_assessment_shape(self) -> None:
        picked = item(
            "Rare comet visible worldwide this weekend",
            source="BBC",
            source_tier="secondary",
        )
        result = trending_assessment(picked)
        self.assertEqual(result.level, "P1")
        self.assertEqual(result.category, "trending")
        self.assertTrue(result.event_key.startswith("trending:"))
        self.assertFalse(result.requires_corroboration)

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

    def test_stale_lineage_restarts_at_day_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RadarStore(Path(directory) / "radar.sqlite3")
            store.record_lineage_sent(
                topic_anchor="fed-policy",
                event_key="old",
                material_hash="old",
                day_number=1,
                now=NOW,
            )
            allowed, reason, day = store.lineage_decision(
                topic_anchor="fed-policy",
                material_hash="new",
                now=NOW + timedelta(days=42),
                max_gap_days=7,
            )
            self.assertTrue(allowed)
            self.assertEqual(reason, "lineage_expired")
            self.assertEqual(day, 1)
            store.record_lineage_sent(
                topic_anchor="fed-policy",
                event_key="new",
                material_hash="new",
                day_number=day,
                now=NOW + timedelta(days=42),
            )
            allowed, reason, day = store.lineage_decision(
                topic_anchor="fed-policy",
                material_hash="newer",
                now=NOW + timedelta(days=43, minutes=5),
                max_gap_days=7,
            )
            self.assertEqual((allowed, reason, day), (True, "cross_day_update", 2))
            store.close()


class RedactionTests(unittest.TestCase):
    def test_query_string_credentials_are_masked(self) -> None:
        url = "https://financialmodelingprep.com/stable/x?from=1&apikey=abcd1234efgh"
        cleaned = redact_secrets(f"401 Client Error for url: {url}")
        self.assertNotIn("abcd1234efgh", cleaned)
        self.assertIn("apikey=***", cleaned)

    def test_registered_secret_is_masked_anywhere_it_appears(self) -> None:
        register_secrets(["s3cret-token-value-01"])
        cleaned = redact_secrets("bearer s3cret-token-value-01 rejected")
        self.assertNotIn("s3cret-token-value-01", cleaned)
        self.assertIn("***", cleaned)

    def test_bot_token_in_telegram_url_is_masked(self) -> None:
        cleaned = redact_secrets("POST https://api.telegram.org/bot12345:AAH-xyz/send")
        self.assertNotIn("AAH-xyz", cleaned)

    def test_formatter_scrubs_the_traceback_too(self) -> None:
        register_secrets(["traceback-secret-9911"])
        formatter = RedactingFormatter("%(message)s")
        try:
            raise RuntimeError("failed with key traceback-secret-9911")
        except RuntimeError:
            record = logging.LogRecord(
                "radar", logging.ERROR, __file__, 1, "boom", None, sys.exc_info()
            )
        rendered = formatter.format(record)
        self.assertIn("RuntimeError", rendered)
        self.assertNotIn("traceback-secret-9911", rendered)


class StoreGuardTests(unittest.TestCase):
    def _store(self, directory: str) -> RadarStore:
        return RadarStore(Path(directory) / "state" / "radar.sqlite3")

    def _message(self, *event_keys: str, content_hash: str = "hash-1"):
        return RenderedMessage(
            level="P0",
            html="<b>x</b>",
            plain_text="x",
            event_keys=list(event_keys),
            content_hash=content_hash,
            evidence=[],
            created_at=NOW,
        )

    def test_queued_event_is_not_queued_a_second_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.queue_delivery(
                self._message("central_bank:abc"),
                thread_id="412",
                next_retry_at=NOW,
                error="quiet_window",
            )
            self.assertTrue(store.event_key_pending("central_bank:abc"))
            allowed, reason = store.topic_decision(
                event_key="central_bank:abc",
                material_hash="m2",
                level="P0",
                now=NOW,
                cooldown_hours=6.0,
            )
            self.assertFalse(allowed)
            self.assertEqual(reason, "already_queued")
            store.close()

    def test_anchor_holds_a_minimum_interval_between_sends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.record_lineage_sent(
                topic_anchor="geopolitics:strike:IRAN",
                event_key="geopolitics:day1",
                material_hash="m1",
                day_number=1,
                now=NOW,
            )
            allowed, reason, _ = store.lineage_decision(
                topic_anchor="geopolitics:strike:IRAN",
                material_hash="m2",
                now=NOW + timedelta(minutes=10),
                max_gap_days=7.0,
                min_interval_minutes=30.0,
                daily_cap=6,
            )
            self.assertFalse(allowed)
            self.assertEqual(reason, "anchor_min_interval")
            allowed, _, day = store.lineage_decision(
                topic_anchor="geopolitics:strike:IRAN",
                material_hash="m3",
                now=NOW + timedelta(minutes=45),
                max_gap_days=7.0,
                min_interval_minutes=30.0,
                daily_cap=6,
            )
            self.assertTrue(allowed)
            self.assertEqual(day, 1)
            store.close()

    def test_anchor_daily_cap_stops_a_runaway_story(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            anchor = "geopolitics:strike:IRAN"
            moment = NOW
            for index in range(3):
                store.record_lineage_sent(
                    topic_anchor=anchor,
                    event_key=f"geopolitics:day{index}",
                    material_hash=f"m{index}",
                    day_number=1,
                    now=moment,
                )
                moment += timedelta(minutes=40)
            allowed, reason, _ = store.lineage_decision(
                topic_anchor=anchor,
                material_hash="m-next",
                now=moment,
                max_gap_days=7.0,
                min_interval_minutes=30.0,
                daily_cap=3,
            )
            self.assertFalse(allowed)
            self.assertEqual(reason, "anchor_daily_cap")
            store.close()

    def test_export_writes_exactly_one_line_per_event_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            message = self._message("macro:usa:cpi", "macro:usa:gdp")
            store.record_delivery(
                message=message, message_id="1", visible_chars=10, now=NOW
            )
            for key in message.event_keys:
                store.record_topic_sent(
                    event_key=key,
                    material_hash="m",
                    level="P0",
                    source_count=2,
                    summary=f"summary for {key}",
                    now=NOW,
                )
            output = Path(directory) / "state" / "radar_events.jsonl"
            count = store.export_deliveries(output)
            lines = output.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(count, 2)
            self.assertEqual(len(lines), 2)
            keys = {json.loads(line)["event_key"] for line in lines}
            self.assertEqual(keys, set(message.event_keys))
            # Each key must carry its own summary; the old substring join
            # paired keys with whichever topic row happened to match.
            pairs = {
                json.loads(line)["event_key"]: json.loads(line)["last_summary"]
                for line in lines
            }
            self.assertEqual(
                pairs,
                {
                    "macro:usa:cpi": "summary for macro:usa:cpi",
                    "macro:usa:gdp": "summary for macro:usa:gdp",
                },
            )
            store.close()

    def test_export_excludes_infrastructure_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(directory)
            store.record_delivery(
                message=self._message("source-error:ecb_press"),
                message_id="1",
                visible_chars=10,
                now=NOW,
            )
            # Recovery notices ride the same delivery path and must stay out of
            # the market-event ledger too.
            store.record_delivery(
                message=self._message(
                    "source-recovery:ecb_press", content_hash="hash-2"
                ),
                message_id="2",
                visible_chars=10,
                now=NOW,
            )
            output = Path(directory) / "state" / "radar_events.jsonl"
            self.assertEqual(store.export_deliveries(output), 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "")
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
            "Rare comet visible worldwide this weekend",
            identity="m1",
            source="Reuters",
            source_tier="secondary",
            region="us",
        )
        two = item(
            "Lost Rembrandt painting found in attic",
            identity="m2",
            source="Bloomberg",
            source_tier="secondary",
            region="eu",
        )
        assessment_one = trending_assessment(one)
        assessment_two = trending_assessment(two)
        self.assertIsNotNone(assessment_one)
        self.assertIsNotNone(assessment_two)
        message = render_p1(
            [
                (
                    AlertEvent(assessment_one, [one]),
                    Summary("热点", one.title, "全球关注度高，值得了解。"),
                ),
                (
                    AlertEvent(assessment_two, [two]),
                    Summary("热点", two.title, "全球关注度高，值得了解。"),
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
                "Rare deep-sea creature filmed alive " + "超长事实" * 20,
                identity=f"long-{index}",
                source=f"Source {index}",
                source_tier="secondary",
                region="asia",
            )
            assessment = trending_assessment(current)
            self.assertIsNotNone(assessment)
            entries.append(
                (
                    AlertEvent(assessment, [current]),
                    Summary("热点", current.title, "全球关注度高，值得了解。"),
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
        self.assertEqual(manifest["version"], f"global-news-radar/{__version__}")
        self.assertEqual(manifest["visual_spec"]["selected_modality"], "text")
        self.assertEqual(manifest["visual_spec"]["fallback_chain"], ["text"])


class FakeJudge:
    """Deterministic stand-in for TrendingJudge: picks everything offered."""

    def __init__(
        self,
        max_per_day: int = 3,
        *,
        picks: bool = True,
        burst_verdict: bool | None = True,
    ):
        self.enabled = True
        self.max_per_day = max_per_day
        self.picks = picks
        self.burst_verdict = burst_verdict
        self.offered: list[list[NewsItem]] = []
        self.remaining_seen: list[int] = []
        self.burst_calls: list[list[tuple[str, str]]] = []

    def pick(self, candidates: list[NewsItem], remaining: int) -> list[NewsItem]:
        self.offered.append(list(candidates))
        self.remaining_seen.append(remaining)
        if not self.picks:
            return []
        return list(candidates)[: max(0, min(remaining, 2))]

    def judge_burst(self, headlines: list[tuple[str, str]]) -> bool | None:
        self.burst_calls.append(list(headlines))
        return self.burst_verdict

    def stats(self) -> dict[str, int]:
        return {"picked": 0, "failed": 0}


class FakeDelivery:
    def __init__(self, *, fail: bool = False, dry_run: bool = False):
        self.fail = fail
        self.dry_run = dry_run
        self.messages = []
        self.send_enabled = not dry_run

    def send(self, message, thread_id, photo=None):
        self.messages.append((message, thread_id))
        self.photos = getattr(self, "photos", [])
        self.photos.append(photo)
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
                [],
            )
            stats = service.run_once(NOW)
            self.assertEqual(stats["status"], "baseline")
            self.assertEqual(stats["sent"], 0)
            self.assertTrue(service.store.item_seen("baseline-fed"))
            self.assertEqual(len(service.delivery.messages), 0)
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
            service.collect = lambda now: ([first], [], [])
            self.assertEqual(service.run_once(NOW)["sent"], 0)
            service.collect = lambda now: ([second], [], [])
            stats = service.run_once(NOW + timedelta(minutes=2))
            self.assertEqual(stats["sent"], 1)
            self.assertEqual(len(service.delivery.messages), 1)
            service.close()

    def test_p1_sends_only_after_second_unique_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            # Slot quota is max_per_day // 3; 6 leaves room for two picks in
            # one 8-hour slot so the second unique event can join the batch.
            service.trending = FakeJudge(max_per_day=6)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            comet = item(
                "Rare comet visible worldwide this weekend",
                identity="p1-comet",
                source="Reuters",
                source_tier="secondary",
            )
            painting = item(
                "Lost Rembrandt painting found in attic",
                identity="p1-painting",
                source="Bloomberg",
                source_tier="secondary",
                region="eu",
            )
            service.collect = lambda now: ([comet], [], [])
            self.assertEqual(service.run_once(NOW)["sent"], 0)
            service.collect = lambda now: ([painting], [], [])
            stats = service.run_once(NOW + timedelta(minutes=2))
            self.assertEqual(stats["sent"], 1)
            message = service.delivery.messages[0][0]
            self.assertEqual(message.level, "P1")
            self.assertEqual(len(message.event_keys), 2)
            service.close()

    def test_lone_p1_sends_after_solo_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.trending = FakeJudge()
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            comet = item(
                "Rare comet visible worldwide this weekend",
                identity="solo-comet",
                source="Reuters",
                source_tier="secondary",
            )
            service.collect = lambda now: ([comet], [], [])
            self.assertEqual(service.run_once(NOW)["sent"], 0)
            service.collect = lambda now: ([], [], [])
            later = NOW + timedelta(minutes=61)
            stats = service.run_once(later)
            self.assertEqual(stats["sent"], 1)
            message = service.delivery.messages[0][0]
            self.assertEqual(message.level, "P1")
            self.assertEqual(len(message.event_keys), 1)
            service.close()

    def test_trending_slot_cap_limits_buffering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            judge = FakeJudge()
            service.trending = judge
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            local = NOW.astimezone(SGT)
            slot_key = f"{local.date().isoformat()}:{local.hour // 8}"
            service.store.set_meta("trending_slot", slot_key, NOW)
            service.store.set_meta("trending_slot_count", "1", NOW)
            comet = item(
                "Rare comet visible worldwide this weekend",
                identity="cap-comet",
                source="Reuters",
                source_tier="secondary",
            )
            service.collect = lambda now: ([comet], [], [])
            service.run_once(NOW)
            # Quota for max_per_day=3 is one per 8-hour slot, already spent:
            # the judge is never even consulted.
            self.assertEqual(judge.offered, [])
            self.assertEqual(
                len(service.store.ready_p1(NOW, NOW - timedelta(minutes=120))),
                0,
            )
            service.close()

    def test_non_major_source_not_offered_to_judge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            judge = FakeJudge()
            service.trending = judge
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            obscure = item(
                "Rare comet visible worldwide this weekend",
                identity="obscure-comet",
                source="Random Blog",
                source_tier="secondary",
            )
            service.collect = lambda now: ([obscure], [], [])
            service.run_once(NOW)
            self.assertEqual(judge.offered, [])
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
            service.collect = lambda now: ([fed], [], [])
            stats = service.run_once(quiet_now)
            self.assertTrue(stats["quiet_window"])
            self.assertEqual(stats["sent"], 0)
            self.assertEqual(len(service.delivery.messages), 0)
            service.collect = lambda now: ([], [], [])
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
            service.collect = lambda now: ([fed], [], [])
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

    def test_source_error_alert_escapes_raw_error_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            broken = {
                "source_id": "ecb_press",
                "error": (
                    "HTTPSConnectionPool(host='www.ecb.europa.eu'): "
                    "<urlopen error [Errno 8] & connection reset>"
                ),
                "failures": 3,
                "tier": "primary",
            }
            service.collect = lambda now: ([], [broken], [])
            service.run_once(NOW)
            self.assertEqual(len(service.delivery.messages), 1)
            message, thread_id = service.delivery.messages[0]
            self.assertEqual(thread_id, MONITOR_THREAD)
            validate_html(message.html, 4096)
            self.assertIn("&lt;urlopen error", message.html)
            service.close()

    def test_quiet_window_end_follows_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.config["quiet_window"]["end"] = "22:00"
            quiet_now = datetime(2026, 7, 25, 13, 3, tzinfo=UTC)  # 21:03 SGT
            service.store.mark_baseline_complete(quiet_now - timedelta(minutes=20))
            fed = item(
                "Federal Reserve issues FOMC statement",
                identity="quiet-config-fed",
                category_hint="central_bank",
            )
            fed.published_at = quiet_now - timedelta(minutes=2)
            fed.fetched_at = quiet_now
            service.collect = lambda now: ([fed], [], [])
            self.assertTrue(service.run_once(quiet_now)["quiet_window"])
            row = service.store.connection.execute(
                "SELECT next_retry_at FROM pending_delivery"
            ).fetchone()
            next_retry = datetime.fromisoformat(str(row["next_retry_at"]))
            self.assertEqual(next_retry.astimezone(SGT).strftime("%H:%M"), "22:00")
            service.close()

    def test_two_outlets_corroborate_despite_different_event_keys(self) -> None:
        # Same headline, different standfirst: the extra countries in the second
        # summary move the event_key, so exact-key matching alone lost the
        # confirmation and the P0 never fired.
        headline = "Israel launches air strike on Iran nuclear site"
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            first = item(
                headline,
                identity="strike-reuters",
                source="Reuters",
                source_tier="secondary",
                region="global",
            )
            second = item(
                headline,
                identity="strike-cnbc",
                source="CNBC",
                source_tier="secondary",
                region="global",
            )
            second.summary = "Saudi Arabia and Turkey condemned the overnight raid."
            self.assertNotEqual(
                assess(first, service.config).event_key,
                assess(second, service.config).event_key,
            )
            service.collect = lambda now: ([first], [], [])
            self.assertEqual(service.run_once(NOW)["sent"], 0)
            service.collect = lambda now: ([second], [], [])
            stats = service.run_once(NOW + timedelta(minutes=2))
            self.assertEqual(stats["sent"], 1)
            self.assertEqual(service.delivery.messages[0][1], NEWS_THREAD)
            service.close()

    def test_gdelt_fallback_runs_on_a_thin_discovery_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory, baseline=True)
            service.config["discovery"]["enabled"] = True
            service.config["discovery"]["feeds"] = []
            service.config["discovery"]["gdelt_min_items"] = 8
            thin = [
                item(
                    "Markets open higher",
                    identity="thin-1",
                    source="Google News",
                    source_tier="secondary",
                )
            ]
            called: list[str] = []

            def gdelt(now, now_sgt):
                called.append("gdelt")
                return []

            service.gnews.collect = lambda now, now_sgt: thin
            service.gdelt.collect = gdelt
            service.run_once(NOW)
            self.assertEqual(called, ["gdelt"])
            service.close()

    def test_full_discovery_batch_skips_the_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory, baseline=True)
            service.config["discovery"]["enabled"] = True
            service.config["discovery"]["feeds"] = []
            service.config["discovery"]["gdelt_min_items"] = 3
            batch = [
                item(
                    f"Markets headline {index}",
                    identity=f"full-{index}",
                    source="Google News",
                    source_tier="secondary",
                )
                for index in range(5)
            ]
            called: list[str] = []
            service.gnews.collect = lambda now, now_sgt: batch
            service.gdelt.collect = lambda now, now_sgt: called.append("gdelt") or []
            service.run_once(NOW)
            self.assertEqual(called, [])
            service.close()

    def test_source_failure_never_leaks_the_api_key(self) -> None:
        key = "FMPKEY8f3ac91d55e24b17"
        register_secrets([key])
        url = f"https://financialmodelingprep.com/stable/x?from=1&apikey={key}"
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))

            def boom():
                raise RuntimeError(f"401 Client Error for url: {url}")

            def collect(now):
                errors: list[dict] = []
                for _ in range(3):
                    service._source_call("fmp_macro", boom, now, errors, tier="primary")
                return [], errors, []

            service.collect = collect
            stats = service.run_once(NOW)
            message, thread_id = service.delivery.messages[0]
            dump = "\n".join(service.store.connection.iterdump())
            heartbeat = Path(directory, "state", "heartbeat.json").read_text("utf-8")
            self.assertEqual(thread_id, MONITOR_THREAD)
            self.assertNotIn(key, message.html)
            self.assertNotIn(key, message.plain_text)
            self.assertNotIn(key, json.dumps(stats, ensure_ascii=False))
            self.assertNotIn(key, dump)
            self.assertNotIn(key, heartbeat)
            self.assertIn("apikey=***", message.plain_text)
            service.close()

    def test_flapping_source_alerts_before_three_consecutive_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            outcomes = [True, False, True, False, True]

            def collect(now):
                errors: list[dict] = []
                for fails in outcomes:

                    def call(fails=fails):
                        if fails:
                            raise RuntimeError("gateway timeout")
                        return []

                    service._source_call("ecb_press", call, now, errors, tier="primary")
                return [], errors, []

            service.collect = collect
            service.run_once(NOW)
            self.assertEqual(len(service.delivery.messages), 1)
            message, thread_id = service.delivery.messages[0]
            self.assertEqual(thread_id, MONITOR_THREAD)
            self.assertIn("60%", message.plain_text)
            service.close()

    def _drive_source(
        self,
        service: RadarService,
        source_id: str,
        outcomes: list[bool],
        tier: str,
        start: datetime,
        step_minutes: int = 10,
    ) -> None:
        """Run one cycle per outcome, the way a real poll loop would.

        Health state only advances per cycle, so failures spread across cycles
        are what the thresholds actually measure. Collapsing them into a single
        cycle would test something the daemon never does.
        """
        for offset, fails in enumerate(outcomes):

            def collect(now, fails=fails):
                errors: list[dict] = []
                recoveries: list[dict] = []

                def call():
                    if fails:
                        raise RuntimeError("502 Server Error: Bad Gateway")
                    return []

                service._source_call(source_id, call, now, errors, recoveries, tier)
                return [], errors, recoveries

            service.collect = collect
            service.run_once(start + timedelta(minutes=step_minutes * offset))

    def test_discovery_source_stays_quiet_until_its_own_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            # Discovery runs every ten minutes, so the old shared threshold of 3
            # interrupted after half an hour of one commercial CDN misbehaving.
            self._drive_source(service, "investing_news", [True] * 3, "secondary", NOW)
            self.assertEqual(service.delivery.messages, [])
            self._drive_source(
                service,
                "investing_news",
                [True] * 3,
                "secondary",
                NOW + timedelta(minutes=30),
            )
            self.assertEqual(len(service.delivery.messages), 1)
            self.assertIn("连续失败 6 次", service.delivery.messages[0][0].plain_text)
            service.close()

    def test_primary_source_still_alerts_at_three(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            self._drive_source(service, "fed_all", [True] * 3, "primary", NOW)
            self.assertEqual(len(service.delivery.messages), 1)
            message, thread_id = service.delivery.messages[0]
            self.assertEqual(thread_id, MONITOR_THREAD)
            self.assertIn("连续失败 3 次", message.plain_text)
            service.close()

    def test_recovery_closes_an_announced_outage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            # Fail at +0/+10/+20 (alert on the third), recover at +30.
            self._drive_source(
                service, "fed_all", [True, True, True, False], "primary", NOW
            )
            self.assertEqual(len(service.delivery.messages), 2)
            recovery, thread_id = service.delivery.messages[1]
            self.assertEqual(thread_id, MONITOR_THREAD)
            self.assertIn("fed_all 已恢复", recovery.plain_text)
            self.assertIn("中断 30 分钟", recovery.plain_text)
            self.assertEqual(recovery.event_keys, ["source-recovery:fed_all"])
            service.close()

    def test_recovery_is_silent_when_the_outage_was_never_announced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            # Two failures never reach the primary threshold, so nobody was told
            # the source was down and nobody needs to be told it came back.
            self._drive_source(service, "fed_all", [True, True, False], "primary", NOW)
            self.assertEqual(service.delivery.messages, [])
            service.close()

    def test_a_new_outage_after_recovery_alerts_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            self._drive_source(
                service, "fed_all", [True, True, True, False], "primary", NOW
            )
            self.assertEqual(len(service.delivery.messages), 2)
            # The twelve-hour cooldown exists to silence repeats inside one
            # outage. Once the incident was announced as recovered, the next
            # outage is a new incident and must not be swallowed for a day.
            self._drive_source(
                service,
                "fed_all",
                [True, True, True],
                "primary",
                NOW + timedelta(minutes=120),
            )
            self.assertEqual(len(service.delivery.messages), 3)
            service.close()

    def test_a_bouncing_source_cannot_realert_inside_the_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            self._drive_source(
                service, "fed_all", [True, True, True, False], "primary", NOW
            )
            self.assertEqual(len(service.delivery.messages), 2)
            # Recovery landed at +30; source_realert_minutes is 60, so an outage
            # that completes at +50 is still inside the gap.
            self._drive_source(
                service,
                "fed_all",
                [True, True, True],
                "primary",
                NOW + timedelta(minutes=40),
                step_minutes=5,
            )
            self.assertEqual(len(service.delivery.messages), 2)
            service.close()

    def test_flapping_discovery_source_respects_its_tier_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            # 3 failures in 5 calls is a 60% rate, over the flapping threshold.
            # Without a tier-aware failure floor the rate trigger would page at
            # 3 failures and quietly undo the discovery threshold of 6.
            self._drive_source(
                service,
                "investing_news",
                [True, False, True, False, True],
                "secondary",
                NOW,
            )
            self.assertEqual(service.delivery.messages, [])
            service.close()

    def test_upgrade_does_not_announce_a_pre_upgrade_outage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            # Exactly the state this deployment inherited: alerted yesterday,
            # healthy again, and no recovered_at because the column is new.
            service.store.connection.execute(
                """
                INSERT INTO source_health(
                    source_id, consecutive_failures, last_success_at, last_alert_at
                ) VALUES('investing_news', 0, ?, ?)
                """,
                (
                    (NOW - timedelta(hours=1)).isoformat(),
                    (NOW - timedelta(hours=12)).isoformat(),
                ),
            )
            service.store.connection.execute(
                "UPDATE source_health SET recovered_at=NULL WHERE source_id=?",
                ("investing_news",),
            )
            service.store.connection.commit()
            service.store._migrate()
            # One failure, then a success: the old alert must not resurface.
            self._drive_source(
                service, "investing_news", [True, False], "secondary", NOW
            )
            self.assertEqual(service.delivery.messages, [])
            service.close()

    def test_readonly_cycle_consumes_no_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            fed = item(
                "Federal Reserve issues FOMC statement",
                identity="readonly-fed",
                category_hint="central_bank",
            )
            service.collect = lambda now: ([fed], [], [])
            stats = service.run_once(NOW, readonly=True)
            self.assertEqual(stats["status"], "readonly")
            self.assertEqual(stats["eligible"], 1)
            self.assertEqual(stats["preview"][0]["level"], "P0")
            for table in ("seen_items", "observations", "seen_topics", "deliveries"):
                count = service.store.connection.execute(
                    f"SELECT count(*) FROM {table}"  # noqa: S608
                ).fetchone()[0]
                self.assertEqual(count, 0, table)
            self.assertEqual(service.store.get_meta("cycle_count", "0"), "0")
            self.assertEqual(len(service.delivery.messages), 0)
            self.assertFalse(Path(directory, "state", "heartbeat.json").exists())
            service.close()

    def test_health_rejects_a_corrupt_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("state").mkdir()
            root.joinpath("state", "heartbeat.json").write_text(
                '{"timestamp": "2026-07-25T10', encoding="utf-8"
            )
            healthy, payload = check_health(root, 600)
            self.assertFalse(healthy)
            self.assertEqual(payload["status"], "unreadable_heartbeat")

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

    def _service(self, root: Path, env: dict[str, str]) -> RadarService:
        for name in ("state", "logs", "outbox"):
            root.joinpath(name).mkdir(parents=True, exist_ok=True)
        return RadarService(
            root=root,
            config=copy.deepcopy(CONFIG),
            env=env,
            send_enabled=False,
        )

    def test_blank_env_value_falls_back_to_config(self) -> None:
        # .env.example ships blank lines to fill in, and load_env stores those
        # as "". A blank line must mean "unset", not "send an empty route".
        with tempfile.TemporaryDirectory() as directory:
            service = self._service(
                Path(directory),
                {
                    "TELEGRAM_NEWS_THREAD_ID": "",
                    "TELEGRAM_MONITOR_THREAD_ID": "",
                    "RADAR_USER_AGENT": "",
                },
            )
            self.assertEqual(
                service.news_thread_id, str(CONFIG["telegram"]["news_thread_id"])
            )
            self.assertEqual(
                service.monitor_thread_id, str(CONFIG["telegram"]["monitor_thread_id"])
            )
            self.assertIn(
                f"global-news-radar/{__version__}",
                service.client.session.headers["User-Agent"],
            )
            service.close()

    def test_environment_enables_optional_collectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(CONFIG["llm"]["enabled"])
            self.assertFalse(CONFIG["structured"]["fmp_enabled"])
            service = self._service(
                root,
                {
                    "RADAR_LLM_ENABLED": "true",
                    "RADAR_FMP_ENABLED": "1",
                    "LLM_API_BASE": "http://127.0.0.1:8317/v1",
                    "LLM_API_KEY": "key",
                    "LLM_MODEL": "model",
                },
            )
            self.assertTrue(service.summarizer.enabled)
            self.assertTrue(service.fmp_enabled)
            service.close()

            off = self._service(root, {"RADAR_LLM_ENABLED": "off"})
            self.assertFalse(off.summarizer.enabled)
            self.assertFalse(off.fmp_enabled)
            off.close()

    def test_unparseable_toggle_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                self._service(Path(directory), {"RADAR_FMP_ENABLED": "maybe"})

    def test_legacy_llm_environment_names_still_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self._service(
                Path(directory),
                {
                    "RADAR_LLM_ENABLED": "yes",
                    "FE_LLM_API_BASE": "http://127.0.0.1:8317/v1",
                    "FE_LLM_API_KEY": "legacy-key",
                    "FE_LLM_MODEL": "legacy-model",
                },
            )
            self.assertEqual(service.summarizer.base_url, "http://127.0.0.1:8317/v1")
            self.assertEqual(service.summarizer.model, "legacy-model")
            self.assertEqual(redact_secrets("legacy-key leaked"), "*** leaked")
            service.close()


class WatchdogTests(unittest.TestCase):
    """The watchdog is what reports a daemon that cannot report itself."""

    def _heartbeat(
        self, root: Path, *, age_seconds: float, deadletters: int = 0
    ) -> None:
        root.joinpath("state").mkdir(parents=True, exist_ok=True)
        stamp = NOW - timedelta(seconds=age_seconds)
        root.joinpath("state", "heartbeat.json").write_text(
            json.dumps(
                {
                    "timestamp": stamp.isoformat(),
                    "stats": {"deadletter_deliveries": deadletters},
                }
            ),
            encoding="utf-8",
        )

    def _run(
        self,
        root: Path,
        *,
        now: datetime = NOW,
        env: dict[str, str] | None = None,
    ) -> tuple[bool, dict, mock.Mock]:
        sent = mock.Mock(
            return_value={"message_id": "1", "fallback": None},
        )
        with mock.patch("radar.watchdog.post_telegram", sent):
            healthy, report = run_watchdog(
                root,
                CONFIG,
                {"TELEGRAM_BOT_TOKEN": "wd-token", **(env or {})},
                max_age_seconds=900,
                send_enabled=True,
                now=now,
            )
        return healthy, report, sent

    def test_healthy_daemon_stays_silent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._heartbeat(root, age_seconds=60)
            healthy, report, sent = self._run(root)
        self.assertTrue(healthy)
        self.assertFalse(report["alerted"])
        sent.assert_not_called()

    def test_stale_heartbeat_alerts_once_then_respects_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._heartbeat(root, age_seconds=4000)
            healthy, first, sent = self._run(root)
            self.assertFalse(healthy)
            self.assertTrue(first["alerted"])
            body = sent.call_args.kwargs["html_text"]
            # A second cron run an hour later must not re-alarm: a dead daemon
            # stays dead, and repeating it only teaches the operator to ignore it.
            _, second, resent = self._run(root, now=NOW + timedelta(hours=1))
            self.assertFalse(second["alerted"])
            resent.assert_not_called()
            # Past the cooldown the outage is worth restating.
            _, third, third_send = self._run(root, now=NOW + timedelta(hours=7))
        self.assertIn("新闻雷达停摆", body)
        self.assertIn("心跳过期", body)
        self.assertTrue(third["alerted"])
        third_send.assert_called_once()

    def test_recovery_closes_an_announced_outage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._heartbeat(root, age_seconds=4000)
            self._run(root)
            later = NOW + timedelta(minutes=42)
            self._heartbeat(root, age_seconds=-2520 + 60)  # fresh again at `later`
            healthy, report, sent = self._run(root, now=later)
            body = sent.call_args.kwargs["html_text"]
        self.assertTrue(healthy)
        self.assertTrue(report["recovered"])
        self.assertEqual(report["outage_minutes"], 42)
        self.assertIn("新闻雷达恢复", body)
        self.assertIn("42", body)

    def test_blip_that_never_alerted_sends_no_lone_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._heartbeat(root, age_seconds=60)
            healthy, report, sent = self._run(root)
        self.assertTrue(healthy)
        self.assertFalse(report["recovered"])
        sent.assert_not_called()

    def test_deadletters_are_reported_even_with_a_fresh_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._heartbeat(root, age_seconds=60, deadletters=2)
            healthy, report, sent = self._run(root)
            body = sent.call_args.kwargs["html_text"]
        self.assertFalse(healthy)
        self.assertEqual(report["status"], "deadletter")
        self.assertIn("死信 2 条", body)

    def test_missing_heartbeat_alerts_to_the_monitor_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("state").mkdir(parents=True)
            healthy, report, sent = self._run(root)
            kwargs = sent.call_args.kwargs
        self.assertFalse(healthy)
        self.assertEqual(report["status"], "missing_heartbeat")
        # Blank env means "not set" here exactly as it does for the daemon, so
        # the two cannot drift into routing infra alerts to different topics.
        self.assertEqual(kwargs["thread_id"], MONITOR_THREAD)
        self.assertEqual(kwargs["chat_id"], str(CONFIG["telegram"]["chat_id"]))

    def test_environment_overrides_the_monitor_route(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("state").mkdir(parents=True)
            _, _, sent = self._run(
                root,
                env={"TELEGRAM_CHAT_ID": "-100777", "TELEGRAM_MONITOR_THREAD_ID": "61"},
            )
            kwargs = sent.call_args.kwargs
        self.assertEqual(kwargs["chat_id"], "-100777")
        self.assertEqual(kwargs["thread_id"], "61")

    def test_a_failed_alert_is_retried_next_run(self) -> None:
        # Recording the alert before Telegram confirmed it would swallow the
        # only warning the operator gets for the whole cooldown.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._heartbeat(root, age_seconds=4000)
            broken = mock.Mock(side_effect=RuntimeError("telegram down"))
            with mock.patch("radar.watchdog.post_telegram", broken):
                _, failed = run_watchdog(
                    root,
                    CONFIG,
                    {"TELEGRAM_BOT_TOKEN": "wd-token"},
                    max_age_seconds=900,
                    send_enabled=True,
                    now=NOW,
                )
            _, retried, sent = self._run(root, now=NOW + timedelta(minutes=5))
        self.assertFalse(failed["alerted"])
        self.assertIn("telegram down", failed["send_error"])
        self.assertTrue(retried["alerted"])
        sent.assert_called_once()

    def test_corrupt_watchdog_state_does_not_stop_the_watch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._heartbeat(root, age_seconds=4000)
            root.joinpath("state", "watchdog.json").write_text("{not json", "utf-8")
            healthy, report, sent = self._run(root)
        self.assertFalse(healthy)
        self.assertTrue(report["alerted"])
        sent.assert_called_once()

    def test_watchdog_never_touches_daemon_state(self) -> None:
        # A locked or corrupt SQLite file is one of the failures the watchdog
        # has to report, so it must not need that file to do its job.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._heartbeat(root, age_seconds=4000)
            self._run(root)
            written = {p.name for p in root.joinpath("state").iterdir()}
        self.assertEqual(written, {"heartbeat.json", "watchdog.json"})
        self.assertFalse(root.joinpath("outbox").exists())


class NumberGuardTests(unittest.TestCase):
    """The guard has to reject invented magnitudes without rejecting translation.

    Every case here is taken from production: between 2026-07-27 and 2026-08-03
    the guard fired 112 times and passed 0 times, so every alert went out as the
    untranslated English source line.
    """

    def test_percent_sign_is_a_unit_not_an_invented_number(self) -> None:
        # Source carries the unit in its own field, so writing -0.14% from
        # actual=-0.14 is the correct rendering, not a new number.
        source = '{"actual": -0.14, "estimate": 0.1, "unit": "%"}'
        written = "环比-0.14%，低于预期0.1%"
        self.assertTrue(numeric_values(written).issubset(echoable_numbers(source)))

    def test_english_scale_words_allow_chinese_rescaling(self) -> None:
        # 2026-08-03 production rejections ['80'], ['38'], ['2040'], ['100']:
        # all were billion→亿 rescalings of correct translations.
        cases = {
            "up to $8 billion in cash": "80亿",
            "a $3.8bn deal": "38亿",
            "worth 204 billion pounds": "2040亿",
            "a $10 billion pledge": "100亿",
            "about 5 million users": "500万",
            "roughly 12k staff": "12000",
        }
        for source, written in cases.items():
            self.assertTrue(
                numeric_values(written).issubset(echoable_numbers(source)),
                f"{source} -> {written}",
            )

    def test_scale_words_do_not_open_the_gate_wide(self) -> None:
        allowed = echoable_numbers("up to $8 billion in cash")
        self.assertNotIn("81", allowed)
        self.assertNotIn("800", allowed)

    def test_minus_sign_survives_a_chinese_character(self) -> None:
        # The old lookbehind treated CJK as a word character and dropped the
        # sign, turning -0.14 into 0.14 and reporting it as invented.
        self.assertEqual(numeric_values("环比-0.14%"), {"-0.14"})

    def test_month_name_may_become_a_digit(self) -> None:
        # Jul reads as 7月 in Chinese; the digit is a translation, not a claim.
        self.assertIn("7", echoable_numbers("ID Inflation Rate MoM (Jul)"))
        self.assertNotIn("7", echoable_numbers("ID Inflation Rate MoM (Aug)"))

    def test_quarter_label_may_become_a_digit(self) -> None:
        self.assertIn("2", echoable_numbers("Q2 revenue beat"))

    def test_iso_date_does_not_leak_a_negative_month(self) -> None:
        self.assertNotIn("-8", numeric_values("2026-08-03T12:00:00+08:00"))

    def test_invented_magnitude_is_still_rejected(self) -> None:
        source = '{"actual": -0.14, "estimate": 0.1, "unit": "%"}'
        invented = "环比-0.14%，年化-1.7%"
        self.assertFalse(numeric_values(invented).issubset(echoable_numbers(source)))

    def test_flipped_sign_is_still_rejected(self) -> None:
        # Deflation reported as inflation is exactly what the guard is for.
        source = '{"actual": -0.14, "unit": "%"}'
        self.assertFalse(
            numeric_values("环比+0.14%").issubset(echoable_numbers(source))
        )


class LlmSummarizerTests(unittest.TestCase):
    def _event(self, title: str) -> AlertEvent:
        moment = datetime(2026, 8, 3, 12, 0, tzinfo=UTC)
        item = NewsItem(
            identity="macro-1",
            title=title,
            url="https://example.com/macro",
            source="FMP Economic Calendar",
            source_id="fmp_macro",
            source_tier="primary",
            published_at=moment,
            fetched_at=moment,
            actual=-0.14,
            estimate=0.1,
            unit="%",
        )
        assessment = Assessment(
            level="P0",
            category="macro",
            event_key="macro:test",
            material_hash="hash",
            reason="deviation",
            requires_corroboration=False,
            region="global",
            action="watch",
            entities=(),
        )
        return AlertEvent(assessment=assessment, items=[item])

    def _summarizer(self) -> LlmSummarizer:
        return LlmSummarizer(
            enabled=True,
            base_url="http://127.0.0.1:8317/v1",
            api_key="key",
            model="model",
            timeout=5,
            max_output_tokens=200,
        )

    def _reply(self, payload: dict[str, str]) -> mock.Mock:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        content = json.dumps(payload, ensure_ascii=False)
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        return response

    def test_faithful_translation_reaches_the_message(self) -> None:
        event = self._event("ID Inflation Rate MoM (Jul): actual -0.14, estimate 0.1")
        payload = {
            "title": "印尼7月通胀环比转负",
            "fact": "印尼7月CPI环比-0.14%，低于预期0.1%",
            "impact": "利率预期与风险资产定价可能快速调整。",
        }
        with mock.patch(
            "radar.summarizer.requests.post", return_value=self._reply(payload)
        ):
            summary = self._summarizer().summarize(event)
        self.assertEqual(summary.fact, payload["fact"])
        self.assertEqual(summary.title, payload["title"])

    def test_overlong_title_keeps_the_translated_body(self) -> None:
        # Losing the whole summary over title length is what shipped English.
        event = self._event("ID Inflation Rate MoM (Jul): actual -0.14, estimate 0.1")
        payload = {
            "title": "印尼七月消费者物价指数环比意外转为负值创下今年新低",
            "fact": "印尼7月CPI环比-0.14%，低于预期0.1%",
            "impact": "利率预期与风险资产定价可能快速调整。",
        }
        with mock.patch(
            "radar.summarizer.requests.post", return_value=self._reply(payload)
        ):
            summary = self._summarizer().summarize(event)
        self.assertEqual(summary.title, "经济数据大幅偏离")
        self.assertEqual(summary.fact, payload["fact"])

    def test_invented_number_still_falls_back_to_source(self) -> None:
        event = self._event("ID Inflation Rate MoM (Jul): actual -0.14, estimate 0.1")
        payload = {
            "title": "印尼通胀转负",
            "fact": "印尼7月CPI环比-0.14%，年化降幅达1.7%",
            "impact": "利率预期可能调整。",
        }
        with mock.patch(
            "radar.summarizer.requests.post", return_value=self._reply(payload)
        ):
            summary = self._summarizer().summarize(event)
        self.assertEqual(summary.title, "经济数据大幅偏离")
        self.assertIn("ID Inflation Rate MoM", summary.fact)

    def test_scale_conversion_is_not_invented(self) -> None:
        # $8 billion correctly becomes 80亿; the gate rejected exactly this
        # in production (message 26527) and shipped the English title instead.
        event = self._event("Curium to acquire Lantheus for up to $8 billion in cash")
        payload = {
            "title": "居里制药收购朗特斯",
            "fact": (
                "Curium（居里制药）将以最高80亿美元现金收购"
                "Lantheus（美国放射性药物公司）。"
            ),
            "impact": "相关行业格局可能变化。",
        }
        with mock.patch(
            "radar.summarizer.requests.post", return_value=self._reply(payload)
        ) as post:
            summary = self._summarizer().summarize(event)
        self.assertEqual(summary.fact, payload["fact"])
        self.assertEqual(len(post.call_args_list), 1)

    def test_invented_number_recovers_via_feedback_retry(self) -> None:
        event = self._event("ID Inflation Rate MoM (Jul): actual -0.14, estimate 0.1")
        bad = {
            "title": "印尼通胀转负",
            "fact": "印尼7月CPI环比-0.14%，年化降幅达1.7%",
            "impact": "利率预期可能调整。",
        }
        good = {
            "title": "印尼通胀转负",
            "fact": "印尼7月CPI环比-0.14%，低于预期0.1%",
            "impact": "利率预期可能调整。",
        }
        with mock.patch(
            "radar.summarizer.requests.post",
            side_effect=[self._reply(bad), self._reply(good)],
        ) as post:
            summary = self._summarizer().summarize(event)
        self.assertEqual(summary.fact, good["fact"])
        second_body = post.call_args_list[1].kwargs["json"]["messages"][1]["content"]
        self.assertIn("revision_note", second_body)
        self.assertIn("1.7", second_body)

    def test_bare_english_recovers_via_retry(self) -> None:
        event = self._event(
            "Clearmind Medicine to acquire 51% stake in EV charging firm"
        )
        bare = {
            "title": "医药公司跨界收购",
            "fact": "Clearmind Medicine将收购EV充电公司51%股权",
            "impact": "相关公司估值可能变化。",
        }
        glossed = {
            "title": "医药公司跨界收购",
            "fact": ("Clearmind Medicine（加拿大医药公司）将收购电动车充电公司51%股权"),
            "impact": "相关公司估值可能变化。",
        }
        with mock.patch(
            "radar.summarizer.requests.post",
            side_effect=[self._reply(bare), self._reply(glossed)],
        ):
            summary = self._summarizer().summarize(event)
        self.assertEqual(summary.fact, glossed["fact"])

    def test_stubborn_bare_english_still_beats_english_fallback(self) -> None:
        event = self._event(
            "Clearmind Medicine to acquire 51% stake in EV charging firm"
        )
        bare = {
            "title": "医药公司跨界收购",
            "fact": "Clearmind Medicine将收购EV充电公司51%股权",
            "impact": "相关公司估值可能变化。",
        }
        with mock.patch(
            "radar.summarizer.requests.post", return_value=self._reply(bare)
        ):
            summary = self._summarizer().summarize(event)
        self.assertEqual(summary.fact, bare["fact"])


class BareEnglishSpanTests(unittest.TestCase):
    def test_flags_unglossed_company_names(self) -> None:
        self.assertEqual(
            bare_english_spans("Clearmind Medicine将收购EV充电公司51%股权"),
            ["Clearmind Medicine"],
        )

    def test_accepts_glossed_names_tickers_and_acronyms(self) -> None:
        self.assertEqual(
            bare_english_spans("Visa（维萨）收购案获批，TSLA 大涨，美国CPI回落"),
            [],
        )

    def test_gloss_once_covers_bare_repeats(self) -> None:
        # 中文媒体惯例：首次注释，后文裸用。逐次要求括号会误伤正确译文。
        text = (
            "Curium拟收购Lantheus\n"
            "Curium（居里）将收购Lantheus（放射药企）\n"
            "Lantheus股价可能上涨"
        )
        self.assertEqual(bare_english_spans(text), [])

    def test_newline_join_keeps_fields_apart(self) -> None:
        # 标题结尾和正文开头各有一个名字：换行拼接下它们是两个独立词组，
        # 各自凭全文注释豁免；空格拼接会把它们粘成一个幻影词组。
        self.assertEqual(
            bare_english_spans(
                "收购Lantheus\nCurium（居里）完成\nLantheus（放射药企）"
            ),
            [],
        )


FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"x" * 2000


class ComicArtistTests(unittest.TestCase):
    def _artist(self, **overrides) -> ComicArtist:
        settings = {
            "enabled": True,
            "base_url": "http://127.0.0.1:8317/v1",
            "api_key": "key",
            "scene_model": "scene-model",
            "image_model": "grok-imagine-image",
            "scene_timeout": 5,
            "image_timeout": 5,
        }
        settings.update(overrides)
        return ComicArtist(**settings)

    def _scene_reply(self, scenes: list[str]) -> mock.Mock:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        content = json.dumps({"scenes": scenes}, ensure_ascii=False)
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        return response

    def _image_reply(self, photo: bytes) -> mock.Mock:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [{"b64_json": base64.b64encode(photo).decode("ascii")}]
        }
        return response

    def test_disabled_artist_makes_no_calls(self) -> None:
        self.assertFalse(self._artist(api_key="").enabled)
        artist = self._artist(enabled=False)
        with mock.patch("radar.comic.requests.post") as post:
            self.assertIsNone(artist.generate([("macro", "事实")]))
        post.assert_not_called()

    def test_generated_bytes_decoded_and_counted(self) -> None:
        artist = self._artist(scene_model="")
        with mock.patch(
            "radar.comic.requests.post", return_value=self._image_reply(FAKE_JPEG)
        ) as post:
            photo = artist.generate([("macro", "事实")])
        self.assertEqual(photo, FAKE_JPEG)
        self.assertEqual(artist.stats(), {"generated": 1, "failed": 0})
        prompt = post.call_args.kwargs["json"]["prompt"]
        self.assertIn("single-panel", prompt)
        self.assertIn(HARD_BANS, prompt)
        self.assertIn(CATEGORY_SCENES["macro"], prompt)

    def test_scene_with_digits_replaced_by_category_fallback(self) -> None:
        # The scene text lands on the image model's canvas: a number that
        # sneaks in becomes a drawn "fact" no gate ever checked.
        artist = self._artist()
        replies = [
            self._scene_reply(["a chart rising 5 percent over a city"]),
            self._image_reply(FAKE_JPEG),
        ]
        with mock.patch("radar.comic.requests.post", side_effect=replies) as post:
            photo = artist.generate([("macro", "CPI 环比 -0.14%")])
        self.assertEqual(photo, FAKE_JPEG)
        prompt = post.call_args_list[1].kwargs["json"]["prompt"]
        self.assertNotIn("5 percent", prompt)
        self.assertIn(CATEGORY_SCENES["macro"], prompt)

    def test_scene_failure_uses_deterministic_scenes(self) -> None:
        artist = self._artist()
        replies = [RuntimeError("scene model down"), self._image_reply(FAKE_JPEG)]
        with mock.patch("radar.comic.requests.post", side_effect=replies) as post:
            photo = artist.generate([("central_bank", "美联储发布决议")])
        self.assertEqual(photo, FAKE_JPEG)
        prompt = post.call_args_list[1].kwargs["json"]["prompt"]
        self.assertIn(CATEGORY_SCENES["central_bank"], prompt)

    def test_four_entries_build_four_panel_prompt(self) -> None:
        artist = self._artist(scene_model="")
        entries = [
            ("macro", "一"),
            ("central_bank", "二"),
            ("geopolitics", "三"),
            ("industry", "四"),
            ("earnings", "五"),
        ]
        with mock.patch(
            "radar.comic.requests.post", return_value=self._image_reply(FAKE_JPEG)
        ) as post:
            photo = artist.generate(entries)
        self.assertEqual(photo, FAKE_JPEG)
        prompt = post.call_args.kwargs["json"]["prompt"]
        # The fifth entry is dropped: one image holds at most four panels.
        self.assertIn("4-panel", prompt)
        self.assertIn("Panel 4:", prompt)
        self.assertNotIn("Panel 5:", prompt)
        self.assertIn(HARD_BANS, prompt)

    def test_image_failure_returns_none_and_counts(self) -> None:
        artist = self._artist(scene_model="")
        with mock.patch(
            "radar.comic.requests.post", side_effect=RuntimeError("image API down")
        ):
            self.assertIsNone(artist.generate([("macro", "事实")]))
        self.assertEqual(artist.stats(), {"generated": 0, "failed": 1})


class TrendingJudgeTests(unittest.TestCase):
    def _judge(self, **overrides) -> TrendingJudge:
        settings = {
            "enabled": True,
            "base_url": "http://127.0.0.1:8317/v1",
            "api_key": "key",
            "model": "judge-model",
            "timeout": 5,
            "max_per_day": 3,
        }
        settings.update(overrides)
        return TrendingJudge(**settings)

    def _candidates(self) -> list[NewsItem]:
        return [
            item(
                "Rare comet visible worldwide this weekend",
                identity="comet-1",
                source="BBC",
                source_tier="secondary",
            ),
            item(
                "Lost Rembrandt painting found in attic",
                identity="painting-1",
                source="Reuters",
                source_tier="secondary",
            ),
        ]

    def _reply(self, picks: list[str]) -> mock.Mock:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        content = json.dumps({"picks": picks}, ensure_ascii=False)
        response.json.return_value = {"choices": [{"message": {"content": content}}]}
        return response

    def test_valid_pick_honored_invented_id_dropped(self) -> None:
        judge = self._judge()
        reply = self._reply(["comet-1", "made-up-id"])
        with mock.patch("radar.trending.requests.post", return_value=reply):
            picks = judge.pick(self._candidates(), 3)
        self.assertEqual([picked.identity for picked in picks], ["comet-1"])
        self.assertEqual(judge.stats()["picked"], 1)

    def test_disabled_or_exhausted_makes_no_calls(self) -> None:
        with mock.patch("radar.trending.requests.post") as post:
            self.assertEqual(self._judge(enabled=False).pick(self._candidates(), 3), [])
            self.assertEqual(self._judge().pick(self._candidates(), 0), [])
            self.assertEqual(self._judge().pick([], 3), [])
        post.assert_not_called()

    def test_judge_failure_counts_and_returns_nothing(self) -> None:
        judge = self._judge()
        with mock.patch(
            "radar.trending.requests.post", side_effect=RuntimeError("down")
        ):
            self.assertEqual(judge.pick(self._candidates(), 3), [])
        self.assertEqual(judge.stats()["failed"], 1)


class PhotoDeliveryTests(unittest.TestCase):
    def _message(self) -> RenderedMessage:
        event = RenderTests()._event(
            "photo-fed",
            "Federal Reserve issues FOMC statement",
        )
        return render_p0(
            event,
            Summary("美联储发布决议", "美联储公布最新决定。", "利率预期可能调整。"),
            NOW,
            400,
        )

    def _accepted(self, message_id: int) -> mock.Mock:
        response = mock.Mock(ok=True, status_code=200, text="ok")
        response.json.return_value = {
            "ok": True,
            "result": {"message_id": message_id},
        }
        return response

    def test_photo_send_uses_multipart_and_records_modality(self) -> None:
        message = self._message()
        with tempfile.TemporaryDirectory() as directory:
            delivery = TelegramDelivery(
                token="test-token",
                chat_id="-1001",
                outbox_dir=Path(directory),
                max_visible_chars=400,
                send_enabled=True,
            )
            with mock.patch(
                "radar.delivery.requests.post", return_value=self._accepted(9100)
            ) as post:
                result = delivery.send(message, "412", photo=FAKE_JPEG)
            self.assertTrue(post.call_args.args[0].endswith("/sendPhoto"))
            data = post.call_args.kwargs["data"]
            files = post.call_args.kwargs["files"]
            manifest = json.loads(Path(directory, "latest.json").read_text())
            jpg = Path(directory, "latest.jpg").read_bytes()
        self.assertEqual(result["modality"], "photo")
        self.assertEqual(result["message_id"], "9100")
        self.assertEqual(data["caption"], message.html)
        self.assertEqual(data["parse_mode"], "HTML")
        self.assertEqual(data["message_thread_id"], 412)
        self.assertEqual(files["photo"][1], FAKE_JPEG)
        self.assertEqual(manifest["modality"], "photo")
        self.assertEqual(manifest["delivery"]["status"], "sent_photo")
        self.assertEqual(jpg, FAKE_JPEG)

    def test_photo_failure_falls_back_to_text_message(self) -> None:
        message = self._message()
        broken = mock.Mock(ok=False, status_code=500, text="err")
        broken.json.return_value = {"ok": False, "description": "Internal error"}
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
                side_effect=[broken, self._accepted(9101)],
            ) as post:
                result = delivery.send(message, "412", photo=FAKE_JPEG)
            self.assertTrue(post.call_args_list[1].args[0].endswith("/sendMessage"))
            manifest = json.loads(Path(directory, "latest.json").read_text())
        self.assertNotIn("modality", result)
        self.assertEqual(result["message_id"], "9101")
        self.assertEqual(manifest["modality"], "text")
        self.assertEqual(manifest["delivery"]["status"], "sent")

    def test_dry_run_with_photo_writes_jpg_without_network(self) -> None:
        message = self._message()
        with tempfile.TemporaryDirectory() as directory:
            delivery = TelegramDelivery(
                token="test-token",
                chat_id="-1001",
                outbox_dir=Path(directory),
                max_visible_chars=400,
                send_enabled=False,
            )
            with mock.patch("radar.delivery.requests.post") as post:
                result = delivery.send(message, "412", photo=FAKE_JPEG)
            post.assert_not_called()
            jpg = Path(directory, "latest.jpg").read_bytes()
        self.assertTrue(result["dry_run"])
        self.assertEqual(jpg, FAKE_JPEG)

    def test_caption_over_limit_drops_photo(self) -> None:
        long_text = "字" * 1100
        message = RenderedMessage(
            level="P0",
            html=long_text,
            plain_text=long_text,
            event_keys=["over-limit"],
            content_hash="deadbeefdeadbeef",
            evidence=[],
            created_at=NOW,
        )
        with tempfile.TemporaryDirectory() as directory:
            delivery = TelegramDelivery(
                token="test-token",
                chat_id="-1001",
                outbox_dir=Path(directory),
                max_visible_chars=2000,
                send_enabled=True,
            )
            with mock.patch(
                "radar.delivery.requests.post", return_value=self._accepted(9102)
            ) as post:
                delivery.send(message, "412", photo=FAKE_JPEG)
            self.assertEqual(len(post.call_args_list), 1)
            self.assertTrue(post.call_args.args[0].endswith("/sendMessage"))
            self.assertFalse(Path(directory, "latest.jpg").exists())


class OutboxPruneTests(unittest.TestCase):
    def test_prunes_old_files_but_keeps_latest(self) -> None:
        from radar.delivery import prune_outbox

        with tempfile.TemporaryDirectory() as directory:
            outbox = Path(directory)
            old = outbox / "20260701T000000Z-p0-aaaa.jpg"
            fresh = outbox / "20260803T000000Z-p0-bbbb.jpg"
            latest = outbox / "latest.jpg"
            for path in (old, fresh, latest):
                path.write_bytes(b"x")
            now_timestamp = NOW.timestamp()
            import os as _os

            _os.utime(old, (now_timestamp - 10 * 86400,) * 2)
            _os.utime(fresh, (now_timestamp - 3600,) * 2)
            _os.utime(latest, (now_timestamp - 10 * 86400,) * 2)
            removed = prune_outbox(outbox, now_timestamp, 7)
            self.assertEqual(removed, 1)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(latest.exists())


class ComicWiringTests(unittest.TestCase):
    def test_p0_publish_passes_comic_to_delivery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            service.comic = mock.Mock(
                for_event=mock.Mock(return_value=FAKE_JPEG), enabled=True
            )
            event = RenderTests()._event(
                "wiring-fed",
                "Federal Reserve issues FOMC statement",
            )
            sent = service._publish_p0(event, NOW, quiet=False)
            self.assertEqual(sent, 1)
            self.assertEqual(service.delivery.photos, [FAKE_JPEG])
            service.close()

    def test_quiet_window_queues_text_without_generating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            service.comic = mock.Mock(
                for_event=mock.Mock(return_value=FAKE_JPEG), enabled=True
            )
            event = RenderTests()._event(
                "quiet-fed",
                "Federal Reserve issues FOMC statement",
            )
            sent = service._publish_p0(event, NOW, quiet=True)
            self.assertEqual(sent, 0)
            service.comic.for_event.assert_not_called()
            service.close()


class ClusterEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = RadarStore(Path(directory.name) / "radar.sqlite3")
        self.addCleanup(self.store.close)
        self.engine = ClusterEngine(
            self.store,
            window_hours=12,
            min_shared_tokens=3,
            burst_min_majors=3,
            burst_window_minutes=90,
            major_sources=["Reuters", "Bloomberg", "Financial Times"],
        )

    @staticmethod
    def _filing(identity: str, source: str, title: str, minutes_old: int) -> NewsItem:
        return item(
            title,
            identity=identity,
            source=source,
            source_tier="secondary",
            minutes_old=minutes_old,
        )

    def _shelf_story(self) -> list[int | None]:
        return [
            self.engine.add(
                self._filing(
                    "shelf-reuters",
                    "Reuters",
                    "Antarctic ice shelf shows record seasonal retreat",
                    20,
                ),
                NOW,
            ),
            self.engine.add(
                self._filing(
                    "shelf-bloomberg",
                    "Bloomberg",
                    "Scientists confirm record retreat of Antarctic ice shelf",
                    12,
                ),
                NOW,
            ),
            self.engine.add(
                self._filing(
                    "shelf-ft",
                    "Financial Times",
                    "Antarctic ice shelf retreat accelerates, researchers warn",
                    6,
                ),
                NOW,
            ),
        ]

    def test_tokenize_keeps_content_words_only(self) -> None:
        tokens = tokenize("Breaking News: US to act on AI rules today")
        self.assertIn("act", tokens)
        self.assertIn("rules", tokens)
        for noise in ("breaking", "news", "today", "us", "to", "ai"):
            self.assertNotIn(noise, tokens)
        digits = tokenize("Magnitude 7 quake leaves 79 dead")
        self.assertIn("79", digits)
        self.assertNotIn("7", digits)

    def test_shared_tokens_join_one_story_and_strangers_found_new(self) -> None:
        first, second, third = self._shelf_story()
        stranger = self.engine.add(
            self._filing(
                "shelf-stranger",
                "Reuters",
                "Parliament debates fisheries quota reform bill",
                10,
            ),
            NOW,
        )
        self.assertEqual(first, second)
        self.assertEqual(first, third)
        self.assertIsNotNone(stranger)
        self.assertNotEqual(first, stranger)

    def test_add_is_idempotent_by_item_identity(self) -> None:
        filing = self._filing(
            "shelf-idem",
            "Reuters",
            "Antarctic ice shelf shows record seasonal retreat",
            20,
        )
        first = self.engine.add(filing, NOW)
        again = self.engine.add(filing, NOW + timedelta(minutes=2))
        self.assertEqual(first, again)
        members = self.store.connection.execute(
            "SELECT COUNT(*) AS n FROM cluster_member"
        ).fetchone()
        self.assertEqual(int(members["n"]), 1)

    def test_sparse_title_is_rejected(self) -> None:
        self.assertIsNone(
            self.engine.add(self._filing("sparse", "Reuters", "Markets today", 5), NOW)
        )

    def test_stale_cluster_does_not_absorb_new_filing(self) -> None:
        first = self.engine.add(
            self._filing(
                "age-a",
                "Reuters",
                "Antarctic ice shelf shows record seasonal retreat",
                20,
            ),
            NOW,
        )
        later = self.engine.add(
            self._filing(
                "age-b",
                "Bloomberg",
                "Scientists confirm record retreat of Antarctic ice shelf",
                12,
            ),
            NOW + timedelta(hours=13),
        )
        self.assertNotEqual(first, later)

    def test_three_majors_inside_window_form_burst_candidate(self) -> None:
        self._shelf_story()
        candidates = self.engine.burst_candidates(NOW)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["representative"].identity, "shelf-reuters")
        self.assertEqual(len(candidate["headlines"]), 3)
        self.assertTrue(candidate["scope"])
        self.engine.mark_promoted(int(candidate["cluster_id"]), NOW)
        self.assertEqual(self.engine.burst_candidates(NOW), [])

    def test_two_majors_are_not_a_burst(self) -> None:
        self._shelf_story()[0]
        self.store.connection.execute(
            "DELETE FROM cluster_member WHERE item_identity = ?", ("shelf-ft",)
        )
        self.store.connection.commit()
        self.assertEqual(self.engine.burst_candidates(NOW), [])

    def test_spread_out_first_filings_are_not_a_burst(self) -> None:
        self.engine.add(
            self._filing(
                "spread-a",
                "Reuters",
                "Antarctic ice shelf shows record seasonal retreat",
                240,
            ),
            NOW,
        )
        self.engine.add(
            self._filing(
                "spread-b",
                "Bloomberg",
                "Scientists confirm record retreat of Antarctic ice shelf",
                120,
            ),
            NOW,
        )
        self.engine.add(
            self._filing(
                "spread-c",
                "Financial Times",
                "Antarctic ice shelf retreat accelerates, researchers warn",
                5,
            ),
            NOW,
        )
        self.assertEqual(self.engine.burst_candidates(NOW), [])

    def test_non_major_pileup_is_not_a_burst(self) -> None:
        for index, source in enumerate(("Some Blog", "Other Blog", "Third Blog")):
            self.engine.add(
                self._filing(
                    f"blogs-{index}",
                    source,
                    "Antarctic ice shelf shows record seasonal retreat",
                    20 - index,
                ),
                NOW,
            )
        self.assertEqual(self.engine.burst_candidates(NOW), [])

    def test_top_clusters_rank_major_consensus_first(self) -> None:
        self.engine.add(
            self._filing(
                "rank-a1",
                "Some Blog",
                "Chile copper mines halt production after nationwide power failure",
                30,
            ),
            NOW,
        )
        self.engine.add(
            self._filing(
                "rank-a2",
                "Other Blog",
                "Nationwide power failure halts Chile copper production",
                25,
            ),
            NOW,
        )
        self.engine.add(
            self._filing(
                "rank-a3",
                "Third Blog",
                "Chile copper production halted by power failure",
                20,
            ),
            NOW,
        )
        self.engine.add(
            self._filing(
                "rank-b1",
                "Reuters",
                "Antarctic ice shelf shows record seasonal retreat",
                10,
            ),
            NOW,
        )
        self.engine.add(
            self._filing(
                "rank-c1",
                "Some Blog",
                "Parliament debates fisheries quota reform bill",
                5,
            ),
            NOW,
        )
        top = self.engine.top_clusters(
            NOW - timedelta(hours=1), NOW + timedelta(hours=1), limit=8
        )
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0]["major_count"], 1)
        self.assertEqual(top[0]["rep_source"], "Reuters")
        self.assertEqual(top[1]["member_count"], 3)

    def test_prune_removes_expired_clusters_and_members(self) -> None:
        self._shelf_story()
        self.engine.prune(NOW + timedelta(hours=49))
        for table in ("story_cluster", "cluster_member"):
            count = self.store.connection.execute(
                f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608
            ).fetchone()
            self.assertEqual(int(count["n"]), 0)


class BurstPromotionTests(unittest.TestCase):
    @staticmethod
    def _burst_trio() -> list[NewsItem]:
        return [
            item(
                "Antarctic ice shelf shows record seasonal retreat",
                identity="burst-reuters",
                source="Reuters",
                source_tier="secondary",
                minutes_old=20,
            ),
            item(
                "Scientists confirm record retreat of Antarctic ice shelf",
                identity="burst-bloomberg",
                source="Bloomberg",
                source_tier="secondary",
                minutes_old=12,
            ),
            item(
                "Antarctic ice shelf retreat accelerates, researchers warn",
                identity="burst-ft",
                source="Financial Times",
                source_tier="secondary",
                minutes_old=6,
            ),
        ]

    def test_three_major_filings_promote_burst_p0(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            judge = FakeJudge(picks=False, burst_verdict=True)
            service.trending = judge
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            service.collect = lambda now: (self._burst_trio(), [], [])
            stats = service.run_once(NOW)
            self.assertEqual(stats["sent"], 1)
            message, thread = service.delivery.messages[0]
            self.assertEqual(message.level, "P0")
            self.assertEqual(thread, NEWS_THREAD)
            self.assertTrue(message.event_keys[0].startswith("world_burst"))
            self.assertEqual(len(judge.burst_calls), 1)
            service.close()

    def test_judge_rejection_is_permanent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            judge = FakeJudge(picks=False, burst_verdict=False)
            service.trending = judge
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            service.collect = lambda now: (self._burst_trio(), [], [])
            self.assertEqual(service.run_once(NOW)["sent"], 0)
            self.assertEqual(len(judge.burst_calls), 1)
            service.collect = lambda now: ([], [], [])
            service.run_once(NOW + timedelta(minutes=2))
            # mark_promoted on rejection: the same cluster is never re-judged.
            self.assertEqual(len(judge.burst_calls), 1)
            self.assertEqual(service.delivery.messages, [])
            service.close()

    def test_judge_outage_retries_next_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            judge = FakeJudge(picks=False, burst_verdict=None)
            service.trending = judge
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            service.collect = lambda now: (self._burst_trio(), [], [])
            self.assertEqual(service.run_once(NOW)["sent"], 0)
            self.assertEqual(len(judge.burst_calls), 1)
            judge.burst_verdict = True
            service.collect = lambda now: ([], [], [])
            stats = service.run_once(NOW + timedelta(minutes=2))
            self.assertEqual(stats["sent"], 1)
            self.assertEqual(len(judge.burst_calls), 2)
            service.close()

    def test_vocab_story_escalates_without_judge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            judge = FakeJudge(picks=False)
            service.trending = judge
            quake = [
                item(
                    "Powerful earthquake strikes northern Chile coast",
                    identity="quake-reuters",
                    source="Reuters",
                    source_tier="secondary",
                    minutes_old=20,
                ),
                item(
                    "Northern Chile earthquake prompts coastal evacuations",
                    identity="quake-bloomberg",
                    source="Bloomberg",
                    source_tier="secondary",
                    minutes_old=12,
                ),
                item(
                    "Chile earthquake: tsunami warning issued for northern coast",
                    identity="quake-ft",
                    source="Financial Times",
                    source_tier="secondary",
                    minutes_old=6,
                ),
            ]
            for filing in quake:
                service.clusters.add(filing, NOW)
            events: list[AlertEvent] = []
            service._burst_promotions(NOW, events)
            self.assertEqual(len(events), 1)
            assessment = events[0].assessment
            self.assertEqual(assessment.level, "P0")
            self.assertFalse(assessment.requires_corroboration)
            self.assertEqual(assessment.category, "disaster")
            self.assertEqual(judge.burst_calls, [])
            service.close()


class MacroDigestTests(unittest.TestCase):
    @staticmethod
    def _macro_event(identity: str, country: str) -> AlertEvent:
        release = NewsItem(
            identity=identity,
            title=f"{country} CPI YoY: actual 3.2, estimate 2.9",
            url="https://example.com/macro",
            source="FMP Economic Calendar",
            source_id="fmp_macro",
            source_tier="structured",
            published_at=NOW - timedelta(minutes=5),
            fetched_at=NOW,
            actual=3.2,
            estimate=2.9,
            unit="%",
            raw={"country": country},
        )
        assessment = Assessment(
            level="P0",
            category="macro",
            event_key=f"macro:cpi:{identity}",
            material_hash=identity,
            reason="surprise 0.3pp",
            requires_corroboration=False,
            region="global",
            action="data_release",
            entities=(),
        )
        return AlertEvent(assessment=assessment, items=[release])

    def test_same_country_prints_coalesce_into_one_digest(self) -> None:
        events = [
            self._macro_event("mac-us-1", "US"),
            self._macro_event("mac-us-2", "US"),
            self._macro_event("mac-jp-1", "JP"),
        ]
        singles, digests = RadarService._coalesce_macro(events)
        self.assertEqual(
            [event.assessment.event_key for event in singles],
            ["macro:cpi:mac-jp-1"],
        )
        self.assertEqual(len(digests), 1)
        self.assertEqual(len(digests[0]), 2)

    def test_distinct_or_unknown_countries_stay_single(self) -> None:
        events = [
            self._macro_event("mac-us", "US"),
            self._macro_event("mac-jp", "JP"),
            self._macro_event("mac-blank", ""),
        ]
        singles, digests = RadarService._coalesce_macro(events)
        self.assertEqual(len(singles), 3)
        self.assertEqual(digests, [])

    def test_digest_chunks_at_render_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            group = [self._macro_event(f"mac-chunk-{i}", "US") for i in range(7)]
            sent = service._publish_macro_digest(group, NOW, quiet=False)
            self.assertEqual(sent, 2)
            first, second = service.delivery.messages
            self.assertEqual(first[0].level, "P1")
            self.assertEqual(len(first[0].event_keys), 5)
            self.assertEqual(len(second[0].event_keys), 2)
            service.close()

    def test_quiet_window_queues_digest_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            group = [self._macro_event(f"mac-quiet-{i}", "US") for i in range(2)]
            sent = service._publish_macro_digest(group, NOW, quiet=True)
            self.assertEqual(sent, 0)
            self.assertEqual(service.delivery.messages, [])
            row = service.store.connection.execute(
                "SELECT last_error FROM pending_delivery"
            ).fetchone()
            self.assertEqual(row["last_error"], "quiet_window")
            service.close()


class BriefComposerTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = RadarStore(Path(directory.name) / "radar.sqlite3")
        self.addCleanup(self.store.close)
        self.engine = ClusterEngine(
            self.store,
            window_hours=12,
            min_shared_tokens=3,
            burst_min_majors=3,
            burst_window_minutes=90,
            major_sources=["Reuters", "Bloomberg", "Financial Times"],
        )
        self.composer = BriefComposer(
            store=self.store,
            engine=self.engine,
            base_url="",
            api_key="",
            model="",
            timeout=5,
            max_items=8,
            max_visible_chars=950,
            morning_sgt="07:30",
            evening_sgt="20:30",
        )

    def test_due_slots_and_expiry(self) -> None:
        # NOW is 18:00 SGT: between the morning window's end and the evening.
        self.assertIsNone(self.composer.due(NOW))
        self.assertEqual(self.composer.due(NOW - timedelta(hours=10)), "morning")
        # A morning missed until 13:00 SGT expires instead of arriving stale.
        self.assertIsNone(self.composer.due(NOW - timedelta(hours=5)))
        self.assertEqual(self.composer.due(NOW + timedelta(hours=3)), "evening")

    def test_mark_sent_makes_each_slot_at_most_once(self) -> None:
        evening = NOW + timedelta(hours=3)
        self.assertEqual(self.composer.due(evening), "evening")
        self.composer.mark_sent("evening", evening)
        self.assertIsNone(self.composer.due(evening))

    def test_compose_empty_window_emits_placeholder(self) -> None:
        message, comic_entries = self.composer.compose(
            "evening", NOW + timedelta(hours=3)
        )
        self.assertEqual(message.level, "BRIEF")
        self.assertEqual(message.event_keys, ["brief:evening:2026-07-25"])
        self.assertIn("世界晚报 · 7月25日 周六", message.plain_text)
        self.assertIn("本时段无达到整理门槛的世界大事。", message.plain_text)
        self.assertIn("窗口13小时 · 告警0条 · 故事簇0个", message.plain_text)
        self.assertLessEqual(visible_length(message.html), 950)
        self.assertEqual(comic_entries, [])

    def test_compose_degrades_to_mechanical_line_without_llm(self) -> None:
        self.engine.add(
            item(
                "Chile copper mines halt production after nationwide power failure",
                identity="brief-reuters",
                source="Reuters",
                source_tier="secondary",
                minutes_old=60,
            ),
            NOW,
        )
        self.engine.add(
            item(
                "Nationwide power failure halts Chile copper production",
                identity="brief-bloomberg",
                source="Bloomberg",
                source_tier="secondary",
                minutes_old=50,
            ),
            NOW,
        )
        message, comic_entries = self.composer.compose(
            "evening", NOW + timedelta(hours=3)
        )
        self.assertIn("2家大社在报", message.plain_text)
        self.assertIn("（2家大社）", message.plain_text)
        self.assertEqual(len(comic_entries), 1)
        self.assertEqual(comic_entries[0][0], "brief")

    def test_gated_line_rejects_invented_numbers_and_bare_english(self) -> None:
        story: dict[str, object] = {
            "titles": [
                "Chile copper mines halt production after nationwide power failure"
            ],
            "major_count": 2,
            "member_count": 2,
            "rep_title": (
                "Chile copper mines halt production after nationwide power failure"
            ),
        }
        clean = self.composer._gated_line("智利铜矿因全国停电停产", story)
        self.assertEqual(clean, "智利铜矿因全国停电停产")
        invented = self.composer._gated_line("智利3座铜矿停产", story)
        self.assertTrue(invented.startswith("2家大社在报"))
        bare = self.composer._gated_line("停产影响 Chile mines 出口", story)
        self.assertTrue(bare.startswith("2家大社在报"))

    def test_continuity_tags_advance_across_days_only(self) -> None:
        stories: list[dict[str, object]] = [
            {
                "id": 7,
                "titles": [
                    "Chile copper mines halt production after nationwide power failure"
                ],
                "major_count": 2,
                "member_count": 3,
                "rep_title": (
                    "Chile copper mines halt production after nationwide power failure"
                ),
            }
        ]
        self.store.set_meta(
            "brief_continuity",
            json.dumps(
                [
                    {
                        "tokens": ["chile", "copper", "production", "failure"],
                        "days": 1,
                        "date": "2026-07-24",
                    }
                ]
            ),
            NOW,
        )
        tags = self.composer._continuity_tags(NOW, stories)
        self.assertEqual(tags, {7: "连续第2天"})
        # The evening rerun on the same date repeats Day-2, not Day-3.
        tags_again = self.composer._continuity_tags(NOW, stories)
        self.assertEqual(tags_again, {7: "连续第2天"})


class BriefServiceTests(unittest.TestCase):
    def test_evening_brief_dispatches_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(hours=1))
            service.briefs_enabled = True
            service.collect = lambda now: ([], [], [])
            due_at = NOW + timedelta(hours=2, minutes=35)  # 20:35 SGT
            service.run_once(due_at)
            self.assertEqual(len(service.delivery.messages), 1)
            message, thread = service.delivery.messages[0]
            self.assertEqual(message.level, "BRIEF")
            self.assertEqual(thread, NEWS_THREAD)
            self.assertEqual(service.store.get_meta("brief_evening_date"), "2026-07-25")
            service.run_once(due_at + timedelta(minutes=2))
            self.assertEqual(len(service.delivery.messages), 1)
            service.close()


class SourceBaselineTests(unittest.TestCase):
    def test_new_source_gets_one_swallowed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory, baseline=True)
            fed = item(
                "Federal Reserve issues FOMC statement",
                identity="grow-fed",
                category_hint="central_bank",
            )
            service.collect = lambda now: ([fed], [], [])
            self.assertEqual(service.run_once(NOW)["status"], "baseline")
            newcomer = item(
                "Bank of England holds interest rates steady",
                identity="grow-yonhap-1",
                source="Yonhap",
                source_tier="primary",
                minutes_old=3,
            )
            service.collect = lambda now: ([newcomer], [], [])
            stats = service.run_once(NOW + timedelta(minutes=2))
            self.assertEqual(stats["sent"], 0)
            self.assertEqual(service.delivery.messages, [])
            self.assertNotEqual(service.store.get_meta("source_baseline:yonhap"), "")
            follow = item(
                "Bank of Japan announces emergency rate hike",
                identity="grow-yonhap-2",
                source="Yonhap",
                source_tier="primary",
                minutes_old=2,
            )
            service.collect = lambda now: ([follow], [], [])
            stats = service.run_once(NOW + timedelta(minutes=4))
            self.assertEqual(stats["sent"], 1)
            self.assertEqual(len(service.delivery.messages), 1)
            service.close()


class SilentPathTests(unittest.TestCase):
    def test_reputable_wait_logs_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            first = item(
                "Fed cuts rates in surprise decision",
                identity="fringe-a",
                source="Random Blog",
                source_tier="secondary",
            )
            second = item(
                "Federal Reserve cuts rates in surprise decision",
                identity="fringe-b",
                source="Obscure Wire",
                source_tier="secondary",
                minutes_old=3,
            )
            service.collect = lambda now: ([first], [], [])
            service.run_once(NOW)
            service.collect = lambda now: ([second], [], [])
            with self.assertLogs("radar.service", level="INFO") as logs:
                stats = service.run_once(NOW + timedelta(minutes=2))
            self.assertEqual(stats["sent"], 0)
            self.assertTrue(any("p0_waiting_reputable" in line for line in logs.output))
            service.close()

    def test_expired_p1_logs_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            service.trending = FakeJudge()
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            teaser = item(
                "Rare comet visible worldwide this weekend",
                identity="expire-comet",
                source="Reuters",
                source_tier="secondary",
            )
            service.collect = lambda now: ([teaser], [], [])
            service.run_once(NOW)
            service.collect = lambda now: ([], [], [])
            with self.assertLogs("radar.service", level="WARNING") as logs:
                service.run_once(NOW + timedelta(minutes=185))
            self.assertTrue(any("p1_expired" in line for line in logs.output))
            self.assertEqual(service.delivery.messages, [])
            service.close()

    def test_near_duplicate_item_is_still_marked_seen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            first = item(
                "Rare comet visible worldwide this weekend",
                identity="comet-a",
                source="Reuters",
                source_tier="secondary",
            )
            second = item(
                "Rare comet visible worldwide this weekend",
                identity="comet-b",
                source="Bloomberg",
                source_tier="secondary",
                minutes_old=3,
            )
            service.collect = lambda now: ([first], [], [])
            service.run_once(NOW)
            service.collect = lambda now: ([second], [], [])
            service.run_once(NOW + timedelta(minutes=2))
            self.assertTrue(service.store.item_seen("comet-b"))
            service.close()


class TrendingSlotResetTests(unittest.TestCase):
    def test_new_slot_restores_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ServiceIntegrationTests().make_service(directory)
            judge = FakeJudge()
            service.trending = judge
            service.store.mark_baseline_complete(NOW - timedelta(minutes=20))
            local = NOW.astimezone(SGT)
            spent_slot = f"{local.date().isoformat()}:{local.hour // 8 - 1}"
            service.store.set_meta("trending_slot", spent_slot, NOW)
            service.store.set_meta("trending_slot_count", "1", NOW)
            teaser = item(
                "Rare comet visible worldwide this weekend",
                identity="slot-comet",
                source="Reuters",
                source_tier="secondary",
            )
            service.collect = lambda now: ([teaser], [], [])
            service.run_once(NOW)
            self.assertEqual(len(judge.offered), 1)
            self.assertEqual(
                service.store.get_meta("trending_slot"),
                f"{local.date().isoformat()}:{local.hour // 8}",
            )
            self.assertEqual(service.store.get_meta("trending_slot_count"), "1")
            service.close()


class EntityAliasTests(unittest.TestCase):
    def test_boeing_headline_does_not_tag_bank_of_england(self) -> None:
        result = trending_assessment(
            item(
                "Boeing wins record aircraft order from Asian carrier",
                source="Reuters",
                source_tier="secondary",
            )
        )
        self.assertNotIn("BOE", result.entities)
        self.assertIn("BOEING", result.entities)

    def test_possessive_fed_headline_still_tags_fed(self) -> None:
        result = assess(
            item(
                "Fed's Powell signals emergency rate cut",
                category_hint="central_bank",
            ),
            CONFIG,
        )
        self.assertIsNotNone(result)
        self.assertIn("FED", result.entities)

    def test_curly_apostrophe_pboc_alias(self) -> None:
        result = assess(
            item(
                "People’s Bank of China cuts key policy rate",
                source="Xinhua",
                category_hint="central_bank",
            ),
            CONFIG,
        )
        self.assertIsNotNone(result)
        self.assertIn("PBOC", result.entities)


class WalCheckpointTests(unittest.TestCase):
    def test_prune_checkpoints_wal_once_per_sgt_day(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RadarStore(Path(directory) / "radar.sqlite3")
            store.prune(NOW, 30, item_retention_days=7, delivery_retention_days=90)
            self.assertEqual(store.get_meta("wal_checkpoint_date"), "2026-07-25")
            store.prune(
                NOW + timedelta(hours=1),
                30,
                item_retention_days=7,
                delivery_retention_days=90,
            )
            self.assertEqual(store.get_meta("wal_checkpoint_date"), "2026-07-25")
            store.close()


class WorldSlotTests(unittest.TestCase):
    def test_discovery_queries_include_world_slot(self) -> None:
        queries = discovery_queries(NOW.astimezone(SGT))
        self.assertEqual(len(queries), 4)
        name, query = queries[3]
        self.assertEqual(name, "world")
        self.assertIn("earthquake", query)

    def test_gdelt_world_query_targets_disaster_and_conflict_themes(self) -> None:
        self.assertIn("theme:NATURAL_DISASTER", GdeltCollector.WORLD_QUERY)
        self.assertIn("theme:ARMEDCONFLICT", GdeltCollector.WORLD_QUERY)


if __name__ == "__main__":
    unittest.main()
