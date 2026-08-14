from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .models import NewsItem, RenderedMessage
from .util import SGT, json_dumps

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS seen_items (
    identity TEXT PRIMARY KEY,
    canonical_url TEXT NOT NULL,
    norm_title TEXT NOT NULL,
    simhash TEXT NOT NULL,
    source TEXT NOT NULL,
    seen_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_seen_items_seen_at ON seen_items(seen_at);

CREATE TABLE IF NOT EXISTS observations (
    identity TEXT PRIMARY KEY,
    event_key TEXT NOT NULL,
    source TEXT NOT NULL,
    item_json TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    simhash TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_observations_event_time
    ON observations(event_key, observed_at);

CREATE TABLE IF NOT EXISTS seen_topics (
    event_key TEXT PRIMARY KEY,
    material_hash TEXT NOT NULL,
    level TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    source_count INTEGER NOT NULL DEFAULT 1,
    last_summary TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS topic_lineage (
    topic_anchor TEXT PRIMARY KEY,
    event_key TEXT NOT NULL,
    material_hash TEXT NOT NULL,
    first_date_sgt TEXT NOT NULL,
    last_date_sgt TEXT NOT NULL,
    last_sent_at TEXT NOT NULL,
    day_number INTEGER NOT NULL DEFAULT 1,
    sends_today INTEGER NOT NULL DEFAULT 0,
    sends_date_sgt TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS p1_buffer (
    event_key TEXT PRIMARY KEY,
    item_json TEXT NOT NULL,
    assessment_json TEXT NOT NULL,
    added_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_delivery (
    content_hash TEXT PRIMARY KEY,
    message_json TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    next_retry_at TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS deliveries (
    content_hash TEXT PRIMARY KEY,
    level TEXT NOT NULL,
    event_keys_json TEXT NOT NULL,
    message_id TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    visible_chars INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS source_health (
    source_id TEXT PRIMARY KEY,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_success_at TEXT,
    last_failure_at TEXT,
    last_error TEXT NOT NULL DEFAULT '',
    last_alert_at TEXT,
    window_start_at TEXT,
    window_attempts INTEGER NOT NULL DEFAULT 0,
    window_failures INTEGER NOT NULL DEFAULT 0,
    failure_started_at TEXT,
    recovered_at TEXT
);

CREATE TABLE IF NOT EXISTS story_cluster (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_at TEXT NOT NULL,
    tokens_json TEXT NOT NULL,
    title TEXT NOT NULL,
    promoted_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_story_cluster_last ON story_cluster(last_at);

CREATE TABLE IF NOT EXISTS cluster_member (
    item_identity TEXT PRIMARY KEY,
    cluster_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    is_major INTEGER NOT NULL DEFAULT 0,
    title TEXT NOT NULL,
    url TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL,
    added_at TEXT NOT NULL,
    item_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cluster_member_cluster ON cluster_member(cluster_id);
"""


# Infrastructure alerts and daily briefs travel through the same delivery path
# as market events. They must never reach seen_topics or the exported
# market-event ledger.
INFRA_EVENT_PREFIXES = ("source-error:", "source-recovery:", "brief:")


def is_infra_event(event_key: str) -> bool:
    return event_key.startswith(INFRA_EVENT_PREFIXES)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class RadarStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=20)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self._migrate()
        self.connection.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created."""
        additions = {
            "topic_lineage": (
                ("sends_today", "INTEGER NOT NULL DEFAULT 0"),
                ("sends_date_sgt", "TEXT NOT NULL DEFAULT ''"),
            ),
            "source_health": (
                ("window_start_at", "TEXT"),
                ("window_attempts", "INTEGER NOT NULL DEFAULT 0"),
                ("window_failures", "INTEGER NOT NULL DEFAULT 0"),
                ("failure_started_at", "TEXT"),
                ("recovered_at", "TEXT"),
            ),
            "observations": (("simhash", "TEXT NOT NULL DEFAULT ''"),),
        }
        for table, columns in additions.items():
            existing = {
                str(row["name"])
                for row in self.connection.execute(f"PRAGMA table_info({table})")
            }
            for name, definition in columns:
                if name not in existing:
                    self.connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {definition}"
                    )
        # A deployment inherits rows that were alerted before recovered_at
        # existed. Left NULL they look like still-open incidents, so the first
        # blip after the upgrade would announce the recovery of an outage that
        # closed before the upgrade. Closing them at their own alert time also
        # keeps their next alert on the pre-existing cooldown rather than the
        # shorter post-recovery gap.
        self.connection.execute(
            """
            UPDATE source_health SET recovered_at=last_alert_at
            WHERE consecutive_failures = 0
              AND last_alert_at IS NOT NULL
              AND recovered_at IS NULL
            """
        )

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return str(row["value"]) if row else default

    def set_meta(self, key: str, value: str, now: datetime) -> None:
        self.connection.execute(
            """
            INSERT INTO meta(key, value, updated_at) VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=excluded.updated_at
            """,
            (key, value, _iso(now)),
        )
        self.connection.commit()

    def baseline_complete(self) -> bool:
        return self.get_meta("baseline_complete") == "1"

    def mark_baseline_complete(self, now: datetime) -> None:
        self.set_meta("baseline_complete", "1", now)

    def item_seen(self, identity: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM seen_items WHERE identity=?", (identity,)
            ).fetchone()
            is not None
        )

    def recent_simhashes(
        self, since: datetime, limit: int = 500
    ) -> list[tuple[str, str]]:
        rows = self.connection.execute(
            """
            SELECT identity, simhash FROM seen_items
            WHERE seen_at >= ? ORDER BY seen_at DESC LIMIT ?
            """,
            (_iso(since), limit),
        ).fetchall()
        return [(str(row["identity"]), str(row["simhash"])) for row in rows]

    def mark_item(
        self,
        *,
        identity: str,
        canonical_url: str,
        norm_title: str,
        simhash: str,
        source: str,
        now: datetime,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO seen_items(
                identity, canonical_url, norm_title, simhash, source, seen_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (identity, canonical_url, norm_title, simhash, source, _iso(now)),
        )
        self.connection.commit()

    def add_observation(
        self,
        event_key: str,
        item: NewsItem,
        observed_at: datetime,
        simhash: str = "",
    ) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO observations(
                identity, event_key, source, item_json, observed_at, simhash
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                item.identity,
                event_key,
                item.source,
                json_dumps(item.to_dict()),
                _iso(observed_at),
                simhash,
            ),
        )
        self.connection.commit()

    def recent_observations(self, event_key: str, since: datetime) -> list[NewsItem]:
        rows = self.connection.execute(
            """
            SELECT item_json FROM observations
            WHERE event_key=? AND observed_at>=?
            ORDER BY observed_at DESC
            """,
            (event_key, _iso(since)),
        ).fetchall()
        return [NewsItem.from_dict(json.loads(row["item_json"])) for row in rows]

    def observation_cluster(
        self, since: datetime, category: str
    ) -> list[tuple[str, str, NewsItem]]:
        """Observations in one category, for near-duplicate corroboration.

        Two outlets covering the same event do not always produce the same
        event_key — one may name an extra country the other omits — so exact-key
        matching alone loses real second-source confirmations.
        """
        rows = self.connection.execute(
            """
            SELECT event_key, simhash, item_json FROM observations
            WHERE observed_at>=? AND event_key LIKE ? AND simhash<>''
            ORDER BY observed_at DESC LIMIT 400
            """,
            (_iso(since), f"{category}:%"),
        ).fetchall()
        return [
            (
                str(row["event_key"]),
                str(row["simhash"]),
                NewsItem.from_dict(json.loads(row["item_json"])),
            )
            for row in rows
        ]

    def event_key_pending(self, event_key: str) -> bool:
        """True when this event is already sitting in the delivery queue.

        seen_topics is only written after a successful send, so during the quiet
        window or a Telegram outage nothing else stops a second phrasing of the
        same event from being queued a second time.
        """
        return (
            self.connection.execute(
                """
                SELECT 1 FROM pending_delivery p
                JOIN json_each(json_extract(p.message_json, '$.event_keys')) k
                WHERE k.value = ? LIMIT 1
                """,
                (event_key,),
            ).fetchone()
            is not None
        )

    def topic_decision(
        self,
        *,
        event_key: str,
        material_hash: str,
        level: str,
        now: datetime,
        cooldown_hours: float,
    ) -> tuple[bool, str]:
        if self.event_key_pending(event_key):
            return False, "already_queued"
        row = self.connection.execute(
            "SELECT * FROM seen_topics WHERE event_key=?", (event_key,)
        ).fetchone()
        if not row:
            return True, "new_topic"
        sent_at = datetime.fromisoformat(str(row["sent_at"]))
        if now - sent_at >= timedelta(hours=cooldown_hours):
            return True, "cooldown_elapsed"
        previous_level = str(row["level"])
        if level == "P0" and previous_level != "P0":
            return True, "severity_upgrade"
        if str(row["material_hash"]) != material_hash:
            return True, "material_update"
        return False, "topic_cooldown"

    def record_topic_sent(
        self,
        *,
        event_key: str,
        material_hash: str,
        level: str,
        source_count: int,
        summary: str,
        now: datetime,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO seen_topics(
                event_key, material_hash, level, sent_at, last_seen_at,
                source_count, last_summary
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_key) DO UPDATE SET
                material_hash=excluded.material_hash,
                level=excluded.level,
                sent_at=excluded.sent_at,
                last_seen_at=excluded.last_seen_at,
                source_count=excluded.source_count,
                last_summary=excluded.last_summary
            """,
            (
                event_key,
                material_hash,
                level,
                _iso(now),
                _iso(now),
                source_count,
                summary[:200],
            ),
        )
        self.connection.commit()

    def lineage_decision(
        self,
        *,
        topic_anchor: str,
        material_hash: str,
        now: datetime,
        max_gap_days: float = 7.0,
        min_interval_minutes: float = 30.0,
        daily_cap: int = 6,
    ) -> tuple[bool, str, int]:
        if not topic_anchor:
            return True, "no_anchor", 1
        row = self.connection.execute(
            "SELECT * FROM topic_lineage WHERE topic_anchor=?", (topic_anchor,)
        ).fetchone()
        if not row:
            return True, "new_lineage", 1
        last_sent = datetime.fromisoformat(str(row["last_sent_at"]))
        # Without an upper bound a dormant anchor keeps counting calendar days,
        # so an unrelated recurrence months later renders as 【Day 43】.
        if max_gap_days > 0 and now - last_sent >= timedelta(days=max_gap_days):
            return True, "lineage_expired", 1
        # material_update alone re-fires whenever a second outlet quotes one more
        # number, so the same story can push repeatedly within minutes.
        if min_interval_minutes > 0 and now - last_sent < timedelta(
            minutes=min_interval_minutes
        ):
            return False, "anchor_min_interval", int(row["day_number"])
        local_today = now.astimezone(SGT).date().isoformat()
        if (
            daily_cap > 0
            and str(row["sends_date_sgt"] or "") == local_today
            and int(row["sends_today"] or 0) >= daily_cap
        ):
            return False, "anchor_daily_cap", int(row["day_number"])
        same_material = str(row["material_hash"]) == material_hash
        if same_material and now - last_sent < timedelta(hours=24):
            return False, "same_fact_24h", int(row["day_number"])
        current_date = now.astimezone(SGT).date()
        first_date = datetime.fromisoformat(str(row["first_date_sgt"])).date()
        if not same_material and current_date > first_date:
            return True, "cross_day_update", (current_date - first_date).days + 1
        if not same_material:
            return True, "same_day_update", int(row["day_number"])
        return True, "new_recurrence", 1

    def record_lineage_sent(
        self,
        *,
        topic_anchor: str,
        event_key: str,
        material_hash: str,
        day_number: int,
        now: datetime,
    ) -> None:
        if not topic_anchor:
            return
        local_date = now.astimezone(SGT).date().isoformat()
        # Day 1 always means "this lineage starts today", whatever caused the
        # restart (fresh recurrence or an expired anchor).
        first_date = local_date if day_number == 1 else None
        self.connection.execute(
            """
            INSERT INTO topic_lineage(
                topic_anchor, event_key, material_hash, first_date_sgt,
                last_date_sgt, last_sent_at, day_number,
                sends_today, sends_date_sgt
            ) VALUES(?, ?, ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(topic_anchor) DO UPDATE SET
                event_key=excluded.event_key,
                material_hash=excluded.material_hash,
                first_date_sgt=CASE
                    WHEN ? IS NOT NULL THEN ?
                    ELSE topic_lineage.first_date_sgt
                END,
                last_date_sgt=excluded.last_date_sgt,
                last_sent_at=excluded.last_sent_at,
                day_number=excluded.day_number,
                sends_today=CASE
                    WHEN topic_lineage.sends_date_sgt = excluded.sends_date_sgt
                    THEN topic_lineage.sends_today + 1
                    ELSE 1
                END,
                sends_date_sgt=excluded.sends_date_sgt
            """,
            (
                topic_anchor,
                event_key,
                material_hash,
                first_date or local_date,
                local_date,
                _iso(now),
                day_number,
                local_date,
                first_date,
                first_date,
            ),
        )
        self.connection.commit()

    def buffer_p1(
        self,
        *,
        event_key: str,
        item: NewsItem,
        assessment: dict[str, object],
        now: datetime,
        expires_at: datetime,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO p1_buffer(
                event_key, item_json, assessment_json, added_at, expires_at
            ) VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(event_key) DO UPDATE SET
                item_json=excluded.item_json,
                assessment_json=excluded.assessment_json,
                expires_at=excluded.expires_at
            """,
            (
                event_key,
                json_dumps(item.to_dict()),
                json_dumps(assessment),
                _iso(now),
                _iso(expires_at),
            ),
        )
        self.connection.commit()

    def pop_expired_p1(self, now: datetime) -> list[tuple[str, str]]:
        """Expired buffer entries, removed and returned so the drop gets logged.

        ready_p1 deletes these silently as a safety net; a P1 that waited for
        a second source and never got one otherwise vanishes without a trace.
        """
        rows = self.connection.execute(
            "SELECT event_key, added_at FROM p1_buffer WHERE expires_at < ?",
            (_iso(now),),
        ).fetchall()
        if rows:
            self.connection.execute(
                "DELETE FROM p1_buffer WHERE expires_at < ?", (_iso(now),)
            )
            self.connection.commit()
        return [(str(row["event_key"]), str(row["added_at"])) for row in rows]

    def ready_p1(
        self, now: datetime, since: datetime, limit: int = 5
    ) -> list[sqlite3.Row]:
        self.connection.execute(
            "DELETE FROM p1_buffer WHERE expires_at < ?", (_iso(now),)
        )
        self.connection.commit()
        return self.connection.execute(
            """
            SELECT * FROM p1_buffer
            WHERE added_at >= ? ORDER BY added_at ASC LIMIT ?
            """,
            (_iso(since), limit),
        ).fetchall()

    def remove_p1(self, event_keys: list[str]) -> None:
        if not event_keys:
            return
        placeholders = ",".join("?" for _ in event_keys)
        self.connection.execute(
            f"DELETE FROM p1_buffer WHERE event_key IN ({placeholders})", event_keys
        )
        self.connection.commit()

    def queue_delivery(
        self,
        message: RenderedMessage,
        *,
        thread_id: str,
        next_retry_at: datetime,
        error: str = "",
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO pending_delivery(
                content_hash, message_json, thread_id, created_at,
                next_retry_at, attempts, last_error
            ) VALUES(?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(content_hash) DO UPDATE SET
                next_retry_at=excluded.next_retry_at,
                last_error=excluded.last_error
            """,
            (
                message.content_hash,
                json_dumps(message.to_dict()),
                thread_id,
                _iso(message.created_at),
                _iso(next_retry_at),
                error[:500],
            ),
        )
        self.connection.commit()

    def due_deliveries(self, now: datetime) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT * FROM pending_delivery
            WHERE next_retry_at <= ? ORDER BY created_at ASC
            """,
            (_iso(now),),
        ).fetchall()

    def mark_delivery_retry(
        self, content_hash: str, next_retry_at: datetime, error: str
    ) -> None:
        self.connection.execute(
            """
            UPDATE pending_delivery
            SET attempts=attempts+1, next_retry_at=?, last_error=?
            WHERE content_hash=?
            """,
            (_iso(next_retry_at), error[:500], content_hash),
        )
        self.connection.commit()

    def delivery_attempts(self, content_hash: str) -> int:
        row = self.connection.execute(
            "SELECT attempts FROM pending_delivery WHERE content_hash=?",
            (content_hash,),
        ).fetchone()
        return int(row["attempts"]) if row else 0

    def record_delivery(
        self,
        *,
        message: RenderedMessage,
        message_id: str,
        visible_chars: int,
        now: datetime,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO deliveries(
                    content_hash, level, event_keys_json, message_id,
                    sent_at, visible_chars
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    message.content_hash,
                    message.level,
                    json_dumps(message.event_keys),
                    message_id,
                    _iso(now),
                    visible_chars,
                ),
            )
            connection.execute(
                "DELETE FROM pending_delivery WHERE content_hash=?",
                (message.content_hash,),
            )

    def delivery_exists(self, content_hash: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM deliveries WHERE content_hash=?", (content_hash,)
            ).fetchone()
            is not None
        )

    def pending_counts(self, max_attempts: int) -> tuple[int, int]:
        row = self.connection.execute(
            """
            SELECT
                count(*) AS total,
                sum(CASE WHEN attempts >= ? THEN 1 ELSE 0 END) AS dead
            FROM pending_delivery
            """,
            (max_attempts,),
        ).fetchone()
        return int(row["total"] or 0), int(row["dead"] or 0)

    def _bump_window(
        self, source_id: str, now: datetime, window_hours: float, failed: bool
    ) -> None:
        row = self.connection.execute(
            "SELECT window_start_at FROM source_health WHERE source_id=?",
            (source_id,),
        ).fetchone()
        started = row["window_start_at"] if row else None
        window = timedelta(hours=window_hours)
        expired = not started or now - datetime.fromisoformat(str(started)) >= window
        if expired:
            self.connection.execute(
                """
                UPDATE source_health
                SET window_start_at=?, window_attempts=1, window_failures=?
                WHERE source_id=?
                """,
                (_iso(now), 1 if failed else 0, source_id),
            )
            return
        self.connection.execute(
            """
            UPDATE source_health
            SET window_attempts=window_attempts+1,
                window_failures=window_failures + ?
            WHERE source_id=?
            """,
            (1 if failed else 0, source_id),
        )

    def source_success(
        self, source_id: str, now: datetime, window_hours: float = 6.0
    ) -> dict[str, Any] | None:
        """Record a successful call, reporting a recovery worth announcing.

        A recovery is only worth announcing when the operator was told about the
        outage in the first place, so the caller alerted on it (``last_alert_at``)
        and that alert belongs to the incident that is closing now rather than to
        an older one that already got its own recovery notice.
        """
        previous = self.connection.execute(
            """
            SELECT consecutive_failures, last_alert_at, failure_started_at,
                   recovered_at
            FROM source_health WHERE source_id=?
            """,
            (source_id,),
        ).fetchone()
        recovery: dict[str, Any] | None = None
        if previous and int(previous["consecutive_failures"] or 0) > 0:
            alerted = previous["last_alert_at"]
            recovered = previous["recovered_at"]
            open_incident = bool(alerted) and (
                not recovered
                or datetime.fromisoformat(str(alerted))
                > datetime.fromisoformat(str(recovered))
            )
            if open_incident:
                started = previous["failure_started_at"]
                recovery = {
                    "source_id": source_id,
                    "failures": int(previous["consecutive_failures"]),
                    "outage_minutes": (
                        round(
                            (now - datetime.fromisoformat(str(started))).total_seconds()
                            / 60
                        )
                        if started
                        else None
                    ),
                }
        self.connection.execute(
            """
            INSERT INTO source_health(source_id, consecutive_failures, last_success_at)
            VALUES(?, 0, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                consecutive_failures=0,
                last_success_at=excluded.last_success_at,
                last_error=''
            """,
            (source_id, _iso(now)),
        )
        self._bump_window(source_id, now, window_hours, failed=False)
        self.connection.commit()
        return recovery

    def source_failure(
        self, source_id: str, error: str, now: datetime, window_hours: float = 6.0
    ) -> int:
        # failure_started_at anchors the outage clock. It moves only on the
        # 0 -> 1 transition, so the recovery notice can report how long the
        # source was actually down rather than how long ago it last succeeded.
        self.connection.execute(
            """
            INSERT INTO source_health(
                source_id, consecutive_failures, last_failure_at, last_error,
                failure_started_at
            ) VALUES(?, 1, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                consecutive_failures=source_health.consecutive_failures+1,
                last_failure_at=excluded.last_failure_at,
                last_error=excluded.last_error,
                failure_started_at=CASE
                    WHEN source_health.consecutive_failures = 0
                    THEN excluded.failure_started_at
                    ELSE source_health.failure_started_at
                END
            """,
            (source_id, _iso(now), error[:500], _iso(now)),
        )
        self._bump_window(source_id, now, window_hours, failed=True)
        self.connection.commit()
        row = self.connection.execute(
            "SELECT consecutive_failures FROM source_health WHERE source_id=?",
            (source_id,),
        ).fetchone()
        return int(row["consecutive_failures"])

    def source_window(self, source_id: str) -> tuple[int, int]:
        """Attempts and failures inside the current health window."""
        row = self.connection.execute(
            """
            SELECT window_attempts, window_failures FROM source_health
            WHERE source_id=?
            """,
            (source_id,),
        ).fetchone()
        if not row:
            return 0, 0
        return int(row["window_attempts"] or 0), int(row["window_failures"] or 0)

    def source_alert_due(
        self,
        source_id: str,
        now: datetime,
        cooldown_hours: float,
        realert_minutes: float = 0.0,
    ) -> bool:
        row = self.connection.execute(
            "SELECT last_alert_at, recovered_at FROM source_health WHERE source_id=?",
            (source_id,),
        ).fetchone()
        if not row or not row["last_alert_at"]:
            return True
        last_alert = datetime.fromisoformat(str(row["last_alert_at"]))
        recovered = row["recovered_at"]
        if recovered:
            recovered_at = datetime.fromisoformat(str(recovered))
            if recovered_at > last_alert:
                # The alerted incident already closed and was announced as
                # recovered. A fresh outage is a new incident and deserves its
                # own alert, but a flapping source must not alert every time it
                # bounces, so it still has to clear the post-recovery gap.
                return now - recovered_at >= timedelta(minutes=realert_minutes)
        return now - last_alert >= timedelta(hours=cooldown_hours)

    def mark_source_alerted(self, source_id: str, now: datetime) -> None:
        self.connection.execute(
            "UPDATE source_health SET last_alert_at=? WHERE source_id=?",
            (_iso(now), source_id),
        )
        self.connection.commit()

    def mark_source_recovered(self, source_id: str, now: datetime) -> None:
        # last_alert_at is deliberately left in place: source_alert_due compares
        # the two timestamps to tell "incident still open" from "incident closed,
        # a new one may alert again".
        self.connection.execute(
            "UPDATE source_health SET recovered_at=? WHERE source_id=?",
            (_iso(now), source_id),
        )
        self.connection.commit()

    def prune(
        self,
        now: datetime,
        retention_days: int,
        *,
        item_retention_days: int = 0,
        delivery_retention_days: int = 0,
    ) -> None:
        cutoff = _iso(now - timedelta(days=retention_days))
        # Discovery pulls two orders of magnitude more headlines than official
        # feeds, so seen_items gets its own shorter horizon.
        item_cutoff = _iso(now - timedelta(days=item_retention_days or retention_days))
        delivery_cutoff = _iso(
            now - timedelta(days=delivery_retention_days or retention_days)
        )
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM seen_items WHERE seen_at < ?", (item_cutoff,)
            )
            connection.execute(
                "DELETE FROM observations WHERE observed_at < ?", (item_cutoff,)
            )
            connection.execute(
                "DELETE FROM seen_topics WHERE last_seen_at < ?", (cutoff,)
            )
            connection.execute(
                "DELETE FROM topic_lineage WHERE last_sent_at < ?", (cutoff,)
            )
            connection.execute(
                "DELETE FROM deliveries WHERE sent_at < ?", (delivery_cutoff,)
            )
            connection.execute(
                """
                DELETE FROM source_health
                WHERE coalesce(last_success_at, last_failure_at, '') < ?
                """,
                (delivery_cutoff,),
            )
        # The daemon holds one connection forever, so nothing else ever
        # checkpoints the WAL and it only grows. One TRUNCATE per SGT day
        # keeps it bounded without stalling every cycle.
        today = now.astimezone(SGT).date().isoformat()
        if self.get_meta("wal_checkpoint_date") != today:
            self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.set_meta("wal_checkpoint_date", today, now)

    def deliveries_between(
        self, start: datetime, end: datetime
    ) -> list[dict[str, str]]:
        """Sent market events inside a window, one row per event key.

        The briefs use this as the "已报" section, so infra alerts and the
        briefs themselves stay excluded via the shared prefix filter.
        """
        infra_filter = " AND ".join(
            f"k.value NOT LIKE '{prefix}%'" for prefix in INFRA_EVENT_PREFIXES
        )
        rows = self.connection.execute(
            f"""
            SELECT d.sent_at, d.level, k.value AS event_key, t.last_summary
            FROM deliveries d
            JOIN json_each(d.event_keys_json) k
            LEFT JOIN seen_topics t ON t.event_key = k.value
            WHERE d.sent_at >= ? AND d.sent_at < ? AND {infra_filter}
            ORDER BY d.sent_at ASC, k.value ASC
            """,
            (_iso(start), _iso(end)),
        ).fetchall()
        return [
            {
                "sent_at": str(row["sent_at"]),
                "level": str(row["level"]),
                "event_key": str(row["event_key"]),
                "summary": str(row["last_summary"] or ""),
            }
            for row in rows
        ]

    def export_deliveries(self, path: Path) -> int:
        # json_each expands one row per event_key and joins on an exact match.
        # The old instr() join matched any key that was a substring of the JSON
        # blob, multiplying rows and pairing keys with the wrong summary.
        infra_filter = " AND ".join(
            f"k.value NOT LIKE '{prefix}%'" for prefix in INFRA_EVENT_PREFIXES
        )
        rows = self.connection.execute(
            f"""
            SELECT d.sent_at, k.value AS event_key, d.content_hash, t.last_summary
            FROM deliveries d
            JOIN json_each(d.event_keys_json) k
            LEFT JOIN seen_topics t ON t.event_key = k.value
            WHERE {infra_filter}
            ORDER BY d.sent_at ASC, k.value ASC
            """
        ).fetchall()
        lines = [
            json_dumps(
                {
                    "date": datetime.fromisoformat(str(row["sent_at"]))
                    .astimezone()
                    .date()
                    .isoformat(),
                    "event_key": str(row["event_key"]),
                    "content_hash": str(row["content_hash"])[:8],
                    "mention_count": 1,
                    "last_summary": str(row["last_summary"] or "")[:50],
                    "source": "radar",
                }
            )
            for row in rows
        ]
        content = "\n".join(lines) + ("\n" if lines else "")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Rewriting an identical ledger on every send burns disk for nothing.
        if path.exists() and path.read_text(encoding="utf-8") == content:
            return len(lines)
        path.write_text(content, encoding="utf-8")
        return len(lines)
