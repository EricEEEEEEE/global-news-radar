from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
    observed_at TEXT NOT NULL
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
    day_number INTEGER NOT NULL DEFAULT 1
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
    last_alert_at TEXT
);
"""


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class RadarStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=20)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.commit()

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
        self, event_key: str, item: NewsItem, observed_at: datetime
    ) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO observations(
                identity, event_key, source, item_json, observed_at
            ) VALUES(?, ?, ?, ?, ?)
            """,
            (
                item.identity,
                event_key,
                item.source,
                json_dumps(item.to_dict()),
                _iso(observed_at),
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

    def topic_decision(
        self,
        *,
        event_key: str,
        material_hash: str,
        level: str,
        now: datetime,
        cooldown_hours: float,
    ) -> tuple[bool, str]:
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
    ) -> tuple[bool, str, int]:
        if not topic_anchor:
            return True, "no_anchor", 1
        row = self.connection.execute(
            "SELECT * FROM topic_lineage WHERE topic_anchor=?", (topic_anchor,)
        ).fetchone()
        if not row:
            return True, "new_lineage", 1
        last_sent = datetime.fromisoformat(str(row["last_sent_at"]))
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
        existing = self.connection.execute(
            "SELECT material_hash FROM topic_lineage WHERE topic_anchor=?",
            (topic_anchor,),
        ).fetchone()
        reset = existing is not None and str(existing["material_hash"]) == material_hash
        first_date = local_date if reset and day_number == 1 else None
        self.connection.execute(
            """
            INSERT INTO topic_lineage(
                topic_anchor, event_key, material_hash, first_date_sgt,
                last_date_sgt, last_sent_at, day_number
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_anchor) DO UPDATE SET
                event_key=excluded.event_key,
                material_hash=excluded.material_hash,
                first_date_sgt=CASE
                    WHEN ? IS NOT NULL THEN ?
                    ELSE topic_lineage.first_date_sgt
                END,
                last_date_sgt=excluded.last_date_sgt,
                last_sent_at=excluded.last_sent_at,
                day_number=excluded.day_number
            """,
            (
                topic_anchor,
                event_key,
                material_hash,
                first_date or local_date,
                local_date,
                _iso(now),
                day_number,
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

    def source_success(self, source_id: str, now: datetime) -> None:
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
        self.connection.commit()

    def source_failure(self, source_id: str, error: str, now: datetime) -> int:
        self.connection.execute(
            """
            INSERT INTO source_health(
                source_id, consecutive_failures, last_failure_at, last_error
            ) VALUES(?, 1, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                consecutive_failures=source_health.consecutive_failures+1,
                last_failure_at=excluded.last_failure_at,
                last_error=excluded.last_error
            """,
            (source_id, _iso(now), error[:500]),
        )
        self.connection.commit()
        row = self.connection.execute(
            "SELECT consecutive_failures FROM source_health WHERE source_id=?",
            (source_id,),
        ).fetchone()
        return int(row["consecutive_failures"])

    def source_alert_due(
        self, source_id: str, now: datetime, cooldown_hours: float
    ) -> bool:
        row = self.connection.execute(
            "SELECT last_alert_at FROM source_health WHERE source_id=?", (source_id,)
        ).fetchone()
        if not row or not row["last_alert_at"]:
            return True
        return now - datetime.fromisoformat(str(row["last_alert_at"])) >= timedelta(
            hours=cooldown_hours
        )

    def mark_source_alerted(self, source_id: str, now: datetime) -> None:
        self.connection.execute(
            "UPDATE source_health SET last_alert_at=? WHERE source_id=?",
            (_iso(now), source_id),
        )
        self.connection.commit()

    def prune(self, now: datetime, retention_days: int) -> None:
        cutoff = _iso(now - timedelta(days=retention_days))
        with self.transaction() as connection:
            connection.execute("DELETE FROM seen_items WHERE seen_at < ?", (cutoff,))
            connection.execute(
                "DELETE FROM observations WHERE observed_at < ?", (cutoff,)
            )
            connection.execute(
                "DELETE FROM seen_topics WHERE last_seen_at < ?", (cutoff,)
            )

    def export_deliveries(self, path: Path) -> int:
        rows = self.connection.execute(
            """
            SELECT d.sent_at, d.event_keys_json, d.content_hash, t.last_summary
            FROM deliveries d
            LEFT JOIN seen_topics t
              ON instr(d.event_keys_json, t.event_key) > 0
            ORDER BY d.sent_at ASC
            """
        ).fetchall()
        lines: list[str] = []
        for row in rows:
            event_keys = json.loads(str(row["event_keys_json"]))
            for event_key in event_keys:
                sent_at = datetime.fromisoformat(str(row["sent_at"]))
                lines.append(
                    json_dumps(
                        {
                            "date": sent_at.astimezone().date().isoformat(),
                            "event_key": event_key,
                            "content_hash": str(row["content_hash"])[:8],
                            "mention_count": 1,
                            "last_summary": str(row["last_summary"] or "")[:50],
                            "source": "radar",
                        }
                    )
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)
