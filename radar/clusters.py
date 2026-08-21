"""Incremental story clustering: one story, many outlets, one unit of work.

The radar's original unit was the single headline, which made cross-outlet
consensus invisible: five majors filing the same story looked like five
unrelated items. A story cluster is a set of shared content tokens; every
fresh headline either joins the active cluster it overlaps most or founds a
new one. Zero new dependencies — plain token overlap, which is where every
production clusterer (EMM, Meridian) lands at this scale.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from .embeddings import TitleEmbedder
from .models import NewsItem
from .store import RadarStore
from .util import json_dumps

LOGGER = logging.getLogger(__name__)

# Content words only: 3+ letter words and 2+ digit numbers. Shorter tokens
# ("US", "Q2") are too ambiguous to define a story on their own.
_TOKEN_RE = re.compile(r"[a-z]{3,}|\d{2,}")

# Grammar words plus newsroom furniture. A token that appears in half of all
# headlines can never distinguish one story from another.
_STOPWORDS = frozenset(
    """
    the and for with from that this will would could should may might than
    more most new not but out off all who what when where why how been being
    about against between during before under above while still since among
    across also just per via they them then there here was were are has have
    had its his her their say says said after over into amid
    breaking news live update updates report reports reported watch video
    analysis opinion exclusive latest today first top big major key market
    markets stocks stock year years month day week time world global
    president minister government officials people country state city
    """.split()
)

# Impact verbs the wires reuse across unrelated beats. "Quake strikes Japan"
# and "Israel strikes Lebanon" share nothing but the verb, yet before this it
# counted toward the shared-token threshold and bridged the two into one
# cluster. Purely grammatical verbs join them: they never identify a story.
_HOMONYM_VERBS = frozenset(
    """
    strike strikes struck hit hits hitting slam slams slammed blast blasts
    rock rocks rocked shake shakes shaken spark sparks sparked target targets
    targeted face faces facing see sees seen urge urges warn warns call calls
    seek seeks unveil unveils reveal reveals announce announces launch
    launches set sets take takes make makes come comes go goes hold holds
    keep keeps leave leaves bring brings add adds get gets put puts
    """.split()
)

# Outlet names that are never the subject of a story. Deliberately limited to
# single unambiguous words: deriving these from the configured source list
# would delete "China", "Post" and "Times" from every headline that means them.
_OUTLET_TOKENS = frozenset(
    """
    reuters bloomberg cnbc marketwatch yonhap aljazeera jazeera kyodo
    politico axios cnn bbc afp upi newsweek forbes barrons
    """.split()
)

_STOPWORDS = _STOPWORDS | _HOMONYM_VERBS | _OUTLET_TOKENS


def tokenize(title: str) -> set[str]:
    return {
        token for token in _TOKEN_RE.findall(title.lower()) if token not in _STOPWORDS
    }


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _member_source_id(member: Any) -> str:
    """The feed id a cluster member came from, or "" when unrecoverable.

    Stored inside item_json rather than its own column, so this stays a read
    of existing rows: no migration, and pre-1.8.0 members simply return "".
    """
    try:
        return str(json.loads(str(member["item_json"])).get("source_id") or "")
    except Exception:  # noqa: BLE001
        return ""


class ClusterEngine:
    """Groups headlines into stories and detects multi-major bursts."""

    def __init__(
        self,
        store: RadarStore,
        *,
        window_hours: int,
        min_shared_tokens: int,
        burst_min_majors: int,
        burst_window_minutes: int,
        major_sources: list[str],
        embedder: TitleEmbedder | None = None,
    ):
        self.store = store
        self.window_hours = window_hours
        self.min_shared = min_shared_tokens
        self.burst_min_majors = burst_min_majors
        self.burst_window = timedelta(minutes=burst_window_minutes)
        self._majors = tuple(name.casefold() for name in major_sources)
        # Optional and fail-open: when absent or broken, clustering is exactly
        # the token-overlap behaviour it has always been.
        self.embedder = embedder

    def is_major(self, source: str) -> bool:
        folded = source.casefold()
        return any(major in folded for major in self._majors)

    def add(self, item: NewsItem, now: datetime) -> int | None:
        """Attach one headline to its story; returns the cluster id.

        Idempotent by item identity. Returns None when the title has too few
        content tokens to ever share min_shared with anything.
        """
        tokens = tokenize(item.title)
        if len(tokens) < self.min_shared:
            return None
        connection = self.store.connection
        existing = connection.execute(
            "SELECT cluster_id FROM cluster_member WHERE item_identity = ?",
            (item.identity,),
        ).fetchone()
        if existing is not None:
            return int(existing["cluster_id"])
        cutoff = _iso(now - timedelta(hours=self.window_hours))
        best_id: int | None = None
        best_key = (0, 0)
        best_tokens: list[str] = []
        candidates: list[tuple[int, str, list[str]]] = []
        for row in connection.execute(
            "SELECT id, tokens_json, title FROM story_cluster WHERE last_at >= ?",
            (cutoff,),
        ):
            cluster_tokens = json.loads(str(row["tokens_json"]))
            candidates.append((int(row["id"]), str(row["title"]), cluster_tokens))
            # Match against the tokens of the *founding* headline, not the
            # grown set. The grown set is a union, so a sprawling cluster
            # accumulates vocabulary and starts matching stories that share
            # nothing with the story it began as -- chain absorption, which is
            # how one earthquake swallowed a news day. The grown set still
            # ranks between equally-founded candidates.
            founding = tokenize(str(row["title"]))
            key = (
                len(tokens.intersection(founding)),
                len(tokens.intersection(cluster_tokens)),
            )
            if key > best_key:
                best_id = int(row["id"])
                best_key = key
                best_tokens = list(cluster_tokens)
        best_overlap = best_key[0]
        if best_overlap < self.min_shared:
            rescued = self._embedding_match(item.title, candidates)
            if rescued is not None:
                best_id, best_tokens = rescued
                best_overlap = self.min_shared
        if best_id is not None and best_overlap >= self.min_shared:
            cluster_id = best_id
            # The token set grows with new phrasings but stays capped, and the
            # cap trims the tail, so the founding tokens (the head) are stable
            # and one sprawling story cannot chain-absorb the whole news day.
            merged = best_tokens + sorted(
                token for token in tokens if token not in set(best_tokens)
            )
            connection.execute(
                "UPDATE story_cluster SET tokens_json = ?, last_at = ? WHERE id = ?",
                (json_dumps(merged[:40]), _iso(now), cluster_id),
            )
        else:
            cursor = connection.execute(
                "INSERT INTO story_cluster "
                "(created_at, last_at, tokens_json, title, promoted_at) "
                "VALUES (?, ?, ?, ?, NULL)",
                (_iso(now), _iso(now), json_dumps(sorted(tokens)), item.title),
            )
            cluster_id = int(cursor.lastrowid or 0)
        connection.execute(
            "INSERT OR IGNORE INTO cluster_member "
            "(item_identity, cluster_id, source, is_major, title, url, "
            "published_at, added_at, item_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item.identity,
                cluster_id,
                item.source,
                int(self.is_major(item.source)),
                item.title,
                item.url,
                _iso(item.published_at),
                _iso(now),
                json_dumps(item.to_dict()),
            ),
        )
        connection.commit()
        return cluster_id

    def _embedding_match(
        self, title: str, candidates: list[tuple[int, str, list[str]]]
    ) -> tuple[int, list[str]] | None:
        """Last-resort merge for stories that share meaning but not words.

        "6.1 quake off Hokkaido" and "Tsunami advisory lifted for northern
        Japan" are one story with almost no token overlap. Only ever adds a
        merge that token overlap already declined, so switching the embedder
        off restores the previous behaviour exactly.
        """
        if self.embedder is None or not self.embedder.available:
            return None
        best: tuple[float, int, list[str]] | None = None
        for cluster_id, founding_title, cluster_tokens in candidates:
            score = self.embedder.similarity(title, founding_title)
            if score is None or score < self.embedder.threshold:
                continue
            if best is None or score > best[0]:
                best = (score, cluster_id, list(cluster_tokens))
        if best is None:
            return None
        LOGGER.info("cluster_embedding_merge cluster=%s score=%.3f", best[1], best[0])
        return best[1], best[2]

    def burst_candidates(self, now: datetime) -> list[dict[str, object]]:
        """Unpromoted clusters where enough distinct majors filed in a burst.

        A burst is burst_min_majors distinct major outlets whose first filings
        on the story all fall inside one sliding burst_window. Returns the
        representative item (earliest major filing), the major headlines for
        the editorial judge, and a stable scope for the event key.
        """
        cutoff = _iso(now - timedelta(hours=self.window_hours))
        connection = self.store.connection
        result: list[dict[str, object]] = []
        for row in connection.execute(
            "SELECT id, tokens_json FROM story_cluster "
            "WHERE promoted_at IS NULL AND last_at >= ?",
            (cutoff,),
        ).fetchall():
            cluster_id = int(row["id"])
            members = connection.execute(
                "SELECT source, is_major, title, published_at, item_json "
                "FROM cluster_member WHERE cluster_id = ? "
                "ORDER BY published_at ASC",
                (cluster_id,),
            ).fetchall()
            first_filings: dict[str, object] = {}
            for member in members:
                if not int(member["is_major"]):
                    continue
                first_filings.setdefault(str(member["source"]).casefold(), member)
            if len(first_filings) < self.burst_min_majors:
                continue
            filings = sorted(
                (
                    datetime.fromisoformat(str(member["published_at"])),
                    str(member["source"]).casefold(),
                )
                for member in first_filings.values()
            )
            burst = False
            for index, (window_start, _) in enumerate(filings):
                inside = {
                    source
                    for filed_at, source in filings[index:]
                    if filed_at - window_start <= self.burst_window
                }
                if len(inside) >= self.burst_min_majors:
                    burst = True
                    break
            if not burst:
                continue
            representative_row = min(
                first_filings.values(), key=lambda member: str(member["published_at"])
            )
            representative = NewsItem.from_dict(
                json.loads(str(representative_row["item_json"]))
            )
            # Head of the stored token list = the founding tokens, which never
            # move (appends go to the tail, the cap trims the tail). A stable
            # scope keeps the burst event key identical across retry cycles.
            tokens = json.loads(str(row["tokens_json"]))
            scope = "-".join(str(token) for token in tokens[:5])
            headlines = [
                (str(member["source"]), str(member["title"]))
                for member in sorted(
                    first_filings.values(),
                    key=lambda member: str(member["published_at"]),
                )
            ]
            result.append(
                {
                    "cluster_id": cluster_id,
                    "representative": representative,
                    "headlines": headlines[:8],
                    "scope": scope,
                }
            )
        return result

    def mark_promoted(self, cluster_id: int, now: datetime) -> None:
        self.store.connection.execute(
            "UPDATE story_cluster SET promoted_at = ? WHERE id = ?",
            (_iso(now), cluster_id),
        )
        self.store.connection.commit()

    def top_clusters(
        self, start: datetime, end: datetime, limit: int
    ) -> list[dict[str, object]]:
        """Ranked stories for the daily brief window.

        Consensus ranks: distinct majors weigh triple, then sheer member
        count. Single-member clusters with no major source are noise and
        stay out entirely.
        """
        connection = self.store.connection
        rows = connection.execute(
            "SELECT c.id, c.promoted_at, "
            "COUNT(m.item_identity) AS member_count, "
            "COUNT(DISTINCT CASE WHEN m.is_major = 1 THEN lower(m.source) END) "
            "AS major_count "
            "FROM story_cluster c JOIN cluster_member m ON m.cluster_id = c.id "
            "WHERE c.last_at >= ? AND c.created_at < ? "
            "GROUP BY c.id "
            "HAVING member_count >= 2 OR major_count >= 1 "
            "ORDER BY (major_count * 3 + member_count) DESC, c.id DESC "
            "LIMIT ?",
            (_iso(start), _iso(end), limit),
        ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            members = connection.execute(
                "SELECT source, is_major, title, url, published_at, item_json "
                "FROM cluster_member WHERE cluster_id = ? "
                "ORDER BY published_at ASC",
                (int(row["id"]),),
            ).fetchall()
            if not members:
                continue
            # The most informative filing, not the earliest. The first wire
            # snap is the shortest ("Strong quake strikes off Japan"); a later
            # major filing carries the toll, the place and the response, which
            # is what a reader of a twice-daily brief actually needs.
            representative = max(
                members,
                key=lambda member: (
                    int(member["is_major"]),
                    len(tokenize(str(member["title"]))),
                    str(member["published_at"]),
                ),
            )
            result.append(
                {
                    "id": int(row["id"]),
                    "member_count": int(row["member_count"]),
                    "major_count": int(row["major_count"]),
                    "score": int(row["major_count"]) * 3 + int(row["member_count"]),
                    "promoted_at": row["promoted_at"],
                    "rep_title": str(representative["title"]),
                    "rep_source": str(representative["source"]),
                    "titles": [str(member["title"]) for member in members],
                    # Majors first, newest first: the brief links at most two
                    # and should link the outlets a reader recognises. Two
                    # stable sorts, because the key mixes a number to reverse
                    # with a timestamp string that cannot be negated.
                    "sources": [
                        (str(member["source"]), str(member["url"]))
                        for member in sorted(
                            sorted(
                                members,
                                key=lambda member: str(member["published_at"]),
                                reverse=True,
                            ),
                            key=lambda member: -int(member["is_major"]),
                        )
                    ],
                    # Which feeds filed this story. Feeds that only ever cover
                    # one domain (a quake wire, a sport desk) classify it more
                    # reliably than its vocabulary does.
                    "source_ids": sorted(
                        {
                            str(_member_source_id(member))
                            for member in members
                            if _member_source_id(member)
                        }
                    ),
                }
            )
        return result

    def prune(self, now: datetime) -> None:
        """Drop clusters long past the rolling window, members included."""
        connection = self.store.connection
        cutoff = _iso(now - timedelta(hours=self.window_hours * 4))
        old = [
            int(row["id"])
            for row in connection.execute(
                "SELECT id FROM story_cluster WHERE last_at < ?", (cutoff,)
            )
        ]
        if not old:
            return
        marks = ",".join("?" * len(old))
        connection.execute(
            f"DELETE FROM cluster_member WHERE cluster_id IN ({marks})", old
        )
        connection.execute(f"DELETE FROM story_cluster WHERE id IN ({marks})", old)
        connection.commit()
