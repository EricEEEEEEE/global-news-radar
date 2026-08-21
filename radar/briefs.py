"""Twice-daily world briefs: the guaranteed product of the radar day.

Alerts fire when something crosses a threshold; the brief fires because the
clock does. Morning covers the overnight window, evening covers the day.
Content is what the radar actually holds — delivered alerts replayed from
the ledger, plus the top story clusters ranked by cross-outlet consensus —
so the brief can never invent a world the collectors did not see.

Ranking by consensus alone is what made the brief a monoculture: one large
disaster is filed by every outlet, so it manufactured the very signal the
ranking rewarded and owned every slot. Stories are therefore classified by
domain and packed under per-domain quotas, and the leftover slots go back to
raw rank so a quiet day still fills.

Every LLM-written line passes the same two gates as alert summaries: no
numbers beyond what the member headlines can echo, no bare English without a
Chinese gloss. A line that fails degrades to a mechanical source-count line;
the LLM being down degrades the whole brief to mechanical lines, never to
silence. Each rejection is logged, because the previous version degraded in
complete silence and looked identical to an LLM outage.
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import UTC, datetime, time, timedelta

import requests

from .clusters import ClusterEngine, tokenize
from .domains import (
    DOMAIN_EMOJI,
    DOMAIN_LABELS,
    DomainClassifier,
    group_sections,
    pack_by_quota,
    resolve_domain,
)
from .models import RenderedMessage
from .render import render_brief
from .store import RadarStore
from .summarizer import bare_english_spans
from .util import SGT, echoable_numbers, json_dumps, numeric_values, redact_secrets

LOGGER = logging.getLogger(__name__)

# Gates are deliberately looser than what the prompt asks for: the prompt is
# where brevity is requested, the gate is where faithfulness is enforced, and
# rejecting an accurate line purely for length buys nothing but an English
# fallback in its place.
_LEAD_LIMIT = 48
_WHY_LIMIT = 48

# Chaos Index weights. Deterministic on purpose -- an LLM asked "how chaotic
# was today" will happily invent a number, and a number nobody can recompute
# is worse than no number.
_CHAOS_ALERT_SATURATION = 8.0
_CHAOS_MAJOR_SATURATION = 6.0
_CHAOS_HISTORY = 30
_CHAOS_BAND = 12


def _parse_hhmm(value: str) -> time:
    hour, minute = str(value).split(":", 1)
    return time(int(hour), int(minute))


def _shorten(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _lead_titles(story: dict[str, object]) -> list[str]:
    """The member headlines the writer is shown, representative first."""
    ordered = story.get("lead_titles") or story.get("titles") or []
    return [str(title) for title in ordered]  # type: ignore[union-attr]


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


class BriefComposer:
    """Assembles one brief per slot; the service schedules and delivers it."""

    def __init__(
        self,
        *,
        store: RadarStore,
        engine: ClusterEngine,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
        max_items: int,
        max_visible_chars: int,
        morning_sgt: str,
        evening_sgt: str,
        classifier: DomainClassifier | None = None,
        quotas: dict[str, int] | None = None,
        source_hints: dict[str, str] | None = None,
        candidate_multiplier: int = 5,
    ):
        self.store = store
        self.engine = engine
        self.llm_enabled = bool(base_url and api_key and model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_items = max_items
        self.max_visible_chars = max_visible_chars
        self.morning = _parse_hhmm(morning_sgt)
        self.evening = _parse_hhmm(evening_sgt)
        self.classifier = classifier or DomainClassifier(
            enabled=False, base_url="", api_key="", model="", timeout=timeout
        )
        self.quotas = dict(quotas or {})
        self.source_hints = dict(source_hints or {})
        self.candidate_multiplier = max(1, int(candidate_multiplier))

    def due(self, now: datetime) -> str | None:
        """Which brief is owed right now, if any.

        A missed slot self-heals for a few hours (daemon restarts, delivery
        stalls), then expires: a morning brief sent in the evening is worse
        than none. One meta date per slot makes each brief at-most-once.
        """
        local = now.astimezone(SGT)
        today = local.date().isoformat()
        if (
            self.morning <= local.time() < time(12, 0)
            and self.store.get_meta("brief_morning_date") != today
        ):
            return "morning"
        if (
            local.time() >= self.evening
            and self.store.get_meta("brief_evening_date") != today
        ):
            return "evening"
        return None

    def mark_sent(self, kind: str, now: datetime) -> None:
        today = now.astimezone(SGT).date().isoformat()
        self.store.set_meta(f"brief_{kind}_date", today, now)

    def _window(self, kind: str, now: datetime) -> tuple[datetime, datetime]:
        local = now.astimezone(SGT)
        if kind == "morning":
            end = local.replace(
                hour=self.morning.hour,
                minute=self.morning.minute,
                second=0,
                microsecond=0,
            )
            start = (end - timedelta(days=1)).replace(
                hour=self.evening.hour, minute=self.evening.minute
            )
        else:
            end = local.replace(
                hour=self.evening.hour,
                minute=self.evening.minute,
                second=0,
                microsecond=0,
            )
            start = end.replace(hour=self.morning.hour, minute=self.morning.minute)
        return start, end

    def compose(
        self, kind: str, now: datetime
    ) -> tuple[RenderedMessage, list[tuple[str, str]]]:
        """Build the brief message plus the comic entries for its top stories."""
        start, end = self._window(kind, now)
        sent = self.store.deliveries_between(start, end)
        candidates = self.engine.top_clusters(
            start, end, limit=self.max_items * self.candidate_multiplier
        )
        start_iso = _utc_iso(start)
        pool = [
            story
            for story in candidates
            # Promoted inside the window = already went out as a burst alert,
            # so it lives in the 已推 section; listing it twice is noise.
            if not (story["promoted_at"] and str(story["promoted_at"]) >= start_iso)
        ]
        # Keywords first, with single-domain feeds as the tie-breaker, and the
        # LLM -- when enabled -- allowed to overrule either.
        baseline = [
            resolve_domain(
                str(story["rep_title"]),
                [str(value) for value in (story.get("source_ids") or [])],
                self.source_hints,
            )
            for story in pool
        ]
        domains = self.classifier.classify(
            [str(story["rep_title"]) for story in pool], baseline
        )
        for story, domain in zip(pool, domains, strict=True):
            story["domain"] = domain
        stories = pack_by_quota(pool, self.quotas, self.max_items)
        LOGGER.info(
            "brief_selection kind=%s pool=%s selected=%s domains=%s",
            kind,
            len(pool),
            len(stories),
            ",".join(sorted({str(story["domain"]) for story in stories})),
        )

        continuity = self._continuity_tags(now, stories)
        written = self._translate(stories, self._previous_leads())

        leads: list[str] = []
        for index, story in enumerate(stories):
            entry = written[index] if index < len(written) else {}
            lead = self._gated(
                str(entry.get("lead") or ""), story, _LEAD_LIMIT, "lead"
            ) or self._mechanical_lead(story)
            why = self._gated(str(entry.get("why") or ""), story, _WHY_LIMIT, "why")
            meta_parts: list[str] = []
            majors = int(story["major_count"])  # type: ignore[arg-type]
            if majors >= 2:
                meta_parts.append(f"{majors}家大社")
            tag = continuity.get(int(story["id"]))  # type: ignore[arg-type]
            if tag:
                meta_parts.append(tag)
            note = " · ".join(part for part in [why, *meta_parts] if part)
            story["lead"] = lead
            story["note"] = note
            leads.append(lead)

        sections = [
            {
                "emoji": DOMAIN_EMOJI.get(domain, "🗂"),
                "label": DOMAIN_LABELS.get(domain, "其他"),
                "items": [
                    {
                        "lead": str(story["lead"]),
                        "note": str(story["note"]),
                        "sources": list(story.get("sources") or []),
                    }
                    for story in bucket
                ],
            }
            for domain, bucket in group_sections(stories)
        ]
        replayed = [
            f"{_shorten(str(row['summary']).strip(), 52)}"
            for row in sent[-3:]
            if str(row["summary"]).strip()
        ]

        span_hours = int((end - start).total_seconds() // 3600)
        chaos, verdict = self._chaos_index(now, sent, stories)
        subtitle = f"混乱指数 {chaos}/100 · {verdict} · 覆盖{span_hours}小时"
        footer = f"告警{len(sent)}条 · 故事簇{len(stories)}个 · 候选{len(pool)}个"
        local_date = now.astimezone(SGT).date().isoformat()
        message = render_brief(
            header=self.header(kind, now),
            subtitle=subtitle,
            sections=sections,
            replayed=replayed,
            footer=footer,
            event_key=f"brief:{kind}:{local_date}",
            now=now,
            max_visible_chars=self.max_visible_chars,
        )
        self._remember_leads(now, leads)
        comic_entries = [("brief", lead) for lead in leads[:4]]
        return message, comic_entries

    def header(self, kind: str, now: datetime) -> str:
        local = now.astimezone(SGT)
        weekday = "一二三四五六日"[local.weekday()]
        label = "世界晨报" if kind == "morning" else "世界晚报"
        return f"{label} · {local.month}月{local.day}日 周{weekday}"

    def _mechanical_lead(self, story: dict[str, object]) -> str:
        """Fallback lead when no usable translation came back.

        Just the headline: the note line under it already prints the major
        count and the source names, so a "2家大社在报:" prefix here would say
        the same thing twice in three lines.
        """
        return _shorten(str(story["rep_title"]), 70)

    def _gated(
        self, candidate: str, story: dict[str, object], limit: int, field: str
    ) -> str:
        """Return the line if it survives both gates, else "" — and say why.

        The number gate compares against every member headline of the story: a
        faithful line can only carry numbers some outlet actually printed. The
        English gate is relaxed to admit name-shaped spans the headlines
        themselves printed, because "Klarna" with no Chinese gloss is how a
        Chinese newsroom writes it, while "posts Q2 profit" still fails.
        """
        titles: list[str] = [str(title) for title in story["titles"]]  # type: ignore[union-attr]
        cleaned = " ".join(str(candidate).split())
        if not cleaned:
            return ""
        reason = ""
        if len(cleaned) > limit:
            reason = "too_long"
        elif not (numeric_values(cleaned) <= echoable_numbers("\n".join(titles))):
            reason = "unechoed_number"
        else:
            stray = bare_english_spans(cleaned, titles)
            if stray:
                reason = f"bare_english:{stray[0][:24]}"
        if not reason:
            return cleaned
        LOGGER.info(
            "brief_line_rejected field=%s cluster=%s reason=%s",
            field,
            story.get("id"),
            reason,
        )
        return ""

    def _chaos_index(
        self,
        now: datetime,
        sent: list,
        stories: list[dict[str, object]],
    ) -> tuple[int, str]:
        """A recomputable 0-100 temperature for the window, plus its band.

        Three observable ratios, no judgement: how many alerts fired, how much
        of the brief is conflict or catastrophe, and how hard the single
        biggest story was corroborated. The band compares today against the
        median of the last 30 briefs, so "偏高" means high *for this radar*
        rather than high on an absolute scale nobody calibrated.
        """
        volume = min(1.0, len(sent) / _CHAOS_ALERT_SATURATION)
        if stories:
            hot = sum(
                1
                for story in stories
                if str(story.get("domain")) in {"disaster", "politics"}
            )
            concentration = hot / len(stories)
            intensity = min(
                1.0,
                max(int(story["major_count"]) for story in stories)  # type: ignore[arg-type]
                / _CHAOS_MAJOR_SATURATION,
            )
        else:
            concentration = 0.0
            intensity = 0.0
        index = round(100 * (0.4 * volume + 0.3 * concentration + 0.3 * intensity))
        raw = self.store.get_meta("brief_chaos_history")
        try:
            history = [int(value) for value in json.loads(raw)] if raw else []
        except (json.JSONDecodeError, TypeError, ValueError):
            history = []
        if len(history) < 5:
            verdict = "基线建立中"
        else:
            median = statistics.median(history)
            if index >= median + _CHAOS_BAND:
                verdict = "偏高"
            elif index <= median - _CHAOS_BAND:
                verdict = "偏低"
            else:
                verdict = "正常"
        self.store.set_meta(
            "brief_chaos_history",
            json_dumps((history + [index])[-_CHAOS_HISTORY:]),
            now,
        )
        return index, verdict

    def _previous_leads(self) -> list[str]:
        raw = self.store.get_meta("brief_previous_leads")
        try:
            return [str(value) for value in json.loads(raw)] if raw else []
        except (json.JSONDecodeError, TypeError):
            return []

    def _remember_leads(self, now: datetime, leads: list[str]) -> None:
        self.store.set_meta("brief_previous_leads", json_dumps(leads[:8]), now)

    def _continuity_tags(
        self, now: datetime, stories: list[dict[str, object]]
    ) -> dict[int, str]:
        """Day-N tags: a story overlapping a previous brief's story continues it.

        State is one meta row of {tokens, days, date} entries — the previous
        brief's stories. Days only advance across calendar days, so the
        evening brief repeats the morning's Day-N instead of inflating it.
        """
        local_date = now.astimezone(SGT).date().isoformat()
        raw = self.store.get_meta("brief_continuity")
        try:
            previous = json.loads(raw) if raw else []
        except json.JSONDecodeError:
            previous = []
        tags: dict[int, str] = {}
        fresh: list[dict[str, object]] = []
        for story in stories:
            tokens = tokenize(str(story["rep_title"]))
            days = 1
            for entry in previous:
                overlap = tokens.intersection(entry.get("tokens", []))
                if len(overlap) >= 3:
                    days = int(entry.get("days", 1))
                    if str(entry.get("date")) != local_date:
                        days += 1
                    break
            if days > 1:
                tags[int(story["id"])] = f"连续第{days}天"  # type: ignore[arg-type]
            fresh.append({"tokens": sorted(tokens), "days": days, "date": local_date})
        self.store.set_meta("brief_continuity", json_dumps(fresh), now)
        return tags

    def _translate(
        self, stories: list[dict[str, object]], previous_leads: list[str]
    ) -> list[dict[str, str]]:
        """One batched LLM call: every story to a Chinese lead plus a why-line.

        The model sees the member headlines, not just the representative one,
        so the why-line has material to be drawn from rather than invented —
        and both fields still go through the gates afterwards. Yesterday's
        leads ride along so a running story is phrased as a development
        instead of restarting from zero every twelve hours.
        """
        if not self.llm_enabled or not stories:
            return []
        payload = [
            {
                "index": index,
                "titles": _lead_titles(story)[:5],
            }
            for index, story in enumerate(stories)
        ]
        prompt = {
            "task": (
                "把每组同一事件的英文新闻标题改写成一条简体中文要点，"
                "供世界新闻日报使用。忠实原意，不加评论，不加猜测。"
            ),
            "requirements": {
                "count": f"lines 数组必须正好 {len(stories)} 条，顺序对应 index",
                "lead": "lead 为事件本身，不超过28个字，不写编辑评价",
                "why": (
                    "why 为这条为什么值得关注，不超过30个字，"
                    "只能依据所给标题中已出现的事实，无法从标题得出时留空字符串"
                ),
                "style": "数字与原文一致；英文人名公司名首次出现须紧跟（中文）注释",
                "output": '严格 JSON: {"lines": [{"lead": "...", "why": "..."}, ...]}',
            },
            "previous_brief_leads": previous_leads[:8],
            "stories": payload,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "temperature": 0.2,
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是新闻编辑。只输出 JSON。",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(prompt, ensure_ascii=False),
                        },
                    ],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            lines = json.loads(content)["lines"]
            return [
                {
                    "lead": str(entry.get("lead") or ""),
                    "why": str(entry.get("why") or ""),
                }
                if isinstance(entry, dict)
                else {"lead": str(entry), "why": ""}
                for entry in lines
            ]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("brief_translate_failed error=%s", redact_secrets(exc))
            return []
