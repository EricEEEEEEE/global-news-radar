"""Twice-daily world briefs: the guaranteed product of the radar day.

Alerts fire when something crosses a threshold; the brief fires because the
clock does. Morning covers the overnight window, evening covers the day.
Content is what the radar actually holds — delivered alerts replayed from
the ledger, plus the top story clusters ranked by cross-outlet consensus —
so the brief can never invent a world the collectors did not see.

Every LLM-written line passes the same two gates as alert summaries: no
numbers beyond what the member headlines can echo, no bare English without a
Chinese gloss. A line that fails degrades to a mechanical source-count line;
the LLM being down degrades the whole brief to mechanical lines, never to
silence.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, time, timedelta

import requests

from .clusters import ClusterEngine, tokenize
from .models import RenderedMessage
from .render import render_brief
from .store import RadarStore
from .summarizer import bare_english_spans
from .util import SGT, echoable_numbers, json_dumps, numeric_values, redact_secrets

LOGGER = logging.getLogger(__name__)


def _parse_hhmm(value: str) -> time:
    hour, minute = str(value).split(":", 1)
    return time(int(hour), int(minute))


def _shorten(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


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
        clusters = self.engine.top_clusters(start, end, limit=self.max_items)
        start_iso = _utc_iso(start)
        stories = [
            story
            for story in clusters
            # Promoted inside the window = already went out as a burst alert,
            # so it lives in the 已推 section; listing it twice is noise.
            if not (story["promoted_at"] and str(story["promoted_at"]) >= start_iso)
        ]
        continuity = self._continuity_tags(now, stories)
        translated = self._translate([str(story["rep_title"]) for story in stories])

        lines: list[str] = []
        for row in sent[-3:]:
            summary = str(row["summary"]).strip()
            if summary:
                lines.append(f"已推 | {_shorten(summary, 58)}")
        story_lines: list[str] = []
        for index, story in enumerate(stories):
            candidate = translated[index] if index < len(translated) else ""
            line = self._gated_line(candidate, story)
            suffix_parts: list[str] = []
            majors = int(story["major_count"])  # type: ignore[arg-type]
            if majors >= 2:
                suffix_parts.append(f"{majors}家大社")
            tag = continuity.get(int(story["id"]))  # type: ignore[arg-type]
            if tag:
                suffix_parts.append(tag)
            if suffix_parts:
                line = f"{line}（{'·'.join(suffix_parts)}）"
            story_lines.append(line)
        lines.extend(story_lines)
        if not lines:
            lines.append("本时段无达到整理门槛的世界大事。")

        span_hours = int((end - start).total_seconds() // 3600)
        footer = f"窗口{span_hours}小时 · 告警{len(sent)}条 · 故事簇{len(stories)}个"
        local_date = now.astimezone(SGT).date().isoformat()
        message = render_brief(
            header=self._header(kind, now),
            lines=lines,
            footer=footer,
            event_key=f"brief:{kind}:{local_date}",
            now=now,
            max_visible_chars=self.max_visible_chars,
        )
        comic_entries = [("brief", line) for line in story_lines[:4]]
        return message, comic_entries

    def _header(self, kind: str, now: datetime) -> str:
        local = now.astimezone(SGT)
        weekday = "一二三四五六日"[local.weekday()]
        label = "世界晨报" if kind == "morning" else "世界晚报"
        return f"{label} · {local.month}月{local.day}日 周{weekday}"

    def _gated_line(self, candidate: str, story: dict[str, object]) -> str:
        """One Chinese line if it survives both gates, else a mechanical one.

        The number gate compares against every member headline of the story:
        a faithful line can only carry numbers some outlet actually printed.
        """
        titles: list[str] = [str(title) for title in story["titles"]]  # type: ignore[union-attr]
        allowed = echoable_numbers("\n".join(titles))
        cleaned = " ".join(str(candidate).split())
        if (
            cleaned
            and len(cleaned) <= 60
            and numeric_values(cleaned) <= allowed
            and not bare_english_spans(cleaned)
        ):
            return cleaned
        majors = int(story["major_count"])  # type: ignore[arg-type]
        members = int(story["member_count"])  # type: ignore[arg-type]
        prefix = f"{majors}家大社在报" if majors else f"{members}条报道"
        return f"{prefix}: {_shorten(str(story['rep_title']), 70)}"

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

    def _translate(self, titles: list[str]) -> list[str]:
        """One batched LLM call: every story title to one Chinese line."""
        if not self.llm_enabled or not titles:
            return []
        prompt = {
            "task": (
                "把每条英文新闻标题改写成一句简体中文要点，供世界新闻日报使用。"
                "忠实原意，不加评论，不加猜测。"
            ),
            "requirements": {
                "count": f"lines 数组必须正好 {len(titles)} 条，顺序对应输入",
                "style": (
                    "每条不超过40个字；数字与原文一致；英文人名公司名须紧跟（中文）注释"
                ),
                "output": '严格 JSON: {"lines": ["...", ...]}',
            },
            "titles": titles,
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
                    "max_tokens": 1200,
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
            return [str(value) for value in json.loads(content)["lines"]]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("brief_translate_failed error=%s", redact_secrets(exc))
            return []
