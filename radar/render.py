from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime
from typing import Any

from . import __version__
from .models import AlertEvent, RenderedMessage
from .summarizer import Summary
from .util import SGT, json_dumps, strip_html, visible_length

ALLOWED_TAG_PATTERN = re.compile(
    r"</?(?:b|i|blockquote)>|<a href=\"https://[^\"]+\">|</a>"
)
REGION_LABELS = {
    "us": "🇺🇸",
    "china": "🇨🇳",
    "eu": "🇪🇺",
    "europe": "🇪🇺",
    "asia": "🇯🇵",
    "crypto": "🪙",
    "global": "🌐",
}
TEXT_SCORES = {"text": 98, "image": 18, "video": 0}
TEXT_FEATURE_GATE = {
    "rich_messages": False,
    "images": False,
    "videos": False,
    "max_rich_text_chars": 32768,
    "max_html_chars": 400,
}


def validate_html(value: str, max_visible_chars: int) -> None:
    remainder = ALLOWED_TAG_PATTERN.sub("", value)
    if "<" in remainder or ">" in remainder:
        raise ValueError("unsupported or unsafe Telegram HTML")
    for tag in ("b", "i", "blockquote", "a"):
        openings = len(re.findall(rf"<{tag}(?:\s[^>]*)?>", value))
        closings = value.count(f"</{tag}>")
        if openings != closings:
            raise ValueError(f"unbalanced Telegram HTML tag: {tag}")
    if visible_length(value) > max_visible_chars:
        raise ValueError(
            f"Telegram message too long: {visible_length(value)}>{max_visible_chars}"
        )


def _clip(value: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= limit:
        return normalized
    if limit <= 1:
        return "…"[:limit]
    return normalized[: limit - 1].rstrip() + "…"


def _source_link(event: AlertEvent, max_name_chars: int = 22) -> str:
    item = event.primary
    source = html.escape(_clip(item.source, max_name_chars))
    if item.url.startswith("https://"):
        return f'<a href="{html.escape(item.url, quote=True)}">{source}</a>'
    return source


def _source_links(event: AlertEvent, max_sources: int = 2) -> str:
    ordered = [event.primary] + [
        item
        for item in sorted(
            event.items,
            key=lambda item: item.published_at,
            reverse=True,
        )
        if item is not event.primary
    ]
    links: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        key = item.source.casefold()
        if key in seen:
            continue
        seen.add(key)
        source = html.escape(_clip(item.source, 22))
        if item.url.startswith("https://"):
            source = f'<a href="{html.escape(item.url, quote=True)}">{source}</a>'
        links.append(source)
        if len(links) >= max_sources:
            break
    return " · ".join(links)


def _clock(value: datetime) -> str:
    return value.astimezone(SGT).strftime("%H:%M")


def _visual_spec(
    *,
    primary_question: str,
    headline: str,
    answer: str,
    intent: str,
    grammar: str,
    roles: list[str],
    evidence: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "primary_question": primary_question,
        "headline": headline,
        "answer": answer,
        "semantic_roles": roles,
        "intents": [intent],
        "evidence": evidence,
        "scores": TEXT_SCORES,
        "selected_modality": "text",
        "delivery_format": "html",
        "fallback_chain": ["text"],
        "selection_reason": (
            "一条结论和最多五条事实无需共享坐标或空间关系，"
            "Telegram 原生文字最快且可搜索。"
        ),
        "grammar": grammar,
        "feature_gate": TEXT_FEATURE_GATE,
        "warnings": [
            "Rich Messages 未启用；使用 Telegram HTML，并保留 plain text fallback。"
        ],
    }


def _event_evidence(
    event: AlertEvent,
    summary: Summary,
    now: datetime,
) -> list[dict[str, str]]:
    primary_index = event.items.index(event.primary)
    evidence = [
        {
            "label": "事件标题",
            "value": summary.title,
            "role": "status",
            "source_path": "$.summary.title",
        },
        {
            "label": "核心事实",
            "value": summary.fact,
            "role": "status",
            "source_path": "$.summary.fact",
        },
        {
            "label": "即时影响",
            "value": summary.impact,
            "role": "status",
            "source_path": "$.summary.impact",
        },
        {
            "label": "发生时间",
            "value": event.primary.published_at.isoformat(),
            "role": "time",
            "source_path": f"$.items[{primary_index}].published_at",
        },
        {
            "label": "扫描时间",
            "value": now.isoformat(),
            "role": "time",
            "source_path": "$.rendered_at",
        },
        {
            "label": "事件分类",
            "value": event.assessment.category,
            "role": "category",
            "source_path": "$.assessment.category",
        },
        {
            "label": "事件进展日",
            "value": str(event.assessment.lineage_day),
            "role": "scalar",
            "source_path": "$.assessment.lineage_day",
        },
    ]
    for index, item in enumerate(event.items):
        evidence.extend(
            [
                {
                    "label": f"来源 {index + 1}",
                    "value": item.source,
                    "role": "source",
                    "source_path": f"$.items[{index}].source",
                },
                {
                    "label": f"来源 URL {index + 1}",
                    "value": item.url,
                    "role": "source",
                    "source_path": f"$.items[{index}].url",
                },
            ]
        )
    return evidence


def render_p0(
    event: AlertEvent,
    summary: Summary,
    now: datetime,
    max_visible_chars: int,
) -> RenderedMessage:
    local_time = now.astimezone(SGT).strftime("%H:%M")
    event_time = _clock(event.primary.published_at)
    category_tag = event.assessment.category.replace("_", "")
    day_prefix = (
        f"【Day {event.assessment.lineage_day}】"
        if event.assessment.lineage_day > 1
        else ""
    )
    display_title = _clip(f"{day_prefix}{summary.title}", 28)
    fact = _clip(summary.fact, 155)
    impact = _clip(summary.impact, 90)

    def build(current_fact: str, current_impact: str) -> str:
        return (
            f"🚨 <b>{html.escape(display_title)}</b>\n"
            f"<blockquote>{html.escape(current_fact)}\n"
            f"<b>即时影响</b>：{html.escape(current_impact)}</blockquote>\n"
            f"<i>发生 {event_time} · 雷达 {local_time} SGT</i>\n"
            f"来源 {_source_links(event)} · #{html.escape(category_tag)}"
        )

    body = build(fact, impact)
    overflow = visible_length(body) - max_visible_chars
    if overflow > 0:
        fact = _clip(fact, max(48, len(fact) - overflow - 1))
        body = build(fact, impact)
    overflow = visible_length(body) - max_visible_chars
    if overflow > 0:
        impact = _clip(impact, max(32, len(impact) - overflow - 1))
        body = build(fact, impact)
    validate_html(body, max_visible_chars)
    plain = strip_html(body)
    digest = hashlib.md5(body.encode("utf-8")).hexdigest()  # nosec B324
    evidence = [
        {
            "title": item.title,
            "source": item.source,
            "url": item.url,
            "published_at": item.published_at.isoformat(),
            "source_tier": item.source_tier,
            "event_key": event.assessment.event_key,
            "material_hash": event.assessment.material_hash,
            "assessment_level": event.assessment.level,
            "topic_anchor": event.assessment.topic_anchor,
            "lineage_day": event.assessment.lineage_day,
        }
        for item in event.items
    ]
    visual_spec = _visual_spec(
        primary_question="刚刚发生了什么，为什么现在重要？",
        headline=f"{day_prefix}{summary.title}",
        answer=summary.fact,
        intent="state",
        grammar="verdict-key-values",
        roles=["scalar", "status", "category", "time", "source"],
        evidence=_event_evidence(event, summary, now),
    )
    return RenderedMessage(
        level="P0",
        html=body,
        plain_text=plain,
        event_keys=[event.assessment.event_key],
        content_hash=digest,
        evidence=evidence,
        created_at=now,
        visual_spec=visual_spec,
    )


def render_p1(
    entries: list[tuple[AlertEvent, Summary]],
    now: datetime,
    max_visible_chars: int,
) -> RenderedMessage:
    local_time = now.astimezone(SGT).strftime("%H:%M")
    selected = entries[:5]

    def build(current: list[tuple[AlertEvent, Summary]], fact_cap: int) -> str:
        blocks: list[str] = []
        for event, summary in current:
            icon = REGION_LABELS.get(event.assessment.region, "🌐")
            day_prefix = (
                f"【Day {event.assessment.lineage_day}】"
                if event.assessment.lineage_day > 1
                else ""
            )
            fact = _clip(f"{day_prefix}{summary.fact}", fact_cap)
            blocks.append(
                f"<blockquote>{icon} <b>{html.escape(fact)}</b>\n"
                f"发生 {_clock(event.primary.published_at)} · "
                f"{_source_link(event, 18)}</blockquote>"
            )
        return (
            f"📡 <b>{len(current)} 条需关注的新动态</b>\n"
            + "\n".join(blocks)
            + f"\n<i>雷达 {local_time} SGT</i>"
        )

    while True:
        empty_body = build(selected, 0)
        available = max_visible_chars - visible_length(empty_body)
        fact_cap = max(20, available // len(selected))
        body = build(selected, fact_cap)
        if visible_length(body) <= max_visible_chars:
            break
        if len(selected) > 2:
            selected = selected[:-1]
            continue
        fact_cap = max(8, fact_cap - (visible_length(body) - max_visible_chars))
        body = build(selected, fact_cap)
        break
    validate_html(body, max_visible_chars)
    evidence: list[dict[str, Any]] = []
    visual_evidence: list[dict[str, str]] = [
        {
            "label": "动态数量",
            "value": str(len(selected)),
            "role": "scalar",
            "source_path": "$.entry_count",
        },
        {
            "label": "扫描时间",
            "value": now.isoformat(),
            "role": "time",
            "source_path": "$.rendered_at",
        },
    ]
    for index, (event, summary) in enumerate(selected):
        primary = event.primary
        evidence.append(
            {
                "title": primary.title,
                "source": primary.source,
                "url": primary.url,
                "published_at": primary.published_at.isoformat(),
                "source_tier": primary.source_tier,
                "event_key": event.assessment.event_key,
                "material_hash": event.assessment.material_hash,
                "assessment_level": event.assessment.level,
                "topic_anchor": event.assessment.topic_anchor,
                "lineage_day": event.assessment.lineage_day,
            }
        )
        visual_evidence.extend(
            [
                {
                    "label": f"动态 {index + 1}",
                    "value": summary.fact,
                    "role": "status",
                    "source_path": f"$.entries[{index}].summary.fact",
                },
                {
                    "label": f"发生时间 {index + 1}",
                    "value": primary.published_at.isoformat(),
                    "role": "time",
                    "source_path": f"$.entries[{index}].published_at",
                },
                {
                    "label": f"来源 {index + 1}",
                    "value": primary.source,
                    "role": "source",
                    "source_path": f"$.entries[{index}].source",
                },
                {
                    "label": f"来源 URL {index + 1}",
                    "value": primary.url,
                    "role": "source",
                    "source_path": f"$.entries[{index}].url",
                },
            ]
        )
    digest = hashlib.md5(body.encode("utf-8")).hexdigest()  # nosec B324
    visual_spec = _visual_spec(
        primary_question="过去两小时有哪些刚发生、值得现在知道的新动态？",
        headline=f"{len(selected)} 条需关注的新动态",
        answer="；".join(summary.fact for _, summary in selected),
        intent="digest",
        grammar="html-digest",
        roles=["scalar", "status", "time", "source"],
        evidence=visual_evidence,
    )
    return RenderedMessage(
        level="P1",
        html=body,
        plain_text=strip_html(body),
        event_keys=[event.assessment.event_key for event, _ in selected],
        content_hash=digest,
        evidence=evidence,
        created_at=now,
        visual_spec=visual_spec,
    )


def _brief_links(sources: list[tuple[str, str]], max_sources: int = 2) -> str:
    """Outlet names, hyperlinked where the URL is safe to link.

    The alert path has always done this; the brief did not, so its lines were
    unverifiable dead text. Tags and hrefs cost zero visible characters under
    Telegram's counter, so attribution is free here.
    """
    links: list[str] = []
    seen: set[str] = set()
    for name, url in sources:
        key = str(name).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        label = html.escape(_clip(str(name), 22))
        if str(url).startswith("https://"):
            label = f'<a href="{html.escape(str(url), quote=True)}">{label}</a>'
        links.append(label)
        if len(links) >= max_sources:
            break
    return " · ".join(links)


def _render_brief_item(item: dict[str, Any]) -> str:
    lead = f"<b>{html.escape(str(item.get('lead') or ''))}</b>"
    note = html.escape(str(item.get("note") or "")).strip()
    links = _brief_links(list(item.get("sources") or []))
    tail = " · ".join(part for part in (note, links) if part)
    return f"{lead}\n{tail}" if tail else lead


def render_brief(
    *,
    header: str,
    subtitle: str,
    sections: list[dict[str, Any]],
    replayed: list[str],
    footer: str,
    event_key: str,
    now: datetime,
    max_visible_chars: int,
) -> RenderedMessage:
    """One daily-brief message, sectioned by domain.

    The old shape was a single blockquote of identical bullets: nothing led,
    nothing was attributable, and a reader could not tell an earthquake from
    an earnings report without reading every line. Here each domain is its own
    labelled block, each story leads with a bold line and follows with its
    context and its sources, and the header carries the day's temperature.

    Overflow degrades in the order that costs the least meaning: the replay of
    already-sent alerts goes first, then trailing stories, then whole empty
    sections, and only a lone oversized lead is ever clipped.
    """
    local_time = now.astimezone(SGT).strftime("%H:%M")
    working: list[dict[str, Any]] = [
        {
            "emoji": str(section.get("emoji") or ""),
            "label": str(section.get("label") or ""),
            "items": list(section.get("items") or []),
        }
        for section in sections
    ]
    quoted = [line for line in replayed if str(line).strip()]

    def total_items(current: list[dict[str, Any]]) -> int:
        return sum(len(section["items"]) for section in current)

    def build(current: list[dict[str, Any]], replay: list[str]) -> str:
        blocks = [f"🌍 <b>{html.escape(header)}</b>"]
        if subtitle:
            blocks.append(f"<i>{html.escape(subtitle)}</i>")
        for section in current:
            if not section["items"]:
                continue
            title = (
                f"{html.escape(section['emoji'])} "
                f"<b>{html.escape(section['label'])}</b>"
            )
            body_lines = "\n".join(
                _render_brief_item(item) for item in section["items"]
            )
            blocks.append(f"{title}\n{body_lines}")
        if replay:
            bullets = "\n".join(f"• {html.escape(line)}" for line in replay)
            blocks.append(f"已推提醒\n<blockquote>{bullets}</blockquote>")
        if not current or total_items(current) == 0:
            blocks.insert(1 if subtitle else 1, "（本时段无达到整理门槛的世界大事）")
        blocks.append(f"<i>{html.escape(footer)} · 雷达 {local_time} SGT</i>")
        return "\n\n".join(blocks)

    body = build(working, quoted)
    while quoted and visible_length(body) > max_visible_chars:
        quoted.pop()
        body = build(working, quoted)
    while total_items(working) > 1 and visible_length(body) > max_visible_chars:
        for section in reversed(working):
            if section["items"]:
                section["items"].pop()
                break
        working = [section for section in working if section["items"]]
        body = build(working, quoted)
    if visible_length(body) > max_visible_chars and total_items(working) == 1:
        only = working[0]["items"][0]
        overhead = visible_length(build([{**working[0], "items": [{"lead": ""}]}], []))
        working[0]["items"] = [
            {
                "lead": _clip(
                    str(only.get("lead") or ""), max(24, max_visible_chars - overhead)
                )
            }
        ]
        body = build(working, quoted)
    validate_html(body, max_visible_chars)
    digest = hashlib.md5(body.encode("utf-8")).hexdigest()  # nosec B324
    leads = [
        str(item.get("lead") or "") for section in working for item in section["items"]
    ]
    visual_evidence: list[dict[str, str]] = [
        {
            "label": "条目数量",
            "value": str(len(leads)),
            "role": "scalar",
            "source_path": "$.line_count",
        },
        {
            "label": "分区数量",
            "value": str(len([s for s in working if s["items"]])),
            "role": "scalar",
            "source_path": "$.section_count",
        },
        {
            "label": "生成时间",
            "value": now.isoformat(),
            "role": "time",
            "source_path": "$.rendered_at",
        },
    ]
    for index, lead in enumerate(leads):
        visual_evidence.append(
            {
                "label": f"条目 {index + 1}",
                "value": lead,
                "role": "status",
                "source_path": f"$.lines[{index}]",
            }
        )
    visual_spec = _visual_spec(
        primary_question="过去半天世界发生了什么？",
        headline=header,
        answer="；".join(leads[:4]) or header,
        intent="digest",
        grammar="html-digest",
        roles=["scalar", "status", "time"],
        evidence=visual_evidence,
    )
    return RenderedMessage(
        level="BRIEF",
        html=body,
        plain_text=strip_html(body),
        event_keys=[event_key],
        content_hash=digest,
        evidence=[{"line": lead} for lead in leads],
        created_at=now,
        visual_spec=visual_spec,
    )


def render_brief_cover(
    *,
    header: str,
    event_key: str,
    now: datetime,
    max_visible_chars: int,
) -> RenderedMessage:
    """The caption that carries the brief's comic as its own message.

    The comic used to ride on the brief itself, which capped the whole brief
    at Telegram's 1024-character photo caption and silently dropped the image
    whenever the text won. Splitting them lets the text use the full 4096 and
    the picture arrive intact behind it.
    """
    local_time = now.astimezone(SGT).strftime("%H:%M")
    body = f"🖼 <b>{html.escape(header)}</b>\n<i>今日速览图 · 雷达 {local_time} SGT</i>"
    validate_html(body, max_visible_chars)
    digest = hashlib.md5(body.encode("utf-8")).hexdigest()  # nosec B324
    return RenderedMessage(
        level="BRIEF",
        html=body,
        plain_text=strip_html(body),
        event_keys=[event_key],
        content_hash=digest,
        evidence=[{"cover": header}],
        created_at=now,
        visual_spec=_visual_spec(
            primary_question="今天的世界速览图",
            headline=header,
            answer=header,
            intent="digest",
            grammar="html-cover",
            roles=["status"],
            evidence=[
                {
                    "label": "标题",
                    "value": header,
                    "role": "status",
                    "source_path": "$.header",
                }
            ],
        ),
    )


def outbox_manifest(
    message: RenderedMessage,
    delivery: dict[str, object],
    modality: str = "text",
) -> str:
    return json_dumps(
        {
            "schema_version": "1.0",
            "modality": modality,
            "level": message.level,
            "headline": message.plain_text.splitlines()[0],
            "event_keys": message.event_keys,
            "content_hash": message.content_hash,
            "visible_chars": visible_length(message.html),
            "render_time": message.created_at.astimezone(SGT).isoformat(),
            "timezone": "Asia/Singapore",
            "version": f"global-news-radar/{__version__}",
            "visual_spec": message.visual_spec,
            "evidence": message.evidence,
            "delivery": delivery,
        }
    )
