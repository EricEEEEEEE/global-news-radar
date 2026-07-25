from __future__ import annotations

import re
from datetime import datetime, time
from typing import Any

from .models import Assessment, NewsItem
from .util import SGT, stable_hash

FORECAST_TERMS = (
    "forecast",
    "preview",
    "outlook",
    "analyst",
    "price target",
    "could ",
    "may ",
    "might ",
    "expected to",
    "what to know",
    "watch for",
    "scheduled",
    "upcoming",
    "prediction",
    "预测",
    "展望",
    "分析师",
    "或将",
    "可能",
)

CENTRAL_BANK_ENTITIES = {
    "federal reserve": "FED",
    "fed ": "FED",
    "ecb": "ECB",
    "european central bank": "ECB",
    "bank of japan": "BOJ",
    "boj": "BOJ",
    "people's bank of china": "PBOC",
    "pboc": "PBOC",
    "bank of england": "BOE",
}

ENTITY_ALIASES = {
    **CENTRAL_BANK_ENTITIES,
    "securities and exchange commission": "SEC",
    "u.s. sec": "SEC",
    "sec ": "SEC",
    "commodity futures trading commission": "CFTC",
    "cftc": "CFTC",
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "china": "CHINA",
    "japan": "JAPAN",
    "russia": "RUSSIA",
    "ukraine": "UKRAINE",
    "iran": "IRAN",
    "israel": "ISRAEL",
    "north korea": "NORTH_KOREA",
    "united states": "USA",
    "u.s.": "USA",
}

CENTRAL_BANK_ACTIONS = (
    "raises interest rate",
    "raises rates",
    "hikes rates",
    "cuts interest rate",
    "cuts rates",
    "holds rates",
    "keeps rates",
    "rate decision",
    "emergency meeting",
    "quantitative easing",
    "quantitative tightening",
    "changes policy rate",
    "yield curve control",
    "加息",
    "降息",
    "维持利率",
    "利率决议",
)

OFFICIAL_DECISION_TERMS = (
    "fomc statement",
    "monetary policy decision",
    "monetary policy decisions",
    "statement on monetary policy",
    "policy board decision",
    "利率决议",
    "货币政策决定",
)

GEOPOLITICAL_ACTIONS = (
    "declares war",
    "invasion",
    "invades",
    "military strike",
    "air strike",
    "missile attack",
    "imposes sanctions",
    "new sanctions",
    "blockade",
    "ceasefire collapses",
    "战争爆发",
    "入侵",
    "空袭",
    "导弹袭击",
    "制裁",
)

REGULATORY_ACTIONS = (
    "charges ",
    "sues ",
    "files lawsuit",
    "bans ",
    "ban on",
    "approves spot",
    "rejects spot",
    "enforcement action",
    "cease and desist",
    "license revoked",
    "起诉",
    "禁令",
    "禁止",
    "批准现货",
    "监管执法",
)

CRYPTO_TERMS = (
    "bitcoin",
    "btc",
    "ethereum",
    "crypto",
    "cryptocurrency",
    "stablecoin",
    "exchange-traded fund",
    "spot etf",
    "数字资产",
    "加密",
    "稳定币",
)

P1_ACTIONS = {
    "merger_acquisition": (
        "acquires ",
        "to acquire",
        "agrees to buy",
        "merger agreement",
        "takeover bid",
        "收购",
        "并购",
        "合并协议",
    ),
    "financing": (
        "raises $",
        "funding round",
        "series a",
        "series b",
        "series c",
        "debt financing",
        "融资",
        "募资",
    ),
    "industry": (
        "factory shutdown",
        "supply disruption",
        "export restriction",
        "recall ",
        "data breach",
        "production halt",
        "供应中断",
        "出口限制",
        "数据泄露",
        "停产",
    ),
}

ACTION_LABELS = {
    "central_bank": "policy_decision",
    "geopolitics": "conflict_or_sanctions",
    "crypto_regulation": "regulatory_action",
    "macro": "data_release",
    "earnings": "earnings_release",
    "merger_acquisition": "corporate_transaction",
    "financing": "financing",
    "industry": "industry_disruption",
}


def is_quiet_window(now: datetime, start: str, end: str) -> bool:
    local = now.astimezone(SGT)
    start_time = time.fromisoformat(start)
    end_time = time.fromisoformat(end)
    return start_time <= local.time().replace(tzinfo=None) < end_time


def is_fresh(
    item: NewsItem,
    now: datetime,
    *,
    freshness_minutes: int,
    future_tolerance_minutes: int,
) -> tuple[bool, str]:
    age_seconds = (now - item.published_at).total_seconds()
    if age_seconds < -future_tolerance_minutes * 60:
        return False, "future_timestamp"
    if age_seconds > freshness_minutes * 60:
        return False, "stale"
    return True, "fresh"


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _entities(text: str, symbol: str = "") -> tuple[str, ...]:
    lowered = text.lower()
    found = {
        canonical for alias, canonical in ENTITY_ALIASES.items() if alias in lowered
    }
    if symbol:
        found.add(symbol.upper())
    if not found:
        proper = re.findall(r"\b[A-Z][A-Za-z0-9.&-]{2,}\b", text)
        found.update(token.upper() for token in proper[:3])
    if not found:
        stop = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "after",
            "market",
            "news",
            "breaking",
            "company",
            "agrees",
            "acquire",
            "acquires",
            "merger",
            "funding",
            "round",
        }
        tokens = [
            token
            for token in re.findall(r"\b[a-z][a-z0-9.-]{2,}\b", lowered)
            if token not in stop
        ]
        found.update(token.upper() for token in tokens[:3])
    return tuple(sorted(found)) or ("GLOBAL",)


def _event_key(
    *,
    category: str,
    action: str,
    entities: tuple[str, ...],
    occurred_at: datetime,
) -> str:
    occurrence_date = occurred_at.astimezone(SGT).date().isoformat()
    anchor = "|".join((category, action, ",".join(entities), occurrence_date))
    return f"{category}:{stable_hash(anchor, 20)}"


def _topic_anchor(category: str, action: str, entities: tuple[str, ...]) -> str:
    anchor = "|".join((category, action, ",".join(entities)))
    return f"{category}:{stable_hash(anchor, 20)}"


def _material_hash(item: NewsItem, category: str, entities: tuple[str, ...]) -> str:
    numbers = sorted(
        token.replace(",", "")
        for token in re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?%?", item.title)
    )
    values = [
        category,
        ",".join(entities),
        ",".join(numbers),
        "" if item.actual is None else f"{item.actual:g}",
        "" if item.estimate is None else f"{item.estimate:g}",
    ]
    return stable_hash("|".join(values), 20)


def _structured_assessment(
    item: NewsItem, config: dict[str, Any]
) -> tuple[str, str, str] | None:
    if (
        item.category_hint == "macro"
        and item.actual is not None
        and item.estimate is not None
    ):
        title = item.title.lower()
        if "payroll" in title:
            denominator = abs(item.estimate) or 1.0
            surprise = abs(item.actual - item.estimate) / denominator
            if surprise >= float(config["structured"]["count_relative_surprise"]):
                return "P0", "macro", f"relative surprise {surprise:.1%}"
        else:
            surprise = abs(item.actual - item.estimate)
            if surprise >= float(config["structured"]["rate_surprise_pp"]):
                return "P0", "macro", f"absolute surprise {surprise:g}pp"
        return None
    if item.category_hint == "earnings" and item.actual is not None and item.estimate:
        surprise = abs(item.actual - item.estimate) / abs(item.estimate)
        if surprise >= float(config["structured"]["earnings_surprise_ratio"]):
            return "P0", "earnings", f"EPS surprise {surprise:.1%}"
        return None
    return None


def _headline_surprise(
    item: NewsItem, config: dict[str, Any], text: str
) -> tuple[str, str, str] | None:
    rate_event = any(
        term in text
        for term in ("consumer price index", " cpi ", "inflation rate", " gdp ")
    )
    count_event = any(
        term in text for term in ("nonfarm payroll", "non-farm payroll", "jobs report")
    )
    actual_expected = re.search(
        r"([+-]?\d[\d,]*(?:\.\d+)?)\s*(%|k|m)?"
        r".{0,35}?(?:vs\.?|versus|expected|forecast|consensus)"
        r".{0,20}?([+-]?\d[\d,]*(?:\.\d+)?)\s*(%|k|m)?",
        text,
    )
    if (rate_event or count_event) and actual_expected:
        actual = float(actual_expected.group(1).replace(",", ""))
        expected = float(actual_expected.group(3).replace(",", ""))
        if rate_event and abs(actual - expected) >= float(
            config["structured"]["rate_surprise_pp"]
        ):
            item.actual, item.estimate = actual, expected
            return "P0", "macro", f"headline surprise {abs(actual - expected):g}pp"
        if count_event and abs(actual - expected) / (abs(expected) or 1.0) >= float(
            config["structured"]["count_relative_surprise"]
        ):
            item.actual, item.estimate = actual, expected
            return "P0", "macro", "headline count surprise"

    mega_caps = {symbol.lower() for symbol in config["structured"]["mega_cap_symbols"]}
    entity_values = {value.lower() for value in _entities(text, item.symbol)}
    if mega_caps.intersection(entity_values) and (
        "eps" in text or "earnings per share" in text
    ):
        match = re.search(
            r"(?:eps|earnings per share).{0,20}?([+-]?\$?\d+(?:\.\d+)?)"
            r".{0,35}?(?:vs\.?|versus|expected|estimate)"
            r".{0,20}?([+-]?\$?\d+(?:\.\d+)?)",
            text,
        )
        if match:
            actual = float(match.group(1).replace("$", ""))
            estimate = float(match.group(2).replace("$", ""))
            surprise = abs(actual - estimate) / (abs(estimate) or 1.0)
            if surprise >= float(config["structured"]["earnings_surprise_ratio"]):
                item.actual, item.estimate = actual, estimate
                return "P0", "earnings", f"headline EPS surprise {surprise:.1%}"
    return None


def assess(item: NewsItem, config: dict[str, Any]) -> Assessment | None:
    raw_text = f" {item.title} {item.summary} "
    text = raw_text.lower()
    if item.source_tier not in {"primary", "structured"} and _contains_any(
        text, FORECAST_TERMS
    ):
        return None

    structured = _structured_assessment(item, config) or _headline_surprise(
        item, config, text
    )
    level = ""
    category = ""
    reason = ""
    if structured:
        level, category, reason = structured
    elif (
        any(alias in text for alias in CENTRAL_BANK_ENTITIES)
        and _contains_any(text, CENTRAL_BANK_ACTIONS)
    ) or (
        item.source_tier == "primary"
        and item.category_hint == "central_bank"
        and _contains_any(text, OFFICIAL_DECISION_TERMS)
    ):
        level, category, reason = "P0", "central_bank", "confirmed policy action"
    elif _contains_any(text, GEOPOLITICAL_ACTIONS):
        level, category, reason = "P0", "geopolitics", "major conflict or sanctions"
    elif _contains_any(text, REGULATORY_ACTIONS) and _contains_any(text, CRYPTO_TERMS):
        level, category, reason = "P0", "crypto_regulation", "major crypto regulation"
    else:
        for candidate, terms in P1_ACTIONS.items():
            if _contains_any(text, terms):
                level, category, reason = "P1", candidate, f"matched {candidate}"
                break
    if not level:
        return None

    entities = _entities(raw_text, item.symbol)
    action = ACTION_LABELS[category]
    event_key = _event_key(
        category=category,
        action=action,
        entities=entities,
        occurred_at=item.published_at,
    )
    return Assessment(
        level=level,
        category=category,
        event_key=event_key,
        material_hash=_material_hash(item, category, entities),
        reason=reason,
        requires_corroboration=level == "P0"
        and item.source_tier not in {"primary", "structured"},
        region=item.region,
        action=action,
        entities=entities,
        topic_anchor=_topic_anchor(category, action, entities),
    )
