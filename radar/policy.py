from __future__ import annotations

import re
from datetime import datetime, time
from typing import Any

from .models import Assessment, NewsItem
from .util import SGT, simhash64, stable_hash

FORECAST_TERMS = (
    "forecast",
    "preview",
    "outlook",
    "analyst",
    "price target",
    "expected to",
    "what to know",
    "watch for",
    "scheduled",
    "upcoming",
    "prediction",
)

FORECAST_TERMS_CJK = (
    "预测",
    "展望",
    "分析师",
    "或将",
    "可能",
)

_FORECAST_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in FORECAST_TERMS) + r")s?\b"
)
_FORECAST_MODAL_PATTERN = re.compile(r"\b(?:could|might)\b")
# "may" is both a modal verb and a month; a naive substring match silences
# legitimate May-dated data releases such as "US May CPI rises 4.1% vs 3.2%".
_MAY_TOKEN = re.compile(r"\bmay\b")
_MAY_AS_MONTH_LEAD = re.compile(
    r"\b(?:in|for|of|at|by|from|since|through|during|before|after|until|"
    r"late|early|mid|last|this|next)\s+$"
)
_MAY_AS_MONTH_TRAIL = re.compile(
    r"^\s*(?:\d{1,4}\b|cpi|ppi|pce|gdp|payrolls?|nonfarm|non-farm|jobs|inflation|"
    r"unemployment|retail|trade|industrial|housing|exports?|imports?|data|"
    r"figures|report|reading|print|quarter|sales|output|orders|"
    r"meeting|fomc|session|summit|policy|decision)\b"
)

MACRO_SERIES = (
    ("cpi", ("consumer price index", "consumer prices", "cpi")),
    ("ppi", ("producer price index", "ppi")),
    ("pce", ("personal consumption expenditures", "pce")),
    ("gdp", ("gross domestic product", "gdp")),
    ("nfp", ("nonfarm payroll", "non-farm payroll", "jobs report")),
    ("unemployment", ("unemployment rate",)),
    ("retail_sales", ("retail sales",)),
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

COUNTRY_ALIASES = {
    "china": "CHINA",
    "chinese": "CHINA",
    "japan": "JAPAN",
    "japanese": "JAPAN",
    "russia": "RUSSIA",
    "russian": "RUSSIA",
    "ukraine": "UKRAINE",
    "iran": "IRAN",
    "iranian": "IRAN",
    "israel": "ISRAEL",
    "israeli": "ISRAEL",
    "north korea": "NORTH_KOREA",
    "south korea": "SOUTH_KOREA",
    "taiwan": "TAIWAN",
    "india": "INDIA",
    "germany": "GERMANY",
    "france": "FRANCE",
    "united kingdom": "UK",
    "britain": "UK",
    "euro zone": "EUROZONE",
    "eurozone": "EUROZONE",
    "european union": "EU",
    "venezuela": "VENEZUELA",
    "saudi arabia": "SAUDI_ARABIA",
    "turkey": "TURKEY",
    "brazil": "BRAZIL",
    "mexico": "MEXICO",
    "canada": "CANADA",
    "australia": "AUSTRALIA",
    "hong kong": "HONG_KONG",
    "singapore": "SINGAPORE",
    "united states": "USA",
    "u.s.": "USA",
    "us ": "USA",
    "american": "USA",
}

ENTITY_ALIASES = {
    **CENTRAL_BANK_ENTITIES,
    **COUNTRY_ALIASES,
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
}

CENTRAL_BANK_ACTIONS_CJK = (
    "加息",
    "降息",
    "维持利率",
    "利率决议",
)

_RATE_QUALIFIER = r"(?:the\s+)?(?:main\s+|key\s+|benchmark\s+|policy\s+|interest\s+)*"
# Newsrooms report the same decision as a verb ("Fed slashes rates") or as a
# noun ("Fed announces emergency rate cut"); a verb-only vocabulary silences
# roughly half of every real rate decision.
_CENTRAL_BANK_ACTION_PATTERN = re.compile(
    r"\b(?:"
    r"(?:interest[- ]|policy[- ]|benchmark[- ]|key[- ])?rates?[- ]"
    r"(?:cut|cuts|hike|hikes|rise|increase|increases|reduction|decision|move)s?"
    r"|(?:cut|cuts|lower|lowers|slash|slashes|reduce|reduces|trim|trims"
    r"|raise|raises|hike|hikes|lift|lifts|hold|holds|keep|keeps)"
    rf"\s+{_RATE_QUALIFIER}rates?"
    r"|quantitative (?:easing|tightening)"
    r"|yield curve control"
    r"|emergency meeting"
    r")\b"
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

# World-event vocabularies. All of these arrive from discovery feeds, so the
# existing rule "secondary-source P0 needs two major outlets" is the noise
# gate; the keywords only have to be unambiguous, not exhaustive.
DISASTER_TERMS = (
    "earthquake",
    "tsunami",
    "plane crash",
    "air crash",
    "jet crashes",
    "volcano erupt",
    "dam collapse",
    "bridge collapse",
    "train derail",
    "flash flood",
    "hurricane makes landfall",
    "typhoon makes landfall",
    "地震",
    "海啸",
    "坠机",
    "空难",
    "火山喷发",
    "溃坝",
    "脱轨",
)

POLITICAL_CRISIS_TERMS = (
    # Leading space so "recoup" cannot read as a coup.
    " coup",
    "martial law",
    "assassinat",
    "impeach",
    "state of emergency",
    "president dies",
    "president dead",
    "president resigns",
    "prime minister resigns",
    "政变",
    "戒严",
    "遇刺",
    "弹劾",
    "紧急状态",
    "总统去世",
    "首相去世",
)

HEALTH_EMERGENCY_TERMS = (
    "pandemic",
    "public health emergency",
    "who declares",
    "epidemic",
    "disease outbreak",
    "outbreak spreads",
    "大流行",
    "疫情爆发",
    "公共卫生紧急",
)

# 宏观数据只保留五大经济体：美、中、欧元区、日、英。FMP rows carry an ISO
# country code; headline items match padded aliases so "thus" is not the US.
MAJOR_ECONOMY_CODES = {"US", "CN", "EU", "EA", "JP", "GB", "UK"}
MAJOR_ECONOMY_TERMS = (
    " united states",
    " u.s.",
    " us ",
    " america",
    " china",
    " chinese",
    " euro zone",
    " eurozone",
    " euro area",
    " japan",
    " united kingdom",
    " britain",
    " british",
    " uk ",
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

ACTION_LABELS = {
    "central_bank": "policy_decision",
    "geopolitics": "conflict_or_sanctions",
    "crypto_regulation": "regulatory_action",
    "macro": "data_release",
    "disaster": "major_disaster",
    "political_crisis": "political_upheaval",
    "health_emergency": "health_emergency",
    "trending": "editor_pick",
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


def quiet_window_end(now: datetime, end: str) -> datetime:
    end_time = time.fromisoformat(end)
    return now.astimezone(SGT).replace(
        hour=end_time.hour, minute=end_time.minute, second=0, microsecond=0
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _has_forecast_modal(text: str) -> bool:
    if _FORECAST_MODAL_PATTERN.search(text):
        return True
    for match in _MAY_TOKEN.finditer(text):
        before = text[max(0, match.start() - 12) : match.start()]
        after = text[match.end() : match.end() + 16]
        if _MAY_AS_MONTH_LEAD.search(before) or _MAY_AS_MONTH_TRAIL.match(after):
            continue
        return True
    return False


def is_forecast(text: str) -> bool:
    if _contains_any(text, FORECAST_TERMS_CJK):
        return True
    if _FORECAST_PATTERN.search(text):
        return True
    return _has_forecast_modal(text)


def is_central_bank_action(text: str) -> bool:
    if _contains_any(text, CENTRAL_BANK_ACTIONS_CJK):
        return True
    return bool(_CENTRAL_BANK_ACTION_PATTERN.search(text))


def _macro_series(text: str) -> str:
    for series, terms in MACRO_SERIES:
        if _contains_any(text, terms):
            return series
    return ""


def _entities(text: str, symbol: str = "") -> tuple[str, ...]:
    """Controlled-vocabulary entities only — these are safe to key events on."""
    lowered = text.lower()
    found = {
        canonical for alias, canonical in ENTITY_ALIASES.items() if alias in lowered
    }
    if symbol:
        found.add(symbol.upper())
    return tuple(sorted(found))


def _display_entities(text: str, symbol: str = "") -> tuple[str, ...]:
    """Best-effort entities for humans; unstable across outlets, never keyed on."""
    lowered = text.lower()
    found = set(_entities(text, symbol))
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


def _key_scope(entities: tuple[str, ...], title: str) -> str:
    """The keyable identity of an event.

    Controlled entities are identical no matter which outlet wrote the headline.
    Without one, fall back to the title simhash: still identical for two outlets
    running near-identical wording, and still distinct for unrelated stories,
    which a shared placeholder scope would have merged into one topic.
    """
    if entities:
        return ",".join(entities)
    return f"~{simhash64(title)}"


def _event_key(
    *,
    category: str,
    action: str,
    scope: str,
    occurred_at: datetime,
    detail: str = "",
) -> str:
    occurrence_date = occurred_at.astimezone(SGT).date().isoformat()
    anchor = "|".join((category, action, detail, scope, occurrence_date))
    return f"{category}:{stable_hash(anchor, 20)}"


def _topic_anchor(category: str, action: str, scope: str, detail: str = "") -> str:
    anchor = "|".join((category, action, detail, scope))
    return f"{category}:{stable_hash(anchor, 20)}"


def _material_hash(item: NewsItem, category: str, scope: str) -> str:
    numbers = sorted(
        token.replace(",", "")
        for token in re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?%?", item.title)
    )
    values = [
        category,
        scope,
        ",".join(numbers),
        "" if item.actual is None else f"{item.actual:g}",
        "" if item.estimate is None else f"{item.estimate:g}",
    ]
    return stable_hash("|".join(values), 20)


def _is_major_economy(item: NewsItem, text: str) -> bool:
    # A recognised ISO code decides immediately; anything else — including a
    # provider that starts sending full country names — falls through to the
    # alias check on the text, which always carries the country string.
    if str(item.raw.get("country") or "").strip().upper() in MAJOR_ECONOMY_CODES:
        return True
    return _contains_any(text, MAJOR_ECONOMY_TERMS)


def _structured_assessment(
    item: NewsItem, config: dict[str, Any]
) -> tuple[str, str, str] | None:
    if (
        item.category_hint == "macro"
        and item.actual is not None
        and item.estimate is not None
    ):
        # A 0.2pp CPI miss in Pakistan is a fact, not news the operator can
        # use. Only the economies that price the world get through.
        if not _is_major_economy(item, f" {item.title.lower()} "):
            return None
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
        if not _is_major_economy(item, text):
            return None
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
    return None


def trending_assessment(item: NewsItem) -> Assessment:
    """Assessment for an editor-picked story. P1: batched, never urgent."""
    scope = _key_scope((), item.title)
    action = ACTION_LABELS["trending"]
    return Assessment(
        level="P1",
        category="trending",
        event_key=_event_key(
            category="trending",
            action=action,
            scope=scope,
            occurred_at=item.published_at,
        ),
        material_hash=_material_hash(item, "trending", scope),
        reason="editor pick",
        requires_corroboration=False,
        region=item.region,
        action=action,
        entities=_display_entities(f" {item.title} ", item.symbol),
        topic_anchor=_topic_anchor("trending", action, scope),
    )


def assess(item: NewsItem, config: dict[str, Any]) -> Assessment | None:
    raw_text = f" {item.title} {item.summary} "
    text = raw_text.lower()
    if item.source_tier not in {"primary", "structured"} and is_forecast(text):
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
        and is_central_bank_action(text)
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
    elif _contains_any(text, DISASTER_TERMS):
        level, category, reason = "P0", "disaster", "major disaster"
    elif _contains_any(text, POLITICAL_CRISIS_TERMS):
        level, category, reason = "P0", "political_crisis", "political upheaval"
    elif _contains_any(text, HEALTH_EMERGENCY_TERMS):
        level, category, reason = "P0", "health_emergency", "health emergency"
    if not level:
        return None

    entities = _display_entities(raw_text, item.symbol)
    scope = _key_scope(_entities(raw_text, item.symbol), item.title)
    action = ACTION_LABELS[category]
    # Macro entities collapse to the country (US CPI and US NFP both yield USA),
    # so the series has to enter the keys or the two events become one topic.
    detail = _macro_series(text) if category == "macro" else ""
    event_key = _event_key(
        category=category,
        action=action,
        scope=scope,
        occurred_at=item.published_at,
        detail=detail,
    )
    return Assessment(
        level=level,
        category=category,
        event_key=event_key,
        material_hash=_material_hash(item, category, scope),
        reason=reason,
        requires_corroboration=level == "P0"
        and item.source_tier not in {"primary", "structured"},
        region=item.region,
        action=action,
        entities=entities,
        topic_anchor=_topic_anchor(category, action, scope, detail),
    )
