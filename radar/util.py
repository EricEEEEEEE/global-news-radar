from __future__ import annotations

import email.utils
import hashlib
import html
import json
import logging
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

SGT = ZoneInfo("Asia/Singapore")
TRACKING_KEYS = {
    "fbclid",
    "gclid",
    "ref",
    "source",
    "hl",
    "gl",
    "ceid",
}

REDACTED = "***"
_SECRET_VALUES: set[str] = set()
_SECRET_QUERY_RE = re.compile(
    r"(?i)\b(apikey|api_key|key|token|access_token|secret|password|auth)"
    r"=([^&\s\"'<>]+)"
)
_BOT_TOKEN_RE = re.compile(r"/bot\d+:[A-Za-z0-9_-]+")


def register_secrets(values: Iterable[str]) -> None:
    """Remember credential values so they can be scrubbed from any output."""
    for value in values:
        text = str(value or "").strip()
        if len(text) >= 8:
            _SECRET_VALUES.add(text)


def redact_secrets(value: object) -> str:
    """Remove credentials from text before it is logged, stored or delivered."""
    text = str(value or "")
    if not text:
        return text
    for secret in _SECRET_VALUES:
        if secret in text:
            text = text.replace(secret, REDACTED)
    text = _SECRET_QUERY_RE.sub(rf"\1={REDACTED}", text)
    return _BOT_TOKEN_RE.sub(f"/bot{REDACTED}", text)


class RedactingFormatter(logging.Formatter):
    """Formatter that scrubs credentials from messages *and* tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_secrets(super().format(record))


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    cleaned = value.strip()
    try:
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(cleaned)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_KEYS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("https", host, path, urlencode(sorted(query)), ""))


# A trailing " - Outlet" / " | Outlet" tail. Kept deliberately narrow: no
# separator inside the tail, so "Trump - Putin summit - Reuters" surrenders
# only the last segment.
_SOURCE_SUFFIX_RE = re.compile(r"\s+[-–—|]\s+([^-–—|]{2,60})\s*$")
# Lowercase words that are legitimately part of an outlet name. Everything
# else lowercase in the tail means the tail is prose, not a byline.
_SUFFIX_CONNECTORS = {"of", "the", "and", "de", "la", "le", "du", "des", "for"}


def _looks_like_outlet(tail: str) -> bool:
    """True when the tail reads as a publication name rather than prose.

    "Reuters", "The New York Times", "Voice of America" pass. "what we know",
    "live updates", "explained" do not, so a headline whose own words follow a
    dash keeps them.
    """
    words = tail.split()
    if not words or len(words) > 5:
        return False
    alphabetic = [word for word in words if re.search(r"[A-Za-z]", word)]
    if not alphabetic or len(alphabetic) != len(words):
        return False
    capitalized = [word for word in alphabetic if word[:1].isupper()]
    if not capitalized:
        return False
    return all(
        word in capitalized or word.lower() in _SUFFIX_CONNECTORS for word in alphabetic
    )


def strip_source_suffix(title: str, known_sources: Iterable[str] = ()) -> str:
    """Drop the publisher byline Google News and most wires append to titles.

    Three quarters of the corpus carries one, and it is pure noise for a
    reader: it eats the character budget of a brief line, and it poisons
    tokenisation, where "reuters" becomes a token shared by every unrelated
    Reuters story. Removed only when the tail is recognisably an outlet, and
    never when it would leave a stub behind.
    """
    text = str(title or "").strip()
    match = _SOURCE_SUFFIX_RE.search(text)
    if not match:
        return text
    tail = match.group(1).strip()
    known = {
        str(name).strip().casefold() for name in known_sources if str(name).strip()
    }
    if not (tail.casefold() in known or _looks_like_outlet(tail)):
        return text
    head = text[: match.start()].strip()
    return head if len(head) >= 12 else text


def normalize_title(title: str, source: str = "") -> str:
    value = strip_source_suffix(html.unescape(title).strip()).lower()
    if source:
        value = re.sub(
            rf"\s*[-–—|]\s*{re.escape(source.strip().lower())}\s*$",
            "",
            value,
        )
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value, flags=re.UNICODE)


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).hexdigest()[:length]


def simhash64(value: str) -> str:
    text = normalize_title(value)
    shingles = [text[index : index + 3] for index in range(max(1, len(text) - 2))]
    vector = [0] * 64
    for shingle in shingles:
        bits = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for index in range(64):
            vector[index] += 1 if bits & (1 << index) else -1
    result = 0
    for index, score in enumerate(vector):
        if score >= 0:
            result |= 1 << index
    return f"{result:016x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def strip_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()


def visible_length(value: str) -> int:
    return len(strip_html(value))


# The lookbehind excludes ASCII word characters only. `(?<!\w)` also excluded CJK,
# so a minus sign written as 环比-0.14% was dropped and the token came out as a
# positive 0.14% — the translated number then looked invented next to a source
# that said -0.14. Digits and dots stay excluded so 2026-08-03 does not yield -8.
_NUMBER_RE = re.compile(r"(?<![0-9A-Za-z_.])[+-]?\d[\d,]*(?:\.\d+)?%?")
# Digits glued to letters: Q2, H1, FY26, 10Y. Used only to widen what the model is
# allowed to echo, never to decide what it generated.
_GLUED_DIGIT_RE = re.compile(r"(?<=[A-Za-z])\d+(?:\.\d+)?")

_MONTH_NUMBERS = {
    "january": "1",
    "jan": "1",
    "february": "2",
    "feb": "2",
    "march": "3",
    "mar": "3",
    "april": "4",
    "apr": "4",
    "may": "5",
    "june": "6",
    "jun": "6",
    "july": "7",
    "jul": "7",
    "august": "8",
    "aug": "8",
    "september": "9",
    "sep": "9",
    "sept": "9",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}


def numeric_tokens(value: str) -> set[str]:
    return {token.replace(",", "") for token in _NUMBER_RE.findall(value or "")}


def normalize_number(token: str) -> str:
    """Compare numbers by value, not by how they were typed.

    A percent sign is a unit, not a magnitude: the source carries it in its own
    `unit` field, so a translation writing -0.14% from actual=-0.14 is correct
    rather than invented. Trailing zeros are normalised for the same reason.
    Sign is preserved — flipping one reverses the meaning of the number.
    """
    cleaned = token.replace(",", "").rstrip("%").lstrip("+")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    normalized = value.normalize()
    # Decimal normalises 100 to 1E+2; expand it back so the strings compare.
    return format(normalized, "f")


def numeric_values(value: str) -> set[str]:
    return {normalize_number(token) for token in numeric_tokens(value)}


# English scale words next to a number, e.g. "$8 billion", "3.8bn", "500k".
_SCALE_RE = re.compile(
    r"([+-]?\d[\d,]*(?:\.\d+)?)\s*(trillion|billion|million|thousand|tn|bn|mn|[kmbt])"
    r"(?![a-z])",
    re.IGNORECASE,
)
# Chinese rewrites the magnitude into 万/亿 units, so the digits change:
# $8 billion becomes 80亿, $3.8 billion becomes 38亿, 5 million becomes 500万.
# Each factor is the digit shift that Chinese-scale wording produces.
_SCALE_FACTORS = {
    "trillion": (Decimal(1), Decimal(10000)),  # 8万亿 / 80000亿
    "tn": (Decimal(1), Decimal(10000)),
    "t": (Decimal(1), Decimal(10000)),
    "billion": (Decimal(10),),  # 80亿
    "bn": (Decimal(10),),
    "b": (Decimal(10),),
    "million": (Decimal(100),),  # 800万
    "mn": (Decimal(100),),
    "m": (Decimal(100),),
    "thousand": (Decimal(1000), Decimal("0.1")),  # 8000 / 150 thousand -> 15万
    "k": (Decimal(1000), Decimal("0.1")),
}


def echoable_numbers(value: str) -> set[str]:
    """Numbers a faithful translation may contain, given this source text.

    Wider than what the tokeniser sees literally: rendering a month name or a
    quarter label in Chinese turns Jul into 7月 and Q2 into 第2季度, and an
    English scale word rescales the digits themselves — "$8 billion" is
    correctly written 80亿, which is a "new" number only to a byte-level
    comparison. Four real alerts fell back to English over exactly that.
    """
    allowed = numeric_values(value)
    glued = _GLUED_DIGIT_RE.findall(value or "")
    allowed |= {normalize_number(token) for token in glued}
    lowered = (value or "").lower()
    for name, number in _MONTH_NUMBERS.items():
        if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", lowered):
            allowed.add(number)
    for match in _SCALE_RE.finditer(value or ""):
        try:
            base = Decimal(match.group(1).replace(",", "").lstrip("+"))
        except InvalidOperation:
            continue
        for factor in _SCALE_FACTORS[match.group(2).lower()]:
            allowed.add(format((base * factor).normalize(), "f"))
    return allowed


# Structured rows carry the scale in their own `unit` field ("K", "M", "B"),
# so the number is pre-divided and the tokeniser never sees a scale word next
# to it: payrolls actual=150 with unit "K" is 150 thousand, correctly written
# 15万 or 150000 in Chinese. Non-scale units ("%", "Low") widen nothing.
_UNIT_SCALE_ALIASES = {
    "t": "trillion",
    "tn": "trillion",
    "trn": "trillion",
    "trillion": "trillion",
    "trillions": "trillion",
    "b": "billion",
    "bn": "billion",
    "bln": "billion",
    "billion": "billion",
    "billions": "billion",
    "m": "million",
    "mn": "million",
    "mln": "million",
    "million": "million",
    "millions": "million",
    "k": "thousand",
    "thousand": "thousand",
    "thousands": "thousand",
}


def unit_scaled_values(number: object, unit: str) -> set[str]:
    """Chinese-scale spellings of a value whose unit is an English scale word."""
    key = _UNIT_SCALE_ALIASES.get((unit or "").strip().lower())
    if key is None or number is None:
        return set()
    try:
        base = Decimal(str(number))
    except InvalidOperation:
        return set()
    values = {format(base.normalize(), "f")}
    for factor in _SCALE_FACTORS[key]:
        values.add(format((base * factor).normalize(), "f"))
    return values


def load_env(path: Path) -> dict[str, str]:
    result = dict(os.environ)
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result.setdefault(key.strip(), value.strip().strip("'\""))
    return result


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
