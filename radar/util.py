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


def normalize_title(title: str, source: str = "") -> str:
    value = html.unescape(title).strip().lower()
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


def numeric_tokens(value: str) -> set[str]:
    return {
        token.replace(",", "")
        for token in re.findall(r"(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?%?", value or "")
    }


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


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
