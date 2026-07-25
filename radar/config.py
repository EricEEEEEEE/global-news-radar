from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL = {
    "version",
    "timezone",
    "poll_seconds",
    "freshness_minutes",
    "quiet_window",
    "telegram",
    "runtime",
    "llm",
    "official_feeds",
    "discovery",
    "structured",
}


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config root must be an object")
    missing = REQUIRED_TOP_LEVEL - set(payload)
    if missing:
        raise ValueError(f"missing config fields: {sorted(missing)}")
    if str(payload["timezone"]) != "Asia/Singapore":
        raise ValueError("timezone must be Asia/Singapore")
    if int(payload["discovery"]["max_queries"]) > 4:
        raise ValueError("discovery.max_queries must be <= 4")
    telegram = payload["telegram"]
    try:
        int(str(telegram["chat_id"]))
        news_thread_id = int(str(telegram["news_thread_id"]))
        monitor_thread_id = int(str(telegram["monitor_thread_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Telegram chat and thread IDs must be integers") from exc
    if news_thread_id < 0 or monitor_thread_id < 0:
        raise ValueError("Telegram thread IDs must be >= 0")
    max_visible_chars = int(telegram["max_visible_chars"])
    if not 1 <= max_visible_chars <= 4096:
        raise ValueError("telegram.max_visible_chars must be between 1 and 4096")
    return payload
