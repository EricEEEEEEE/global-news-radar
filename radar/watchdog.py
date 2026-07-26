"""External liveness check for the radar daemon.

The daemon already reports its own source failures, but a process that has died
cannot report that it died, and neither can one wedged mid-cycle. That gap is
the only thing this module exists to close, so it deliberately depends on as
little of the running system as possible:

- it reads ``state/heartbeat.json`` and nothing else the daemon writes;
- it keeps its own small JSON state instead of the daemon's SQLite, because a
  locked or corrupt database is precisely one of the failures it must report;
- it never writes the daemon's outbox, which holds the evidence for the last
  real message that was sent.

Run it from cron, not from the same supervisor that owns the daemon, so the
supervisor going down does not take the alarm with it.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from . import __version__
from .delivery import post_telegram
from .service import check_health
from .util import SGT, atomic_write, redact_secrets, strip_html, utc_now

LOGGER = logging.getLogger(__name__)

STATUS_TEXT = {
    "missing_heartbeat": "心跳文件不存在，进程可能从未成功启动",
    "unreadable_heartbeat": "心跳文件损坏，无法判断进程状态",
    "stale": "心跳过期，进程可能已死或卡在某一轮",
    "deadletter": "有消息彻底发送失败，P0 可能已丢失",
}


def _load_state(path: Path) -> dict[str, Any]:
    # A watchdog that crashes on its own corrupt state file stops watching, so
    # an unreadable state resets to "nothing known" rather than raising.
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _parse_time(value: object) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _footer(now: datetime) -> str:
    return (
        f"<i>{now.astimezone(SGT).strftime('%Y-%m-%d %H:%M')} SGT · "
        f"global-news-radar/{__version__}</i>"
    )


def _down_message(payload: dict[str, Any], max_age_seconds: int, now: datetime) -> str:
    status = str(payload.get("status", "unknown"))
    reason = STATUS_TEXT.get(status, status)
    lines = [html.escape(reason)]
    age = payload.get("age_seconds")
    if age is not None:
        lines.append(
            f"心跳 {int(float(age)) // 60} 分钟前 · 阈值 {max_age_seconds // 60} 分钟"
        )
    deadletters = int(payload.get("deadletter_deliveries") or 0)
    if deadletters:
        lines.append(f"死信 {deadletters} 条")
    error = payload.get("error")
    if error:
        lines.append(html.escape(redact_secrets(error)[:180]))
    body = "\n".join(lines)
    return f"🚨 <b>新闻雷达停摆</b>\n<blockquote>{body}</blockquote>\n{_footer(now)}"


def _recovery_message(down_since: datetime | None, now: datetime) -> str:
    duration = ""
    if down_since is not None:
        minutes = max(0, int((now - down_since).total_seconds() // 60))
        duration = f"中断约 {minutes} 分钟 · "
    body = f"{duration}心跳恢复正常"
    return f"✅ <b>新闻雷达恢复</b>\n<blockquote>{body}</blockquote>\n{_footer(now)}"


def run_watchdog(
    root: Path,
    config: dict[str, Any],
    env: dict[str, str],
    *,
    max_age_seconds: int,
    send_enabled: bool,
    now: datetime | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Check the heartbeat and alert on the transition, not on every run.

    Returns ``(healthy, report)``. Cron runs this every few minutes, so an
    unhealthy radar must not mean an unhealthy radar's worth of messages: the
    first failure alerts, the cooldown covers the rest, and recovery closes the
    incident the same way a source outage does.
    """
    # One timestamp for the whole decision: measuring heartbeat age against a
    # different clock than the outage duration would let the two disagree.
    now = now or utc_now()
    healthy, payload = check_health(root, max_age_seconds, now)
    state_path = root / "state" / "watchdog.json"
    state = _load_state(state_path)
    down_since = _parse_time(state.get("down_since"))
    alerted_at = _parse_time(state.get("alerted_at"))
    cooldown = timedelta(
        hours=float(config["runtime"]["watchdog_alert_cooldown_hours"])
    )

    token = env.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID") or str(config["telegram"]["chat_id"])
    thread_id = env.get("TELEGRAM_MONITOR_THREAD_ID") or str(
        config["telegram"]["monitor_thread_id"]
    )

    report: dict[str, Any] = {
        "status": payload.get("status"),
        "healthy": healthy,
        "alerted": False,
        "recovered": False,
    }

    def _send(text: str) -> bool:
        if not send_enabled:
            LOGGER.info("watchdog_dry_run status=%s", payload.get("status"))
            return True
        try:
            post_telegram(
                token=token,
                chat_id=chat_id,
                thread_id=thread_id,
                html_text=text,
                plain_text=strip_html(text),
            )
        except Exception as exc:  # noqa: BLE001 - cron sees this as a failure
            LOGGER.error("watchdog_alert_failed %s", redact_secrets(exc))
            report["send_error"] = redact_secrets(exc)[:200]
            return False
        return True

    if healthy:
        # Only close an incident that was actually announced. A blip that never
        # reached the operator must not produce a lone "recovered" notice with
        # no warning in front of it.
        if alerted_at is not None and _send(_recovery_message(down_since, now)):
            report["recovered"] = True
            report["outage_minutes"] = (
                max(0, int((now - down_since).total_seconds() // 60))
                if down_since
                else None
            )
        atomic_write(state_path, json.dumps({"status": "ok"}, ensure_ascii=False))
        return True, report

    down_since = down_since or now
    due = alerted_at is None or (now - alerted_at) >= cooldown
    if due and _send(_down_message(payload, max_age_seconds, now)):
        alerted_at = now
        report["alerted"] = True
    atomic_write(
        state_path,
        json.dumps(
            {
                "status": "down",
                "down_since": down_since.isoformat(),
                "alerted_at": alerted_at.isoformat() if alerted_at else None,
            },
            ensure_ascii=False,
        ),
    )
    return False, report
