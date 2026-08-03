from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from .comic import CAPTION_LIMIT
from .models import RenderedMessage
from .render import outbox_manifest, validate_html
from .util import atomic_write, atomic_write_bytes, redact_secrets, visible_length

LOGGER = logging.getLogger(__name__)


def telegram_payload(
    *,
    chat_id: str,
    thread_id: str,
    text: str,
    parse_mode: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    topic_id = int(thread_id)
    if topic_id > 0:
        payload["message_thread_id"] = topic_id
    if parse_mode:
        payload["parse_mode"] = parse_mode
    return payload


def post_telegram(
    *,
    token: str,
    chat_id: str,
    thread_id: str,
    html_text: str,
    plain_text: str,
    timeout: int = 20,
) -> dict[str, Any]:
    """Send one message, degrading to complete plain text if HTML is rejected.

    Kept separate from TelegramDelivery because the watchdog has to be able to
    report that the daemon is down without touching the outbox files that hold
    the evidence for the daemon's last real message.
    """
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json=telegram_payload(
            chat_id=chat_id,
            thread_id=thread_id,
            text=html_text,
            parse_mode="HTML",
        ),
        timeout=timeout,
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Telegram non-JSON response status={response.status_code}"
        ) from exc
    if response.ok and payload.get("ok"):
        return {
            "message_id": str(payload["result"]["message_id"]),
            "fallback": None,
        }
    description = str(payload.get("description") or response.text)[:300]
    if "parse entities" not in description.lower():
        raise RuntimeError(
            f"Telegram send failed status={response.status_code}: {description}"
        )
    LOGGER.error("telegram_html_rejected; retrying plain text")
    fallback = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json=telegram_payload(chat_id=chat_id, thread_id=thread_id, text=plain_text),
        timeout=timeout,
    )
    fallback_payload = fallback.json()
    if not fallback.ok or not fallback_payload.get("ok"):
        raise RuntimeError(
            f"Telegram send failed status={response.status_code}: {description}"
        )
    return {
        "message_id": str(fallback_payload["result"]["message_id"]),
        "fallback": "plain_text",
    }


def post_telegram_photo(
    *,
    token: str,
    chat_id: str,
    thread_id: str,
    caption_html: str,
    plain_caption: str,
    photo: bytes,
    timeout: int = 60,
) -> dict[str, Any]:
    """Send one photo, degrading to a plain-text caption if HTML is rejected.

    Raises when the photo itself cannot be delivered; the caller still owns
    the full text, so news is never lost to a picture problem.
    """
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing")

    def _send(caption: str, parse_mode: str | None) -> requests.Response:
        data: dict[str, object] = {"chat_id": chat_id, "caption": caption}
        topic_id = int(thread_id)
        if topic_id > 0:
            data["message_thread_id"] = topic_id
        if parse_mode:
            data["parse_mode"] = parse_mode
        return requests.post(
            f"https://api.telegram.org/bot{token}/sendPhoto",
            data=data,
            files={"photo": ("comic.jpg", photo, "image/jpeg")},
            timeout=timeout,
        )

    response = _send(caption_html, "HTML")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Telegram non-JSON response status={response.status_code}"
        ) from exc
    if response.ok and payload.get("ok"):
        return {
            "message_id": str(payload["result"]["message_id"]),
            "fallback": None,
        }
    description = str(payload.get("description") or response.text)[:300]
    if "parse entities" not in description.lower():
        raise RuntimeError(
            f"Telegram photo send failed status={response.status_code}: {description}"
        )
    LOGGER.error("telegram_photo_html_rejected; retrying plain caption")
    fallback = _send(plain_caption, None)
    fallback_payload = fallback.json()
    if not fallback.ok or not fallback_payload.get("ok"):
        raise RuntimeError(
            f"Telegram photo send failed status={response.status_code}: {description}"
        )
    return {
        "message_id": str(fallback_payload["result"]["message_id"]),
        "fallback": "plain_caption",
    }


class TelegramDelivery:
    def __init__(
        self,
        *,
        token: str,
        chat_id: str,
        outbox_dir: Path,
        max_visible_chars: int,
        send_enabled: bool,
    ):
        self.token = token
        self.chat_id = chat_id
        self.outbox_dir = outbox_dir
        self.max_visible_chars = max_visible_chars
        self.send_enabled = send_enabled

    def _payload(
        self,
        *,
        thread_id: str,
        text: str,
        parse_mode: str | None = None,
    ) -> dict[str, object]:
        return telegram_payload(
            chat_id=self.chat_id,
            thread_id=thread_id,
            text=text,
            parse_mode=parse_mode,
        )

    def _write_outbox(
        self,
        message: RenderedMessage,
        delivery: dict[str, object],
        photo: bytes | None = None,
    ) -> None:
        stamp = message.created_at.strftime("%Y%m%dT%H%M%SZ")
        stem = f"{stamp}-{message.level.lower()}-{message.content_hash[:8]}"
        modality = "photo" if photo is not None else "text"
        atomic_write(self.outbox_dir / f"{stem}.html", message.html)
        atomic_write(
            self.outbox_dir / f"{stem}.json",
            outbox_manifest(message, delivery, modality=modality),
        )
        atomic_write(self.outbox_dir / "latest.html", message.html)
        atomic_write(
            self.outbox_dir / "latest.json",
            outbox_manifest(message, delivery, modality=modality),
        )
        if photo is not None:
            atomic_write_bytes(self.outbox_dir / f"{stem}.jpg", photo)
            atomic_write_bytes(self.outbox_dir / "latest.jpg", photo)

    def send(
        self,
        message: RenderedMessage,
        thread_id: str,
        photo: bytes | None = None,
    ) -> dict[str, Any]:
        validate_html(message.html, self.max_visible_chars)
        # A caption longer than Telegram's photo limit would truncate facts to
        # keep a picture; the facts win and the message goes out as text.
        if photo is not None and visible_length(message.html) > CAPTION_LIMIT:
            LOGGER.warning(
                "comic_caption_too_long visible=%d", visible_length(message.html)
            )
            photo = None
        prepared = {
            "status": "prepared",
            "thread_id": thread_id,
            "visible_chars": visible_length(message.html),
        }
        self._write_outbox(message, prepared, photo=photo)
        if not self.send_enabled:
            return {"ok": True, "dry_run": True, "message_id": "dry-run"}
        if photo is not None:
            try:
                result = post_telegram_photo(
                    token=self.token,
                    chat_id=self.chat_id,
                    thread_id=thread_id,
                    caption_html=message.html,
                    plain_caption=message.plain_text,
                    photo=photo,
                )
                delivered = {
                    "status": (
                        "sent_photo_plain_caption"
                        if result["fallback"]
                        else "sent_photo"
                    ),
                    "thread_id": thread_id,
                    "message_id": result["message_id"],
                }
                self._write_outbox(message, delivered, photo=photo)
                sent_photo: dict[str, Any] = {
                    "ok": True,
                    "dry_run": False,
                    "message_id": result["message_id"],
                    "modality": "photo",
                }
                if result["fallback"]:
                    sent_photo["fallback"] = result["fallback"]
                return sent_photo
            except Exception as exc:  # noqa: BLE001
                # The comic is presentation; the news must still go out. The
                # text path below is the one that existed before comics.
                LOGGER.error(
                    "comic_delivery_failed_falling_back_to_text %s",
                    redact_secrets(exc),
                )
        result = post_telegram(
            token=self.token,
            chat_id=self.chat_id,
            thread_id=thread_id,
            html_text=message.html,
            plain_text=message.plain_text,
        )
        message_id = result["message_id"]
        fallback = result["fallback"]
        delivered = {
            "status": "sent_plain_fallback" if fallback else "sent",
            "thread_id": thread_id,
            "message_id": message_id,
        }
        self._write_outbox(message, delivered)
        sent: dict[str, Any] = {
            "ok": True,
            "dry_run": False,
            "message_id": message_id,
        }
        if fallback:
            sent["fallback"] = fallback
        return sent
