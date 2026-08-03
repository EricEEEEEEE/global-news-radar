from __future__ import annotations

import base64
import json
import logging
import re
import time

import requests

from .models import AlertEvent
from .summarizer import Summary
from .util import numeric_tokens, redact_secrets

LOGGER = logging.getLogger(__name__)

STYLE = (
    "flat vector editorial comic, warm colors, bold clean outlines, "
    "consistent characters and palette across panels"
)
# The image model will invent whatever detail it is allowed to invent. Facts
# travel in the Telegram caption, which comes verbatim from the number-gated
# summary; the picture is only mood and domain, so everything that could carry
# a fact is banned from the canvas outright.
HARD_BANS = (
    "No text, no letters, no words, no numbers, no captions, no speech "
    "bubbles, no flags, no logos, no brand marks, no real people's faces."
)

# One stock metaphor per category so a failed or unsafe scene line never
# blocks the comic: the fallback is boring but always true to the category.
CATEGORY_SCENES = {
    "central_bank": (
        "a grand columned bank building turning a huge ship's wheel in the sky"
    ),
    "geopolitics": "two chess kings facing off across a cracked stone map table",
    "crypto_regulation": (
        "a glowing coin under a large magnifying glass held by a robed judge"
    ),
    "macro": "a rooftop weather vane spinning hard while people below look up",
    "disaster": (
        "a cracked landscape at dusk with a rescue helicopter sweeping its light"
    ),
    "political_crisis": ("a toppled marble column on the steps of a government palace"),
    "health_emergency": (
        "a giant microscope casting its glow over a shadowed world map"
    ),
    "trending": "a crowd in a town square gathered around a glowing globe",
}
DEFAULT_SCENE = "a lighthouse beam sweeping across a dark stormy sea"

# Telegram photo captions cap at 1024 characters; radar messages are far
# shorter, but the guard keeps a future config change from losing the caption.
CAPTION_LIMIT = 1024


class ComicArtist:
    """Best-effort comic renderer for alerts. Never blocks the message."""

    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str,
        api_key: str,
        scene_model: str,
        image_model: str,
        scene_timeout: int,
        image_timeout: int,
    ):
        self.enabled = enabled and bool(base_url and api_key and image_model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.scene_model = scene_model
        self.image_model = image_model
        self.scene_timeout = scene_timeout
        self.image_timeout = image_timeout
        self.generated = 0
        self.failed = 0

    def stats(self) -> dict[str, int]:
        return {"generated": self.generated, "failed": self.failed}

    def for_event(self, event: AlertEvent, summary: Summary) -> bytes | None:
        return self.generate([(event.assessment.category, summary.fact)])

    def for_digest(self, entries: list[tuple[AlertEvent, Summary]]) -> bytes | None:
        return self.generate(
            [(event.assessment.category, summary.fact) for event, summary in entries]
        )

    def _fallback_scene(self, category: str) -> str:
        return CATEGORY_SCENES.get(category, DEFAULT_SCENE)

    def _scene_lines(self, entries: list[tuple[str, str]]) -> list[str]:
        """One drawable metaphor per entry, in English, with no facts in it.

        The scene writer sees only the gate-passed Chinese summaries. Its
        output goes straight onto an image model's canvas, so the one property
        that must hold — no numbers — is enforced mechanically per line, and
        any line that fails it degrades to the category's stock scene instead
        of degrading the whole comic.
        """
        fallbacks = [self._fallback_scene(category) for category, _ in entries]
        if not self.scene_model:
            return fallbacks
        prompt = {
            "task": (
                "为每条新闻写一个画面隐喻，交给漫画图像模型作画。"
                "只描述画面，不携带事实；用英文输出。"
            ),
            "requirements": {
                "count": f"scenes 数组必须正好 {len(entries)} 个，顺序对应输入",
                "content": (
                    "每个场景一句英文视觉隐喻，不超过 30 词，只写可画的物体和动作"
                ),
                "bans": (
                    "禁止数字、货币符号、机构名、国名、国旗、真实人物，"
                    "禁止任何会被画进图里的文字"
                ),
                "output": '严格 JSON: {"scenes": ["...", ...]}',
            },
            "news": [
                {"category": category, "fact": fact} for category, fact in entries
            ],
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.scene_model,
                    "temperature": 0.6,
                    "max_tokens": 400,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是漫画分镜师。只输出 JSON。",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(prompt, ensure_ascii=False),
                        },
                    ],
                },
                timeout=self.scene_timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            raw_scenes = json.loads(content)["scenes"]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("comic_scene_fallback error=%s", redact_secrets(exc))
            return fallbacks
        scenes: list[str] = []
        for index, fallback in enumerate(fallbacks):
            raw = (
                re.sub(r"\s+", " ", str(raw_scenes[index])).strip()
                if index < len(raw_scenes)
                else ""
            )
            if not raw or len(raw) > 240 or numeric_tokens(raw):
                scenes.append(fallback)
            else:
                scenes.append(raw)
        return scenes

    def _image_prompt(self, scenes: list[str]) -> str:
        if len(scenes) == 1:
            return (
                f"A single-panel editorial illustration, {STYLE}. "
                f"Scene: {scenes[0]}. {HARD_BANS}"
            )
        panels = " ".join(
            f"Panel {index + 1}: {scene}." for index, scene in enumerate(scenes)
        )
        return (
            f"A {len(scenes)}-panel comic strip in one image, arranged in a "
            f"grid, {STYLE}. {panels} {HARD_BANS}"
        )

    def generate(self, entries: list[tuple[str, str]]) -> bytes | None:
        """Return image bytes for the given (category, fact) entries, or None.

        Never raises: the comic is presentation. A failure is counted, logged
        and surfaced in cycle stats, and the caller sends plain text.
        """
        if not self.enabled or not entries:
            return None
        started = time.monotonic()
        try:
            scenes = self._scene_lines(entries[:4])
            response = requests.post(
                f"{self.base_url}/images/generations",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.image_model,
                    "prompt": self._image_prompt(scenes),
                    "n": 1,
                    "response_format": "b64_json",
                },
                timeout=self.image_timeout,
            )
            response.raise_for_status()
            photo = base64.b64decode(response.json()["data"][0]["b64_json"])
            if len(photo) < 1024:
                raise ValueError(f"image too small: {len(photo)} bytes")
        except Exception as exc:  # noqa: BLE001
            self.failed += 1
            LOGGER.error("comic_generation_failed error=%s", redact_secrets(exc))
            return None
        self.generated += 1
        LOGGER.info(
            "comic_generated bytes=%d seconds=%.1f panels=%d",
            len(photo),
            time.monotonic() - started,
            min(len(entries), 4),
        )
        return photo
