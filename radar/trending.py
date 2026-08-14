from __future__ import annotations

import json
import logging

import requests

from .models import NewsItem
from .util import redact_secrets

LOGGER = logging.getLogger(__name__)


class TrendingJudge:
    """Editorial pick of interesting, high-attention world stories.

    "有趣、关注度高" is a qualitative judgment, so an LLM makes it — but only
    inside mechanical rails: it chooses from major-source headlines the radar
    already fetched, it must echo the exact identity of what it picked, and a
    daily cap bounds how much taste it gets to exercise. It never writes a
    word of what is sent; the headline goes through the same summarizer and
    number gate as every other item.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
        max_per_day: int,
    ):
        self.enabled = enabled and bool(base_url and api_key and model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_per_day = max_per_day
        self.picked = 0
        self.failed = 0

    def stats(self) -> dict[str, int]:
        return {"picked": self.picked, "failed": self.failed}

    def pick(self, candidates: list[NewsItem], remaining: int) -> list[NewsItem]:
        if not self.enabled or not candidates or remaining <= 0:
            return []
        limit = min(remaining, 2)
        offered = {item.identity: item for item in candidates[:40]}
        prompt = {
            "task": (
                "你是全球新闻编辑。从下面的标题里挑出真正值得普通读者"
                "现在知道的事：全球高关注度、罕见、或特别有趣。"
                "金融市场数据不算（另有渠道）；公司并购、融资、财报一律不选。"
                "宁缺毋滥，没有就空手而归。"
            ),
            "requirements": {
                "limit": f"最多选 {limit} 条，可以为 0 条",
                "output": '严格 JSON: {"picks": ["<id>", ...]}，id 逐字来自输入',
            },
            "headlines": [
                {"id": item.identity, "source": item.source, "title": item.title}
                for item in offered.values()
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
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": 200,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是克制的新闻编辑。只输出 JSON。",
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
            raw_picks = json.loads(content)["picks"]
        except Exception as exc:  # noqa: BLE001
            self.failed += 1
            LOGGER.warning("trending_judge_failed error=%s", redact_secrets(exc))
            return []
        picks: list[NewsItem] = []
        for value in raw_picks[:limit]:
            # The id must round-trip exactly: an id the judge invented is the
            # editorial version of an invented number, and gets the same door.
            selected = offered.get(str(value))
            if selected is None:
                LOGGER.warning("trending_judge_unknown_id id=%s", str(value)[:60])
                continue
            picks.append(selected)
        self.picked += len(picks)
        if picks:
            LOGGER.info(
                "trending_picked count=%d ids=%s",
                len(picks),
                ",".join(item.identity[:12] for item in picks),
            )
        return picks

    def judge_burst(self, headlines: list[tuple[str, str]]) -> bool | None:
        """Yes/no gate for a vocab-less burst: is this a major world event?

        Several major outlets filing the same story inside 90 minutes is the
        mechanical trigger; this call only separates a genuinely major world
        event from entertainment, sport and routine coverage that also bursts.
        Returns None when the judge cannot answer (disabled, transport, parse)
        so the caller may retry the cluster next cycle; False is a permanent
        editorial rejection.
        """
        if not self.enabled or not headlines:
            return None
        prompt = {
            "task": (
                "多家国际大社在90分钟内同时报道了同一件事。"
                "判断它是否是值得立即推送的重大世界事件："
                "战争或军事冲突升级、重大灾难、政变、恐怖袭击、"
                "公共卫生危机、重大政治剧变等。"
                "娱乐、体育、公司新闻、例行数据、软性专题一律不算。"
            ),
            "requirements": {
                "output": '严格 JSON: {"major_world_event": true 或 false}',
            },
            "headlines": [
                {"source": source, "title": title} for source, title in headlines[:8]
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
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": 60,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是克制的新闻编辑。只输出 JSON。",
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
            verdict = json.loads(content)["major_world_event"]
        except Exception as exc:  # noqa: BLE001
            self.failed += 1
            LOGGER.warning("burst_judge_failed error=%s", redact_secrets(exc))
            return None
        return bool(verdict)
