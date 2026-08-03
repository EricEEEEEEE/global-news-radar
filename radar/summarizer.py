from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import requests

from .models import AlertEvent
from .util import echoable_numbers, numeric_values

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Summary:
    title: str
    fact: str
    impact: str


CATEGORY_TITLES = {
    "central_bank": "央行政策突变",
    "geopolitics": "重大地缘事件",
    "crypto_regulation": "加密监管突发",
    "macro": "经济数据大幅偏离",
    "earnings": "巨头财报意外",
    "merger_acquisition": "重大并购动态",
    "financing": "重大融资动态",
    "industry": "行业突发动态",
}

CATEGORY_IMPACTS = {
    "central_bank": "利率与风险资产定价将立即重估。",
    "geopolitics": "避险需求及相关商品波动可能上升。",
    "crypto_regulation": "相关加密资产与平台面临即时监管重估。",
    "macro": "利率预期与风险资产定价可能快速调整。",
    "earnings": "相关公司及同业盘后定价可能重新调整。",
    "merger_acquisition": "交易双方及同业估值可能出现即时反应。",
    "financing": "融资相关公司与行业估值可能重新定价。",
    "industry": "相关产业链的供需预期可能立即变化。",
}


def deterministic_summary(event: AlertEvent) -> Summary:
    primary = event.primary
    category = event.assessment.category
    title = CATEGORY_TITLES.get(category, "市场突发")
    fact = re.sub(r"\s+", " ", primary.title).strip()
    if len(fact) > 150:
        fact = fact[:147] + "…"
    return Summary(
        title=title,
        fact=fact,
        impact=CATEGORY_IMPACTS.get(category, "相关资产可能出现即时重新定价。"),
    )


class LlmSummarizer:
    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
        max_output_tokens: int,
    ):
        self.enabled = enabled and bool(base_url and api_key and model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def summarize(self, event: AlertEvent) -> Summary:
        fallback = deterministic_summary(event)
        if not self.enabled:
            return fallback
        evidence = [
            {
                "title": item.title,
                "summary": item.summary[:500],
                "source": item.source,
                "published_at": item.published_at.isoformat(),
                "actual": item.actual,
                "estimate": item.estimate,
                "unit": item.unit,
            }
            for item in event.items[:4]
        ]
        prompt = {
            "task": (
                "把已确认的金融突发事件压缩成中文 Telegram 速报。"
                "只翻译和压缩，不判断严重度，不添加数字、因果、预测或交易建议。"
            ),
            "requirements": {
                "title": "15个汉字以内，说明事件本身",
                "fact": "一句，核心事实和已有关键数字",
                "language": (
                    "除股票代码外全部用中文，英文指标缩写要译出："
                    "MoM→环比，YoY→同比，QoQ→季环比，CPI→CPI，GDP→GDP"
                ),
                "impact": "一句，只写即时影响方向；不能声称未观察到的市场反应",
                "output": "严格 JSON: title,fact,impact",
            },
            "category": event.assessment.category,
            "evidence": evidence,
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
                    "max_tokens": self.max_output_tokens,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是事实压缩器。输入已由确定性规则验证。"
                                "禁止新增事实、数字、原因、预测或建议。"
                            ),
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
            data = json.loads(content)
            candidate = Summary(
                title=str(data["title"]).strip(),
                fact=str(data["fact"]).strip(),
                impact=str(data["impact"]).strip(),
            )
            if not candidate.title or not candidate.fact or not candidate.impact:
                raise ValueError("empty LLM field")
            allowed_numbers = echoable_numbers(json.dumps(evidence, ensure_ascii=False))
            generated_numbers = numeric_values(
                f"{candidate.title} {candidate.fact} {candidate.impact}"
            )
            if not generated_numbers.issubset(allowed_numbers):
                invented = sorted(generated_numbers - allowed_numbers)
                raise ValueError(f"LLM invented numbers: {invented}")
            if len(candidate.title) > 20:
                # An overlong title is a formatting miss, not a truthfulness one.
                # Dropping the whole summary over it used to send the untranslated
                # English source instead, which is a worse answer than borrowing
                # the category headline and keeping the translated body.
                LOGGER.info("llm_title_too_long len=%d", len(candidate.title))
                return Summary(
                    title=fallback.title,
                    fact=candidate.fact,
                    impact=candidate.impact,
                )
            return candidate
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("llm_summary_fallback error=%s", exc)
            return fallback
