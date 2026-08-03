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
    "disaster": "重大灾难事件",
    "political_crisis": "政治局势剧变",
    "health_emergency": "公共卫生事件",
    "trending": "全球热点",
}

CATEGORY_IMPACTS = {
    "central_bank": "利率与风险资产定价将立即重估。",
    "geopolitics": "避险需求及相关商品波动可能上升。",
    "crypto_regulation": "相关加密资产与平台面临即时监管重估。",
    "macro": "利率预期与风险资产定价可能快速调整。",
    "disaster": "当地伤亡与基础设施影响待确认。",
    "political_crisis": "该国局势与地区稳定面临不确定性。",
    "health_emergency": "传播范围与防控措施值得关注。",
    "trending": "全球关注度高，值得了解。",
}


# An ASCII word-run of four or more characters that contains a lowercase
# letter. All-caps stays exempt: tickers and indicator acronyms (TSLA, CPI)
# are the one kind of English the operator accepts bare.
_BARE_ENGLISH_RE = re.compile(r"[A-Za-z][A-Za-z0-9 .&'’-]{3,}")


def bare_english_spans(text: str) -> list[str]:
    """English name-runs that lack a Chinese gloss right behind them.

    "Visa（维萨）" is fine; "Clearmind Medicine将收购…" is what the operator
    flagged: a bare English company name in an otherwise Chinese sentence.
    """
    spans: list[str] = []
    for match in _BARE_ENGLISH_RE.finditer(text or ""):
        span = match.group(0).strip(" .&'’-")
        if not re.search(r"[a-z]", span):
            continue
        tail = text[match.end() :].lstrip()
        if tail.startswith("（") or tail.startswith("("):
            continue
        spans.append(span)
    return spans


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

    def _request(
        self,
        event: AlertEvent,
        evidence: list[dict[str, object]],
        feedback: str,
    ) -> Summary:
        prompt: dict[str, object] = {
            "task": (
                "把已确认的突发新闻压缩成中文 Telegram 速报。"
                "只翻译和压缩，不判断严重度，不添加数字、因果、预测或交易建议。"
            ),
            "requirements": {
                "title": "15个汉字以内，说明事件本身",
                "fact": "一句，核心事实和已有关键数字",
                "language": (
                    "除股票代码和 CPI/GDP 这类指标缩写外全部用中文，"
                    "指标缩写译出：MoM→环比，YoY→同比，QoQ→季环比，EV→电动车。"
                    "公司、机构、人名保留英文时必须紧跟括号中文译名或说明，"
                    "例如 Visa（维萨）、Lantheus（美国放射性药物公司）；"
                    "不允许出现没有中文注释的英文词组。"
                ),
                "impact": "一句，只写即时影响方向；不能声称未观察到的市场反应",
                "output": "严格 JSON: title,fact,impact",
            },
            "category": event.assessment.category,
            "evidence": evidence,
        }
        if feedback:
            prompt["revision_note"] = feedback
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
        data = json.loads(response.json()["choices"][0]["message"]["content"])
        return Summary(
            title=str(data["title"]).strip(),
            fact=str(data["fact"]).strip(),
            impact=str(data["impact"]).strip(),
        )

    def _finish(self, candidate: Summary, fallback: Summary) -> Summary:
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
        allowed_numbers = echoable_numbers(json.dumps(evidence, ensure_ascii=False))
        feedback = ""
        # A translation the gate rejected once is usually one correction away
        # from passing; telling the model what was wrong and asking again is
        # cheaper than shipping the English original. One retry, then decide.
        soft_pass: Summary | None = None
        for _ in range(2):
            try:
                candidate = self._request(event, evidence, feedback)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("llm_summary_retry reason=transport error=%s", exc)
                feedback = ""
                continue
            if not candidate.title or not candidate.fact or not candidate.impact:
                feedback = "上一版有字段为空，三个字段都必须给出。"
                LOGGER.warning("llm_summary_retry reason=empty_field")
                continue
            generated_numbers = numeric_values(
                f"{candidate.title} {candidate.fact} {candidate.impact}"
            )
            if not generated_numbers.issubset(allowed_numbers):
                invented = sorted(generated_numbers - allowed_numbers)
                feedback = (
                    f"上一版包含原文没有的数字 {invented}，删除或改回原文数字后重写。"
                )
                LOGGER.warning("llm_summary_retry reason=invented_numbers %s", invented)
                continue
            spans = bare_english_spans(
                f"{candidate.title} {candidate.fact} {candidate.impact}"
            )
            if spans:
                # Formatting miss, not a truthfulness one: keep the draft as a
                # soft pass so a repeat offence still beats the English title.
                soft_pass = candidate
                feedback = (
                    "这些英文词组缺少括号中文注释：" + "、".join(spans[:5]) + "。"
                    "保留英文原名并紧跟（中文译名或说明）后重写。"
                )
                LOGGER.info("llm_summary_retry reason=bare_english %s", spans[:5])
                continue
            return self._finish(candidate, fallback)
        if soft_pass is not None:
            return self._finish(soft_pass, fallback)
        LOGGER.warning("llm_summary_fallback error=retries exhausted")
        return fallback
