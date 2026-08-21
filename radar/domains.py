"""What kind of news a story is, and how many of each kind a brief may carry.

The radar ranked stories by cross-outlet consensus alone, and consensus is
exactly what a single huge event manufactures: one earthquake filed by forty
outlets produced the top eight slots of a brief that was supposed to describe
a whole world. The corpus was never the problem — politics, economy and
science were all present underneath. Ranking was.

So a brief is packed, not sliced: every story gets a domain, each domain gets
a quota, and slots are filled greedily in score order subject to those quotas.
Whatever the quotas leave unfilled is handed back to the ranking, so a genuinely
quiet day still produces a full brief rather than empty sections.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping

import requests

from .util import redact_secrets

LOGGER = logging.getLogger(__name__)

# Priority order, used to break classifier ties and to seed section order.
DOMAIN_ORDER = (
    "disaster",
    "politics",
    "economy",
    "science",
    "tech",
    "society",
    "sport",
    "other",
)

DOMAIN_LABELS = {
    "disaster": "天灾与事故",
    "politics": "政治与地缘",
    "economy": "经济与市场",
    "science": "健康与科学",
    "tech": "科技",
    "society": "社会",
    "sport": "体育",
    "other": "其他",
}

DOMAIN_EMOJI = {
    "disaster": "🌋",
    "politics": "🏛",
    "economy": "📈",
    "science": "🔬",
    "tech": "💻",
    "society": "⚖️",
    "sport": "🏅",
    "other": "🗂",
}

# Substring cues, matched against the lowercased headline. Substrings rather
# than tokens so "earthquake"/"quake" and "evacuat(e|ion)" both land.
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "disaster": (
        "earthquake",
        "quake",
        "magnitude",
        "tsunami",
        "aftershock",
        "hurricane",
        "typhoon",
        "cyclone",
        "flood",
        "wildfire",
        "landslide",
        "volcano",
        "eruption",
        "drought",
        "tornado",
        "evacuat",
        "casualt",
        "death toll",
        "rescue",
        "collapse",
        "derail",
        "blaze",
        "storm",
        "mudslide",
        "famine",
    ),
    "politics": (
        "election",
        "vote",
        "ballot",
        "parliament",
        "senate",
        "congress",
        "coup",
        "sanction",
        "treaty",
        "summit",
        "ceasefire",
        "truce",
        "airstrike",
        "missile",
        "troops",
        "military",
        "invasion",
        "hostage",
        "protest",
        "impeach",
        "cabinet",
        "referendum",
        "diplomat",
        "embassy",
        "nato",
        "militant",
        "rebel",
        "junta",
        " war ",
        "peace talks",
        "minister",
        "president",
        "prime minister",
        "border",
        "occupation",
        # Conflict wires phrase the same event a dozen ways; without these the
        # single largest politics beat scored zero and fell through to "other".
        "strike kills",
        "strikes kill",
        "air strike",
        "drone strike",
        "shelling",
        "gunmen",
        "clashes",
        "offensive",
        "militia",
        "hostilities",
        "security forces",
        "soldiers",
        "peacekeep",
        "annex",
        "lawmakers",
        "sanctions",
        "foreign ministry",
        "opposition leader",
    ),
    "economy": (
        "inflation",
        "cpi",
        "gdp",
        "unemployment",
        "jobless",
        "payroll",
        "rate cut",
        "rate hike",
        "central bank",
        "federal reserve",
        "the fed",
        "ecb",
        "bond",
        "yield",
        "stocks",
        "shares",
        "earnings",
        "revenue",
        "profit",
        "quarterly",
        "ipo",
        "merger",
        "acquisition",
        "bankrupt",
        "layoff",
        "tariff",
        "trade deal",
        " oil ",
        "crude",
        "opec",
        " gold ",
        "dollar",
        "yuan",
        "currency",
        "recession",
        "bitcoin",
        "crypto",
        "ethereum",
        "exchange",
        "investors",
        "economy",
    ),
    "science": (
        "virus",
        "outbreak",
        "pandemic",
        "vaccine",
        "disease",
        "infection",
        "measles",
        "influenza",
        "covid",
        "ebola",
        "cholera",
        "hospital",
        "cancer",
        "clinical trial",
        "researchers",
        "scientists",
        "study finds",
        "nasa",
        "telescope",
        "climate",
        "emissions",
        "species",
        "fossil",
        "spacecraft",
        "asteroid",
        "who warns",
        "health officials",
    ),
    "tech": (
        "artificial intelligence",
        " ai ",
        "chatbot",
        "chip",
        "semiconductor",
        "nvidia",
        "openai",
        "software",
        "smartphone",
        "robot",
        "quantum",
        "satellite",
        "spacex",
        "cyber",
        "hacker",
        "data breach",
        "startup",
        "cloud",
        "algorithm",
        "app store",
        "social media",
        "tiktok",
        "google",
        "apple",
        "microsoft",
        "meta ",
    ),
    "society": (
        "court",
        "trial",
        "verdict",
        "arrest",
        "police",
        "shooting",
        "murder",
        "prison",
        "immigration",
        "migrant",
        "refugee",
        " union ",
        "school",
        "university",
        "church",
        "festival",
        "museum",
        "film",
        "music",
        "award",
        "wedding",
        "funeral",
        "royal",
        "celebrity",
    ),
    # Measured against 48 live BBC Sport and Guardian Sport headlines, not
    # guessed. The competition-name list alone caught 10 of them: most sport
    # copy is match reporting, where the signal sits in team and player names,
    # which is an unbounded set no keyword list can hold. The rest is rescued
    # by domains.source_hints, and these cues only have to cover the headlines
    # that reach the brief from a general feed. Deliberately excluded as too
    # collision-prone in a world feed: "transfer" (wealth transfer), "goal"
    # (inflation goal), "masters" (a degree), "season", "squad", "coach".
    "sport": (
        "world cup",
        "olympic",
        "football",
        "soccer",
        "premier league",
        "nba",
        "tennis",
        "formula 1",
        " f1 ",
        "champions league",
        "tournament",
        "cricket",
        "golf",
        "medal",
        "championship",
        "grand slam",
        "fifa",
        "uefa",
        "playoff",
        " gp ",
        "grand prix",
        "formula one",
        "rugby",
        "wicket",
        "innings",
        "test match",
        "the ashes",
        "hat-trick",
        "first-leg",
        "second-leg",
        "midfielder",
        "goalkeeper",
        "striker",
        "penalty",
        "relegation",
        "europa",
        "la liga",
        "bundesliga",
        "serie a",
        "ligue 1",
        "fa cup",
        "efl",
        "ryder cup",
        "us open",
        "wimbledon",
        "prize fund",
        "nfl",
        "mlb",
        "nhl",
        "ufc",
        "netball",
        "sevens",
        "marathon",
        "tour de france",
        "athletics",
        "horse racing",
    ),
}


def classify_keywords_scored(title: str) -> tuple[str, int]:
    """Deterministic domain for one headline, plus how many cues backed it.

    The count is the caller's confidence signal: one cue in a long headline is
    frequently an accident ("collapsed form" is not a disaster), while two or
    more is real evidence.
    """
    text = f" {str(title or '').lower()} "
    scores: Counter[str] = Counter()
    for domain, cues in _KEYWORDS.items():
        hits = sum(1 for cue in cues if cue in text)
        if hits:
            scores[domain] = hits
    if not scores:
        return "other", 0
    best = max(scores.values())
    for domain in DOMAIN_ORDER:
        if scores.get(domain) == best:
            return domain, best
    return "other", 0


def classify_keywords(title: str) -> str:
    """Deterministic domain for one headline; the floor under the LLM."""
    return classify_keywords_scored(title)[0]


def resolve_domain(
    title: str, source_ids: Iterable[str], hints: Mapping[str, str]
) -> str:
    """Keyword domain for one headline, with single-domain feeds as a prior.

    A feed like BBC Sport or USGS only ever publishes one domain, so its id is
    a fact about the story that no amount of vocabulary matching can beat --
    measured on live sport headlines, keywords alone reach 18 of 48 because the
    signal is team and player names, an unbounded set. The hint therefore wins
    over a weak keyword result, but never over a well-evidenced one: a stadium
    disaster filed by a sport desk should still read as a disaster.
    """
    domain, score = classify_keywords_scored(title)
    if score >= 2:
        return domain
    votes = Counter(hints[source_id] for source_id in source_ids if source_id in hints)
    if not votes:
        return domain
    top = max(votes.values())
    for candidate in DOMAIN_ORDER:
        if votes.get(candidate) == top:
            return candidate
    return domain


class DomainClassifier:
    """Batched LLM labelling with the keyword classifier as a hard floor.

    The LLM never gets to invent a domain: anything outside DOMAIN_ORDER, any
    short response, any exception, and that story keeps its keyword label. So
    the worst case of an LLM outage is slightly coarser sectioning, never a
    missing or mislabelled brief.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
    ):
        self.enabled = bool(enabled and base_url and api_key and model)
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def classify(
        self, titles: list[str], baseline: list[str] | None = None
    ) -> list[str]:
        baseline = list(baseline or [classify_keywords(title) for title in titles])
        if not self.enabled or not titles:
            return baseline
        try:
            labels = self._ask(titles)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("domain_classify_failed error=%s", redact_secrets(exc))
            return baseline
        result = list(baseline)
        for index, label in enumerate(labels[: len(titles)]):
            cleaned = str(label or "").strip().lower()
            if cleaned in DOMAIN_LABELS:
                result[index] = cleaned
        return result

    def _ask(self, titles: list[str]) -> list[str]:
        prompt = {
            "task": "给每条英文新闻标题分配一个领域标签。",
            "labels": list(DOMAIN_ORDER),
            "requirements": {
                "count": f"domains 数组必须正好 {len(titles)} 条，顺序对应输入",
                "vocabulary": "只能使用 labels 中的词，不得新造",
                "output": '严格 JSON: {"domains": ["...", ...]}',
            },
            "titles": titles,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": 600,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "你是新闻分类器。只输出 JSON。"},
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
        return [str(value) for value in json.loads(content)["domains"]]


def pack_by_quota(
    stories: list[dict[str, object]],
    quotas: dict[str, int],
    limit: int,
) -> list[dict[str, object]]:
    """Fill ``limit`` slots from score-ordered ``stories`` under per-domain caps.

    Two passes on purpose. The first enforces the quotas, which is what stops
    one event owning the brief. The second ignores them, which is what stops a
    quiet day producing a three-line brief: unused slots go back to whatever
    ranked highest, quota or not.
    """
    chosen: list[dict[str, object]] = []
    taken: set[int] = set()
    used: Counter[str] = Counter()
    for index, story in enumerate(stories):
        if len(chosen) >= limit:
            break
        domain = str(story.get("domain") or "other")
        if used[domain] >= int(quotas.get(domain, 1)):
            continue
        used[domain] += 1
        taken.add(index)
        chosen.append(story)
    for index, story in enumerate(stories):
        if len(chosen) >= limit:
            break
        if index not in taken:
            chosen.append(story)
    return chosen


def group_sections(
    stories: list[dict[str, object]],
) -> list[tuple[str, list[dict[str, object]]]]:
    """Stories bucketed by domain, biggest story's section first.

    Sections lead with the day's largest story rather than a fixed running
    order, so the brief still opens on what actually mattered.
    """
    buckets: dict[str, list[dict[str, object]]] = {}
    for story in stories:
        buckets.setdefault(str(story.get("domain") or "other"), []).append(story)
    return sorted(
        buckets.items(),
        key=lambda entry: (
            -max(int(story.get("score") or 0) for story in entry[1]),
            DOMAIN_ORDER.index(entry[0]) if entry[0] in DOMAIN_ORDER else 99,
        ),
    )
