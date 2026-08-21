from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import quote_plus

import requests
from defusedxml import ElementTree as SafeET

from .models import NewsItem
from .store import RadarStore
from .util import parse_datetime, stable_hash, strip_source_suffix

LOGGER = logging.getLogger(__name__)


def _child_text(node: Any, *names: str) -> str:
    for name in names:
        child = node.find(name)
        if child is not None and child.text:
            return child.text.strip()
        for candidate in list(node):
            if (
                candidate.tag.rsplit("}", 1)[-1].lower() == name.lower()
                and candidate.text
            ):
                return candidate.text.strip()
    return ""


def _link(node: Any) -> str:
    direct = _child_text(node, "link")
    if direct:
        return direct
    for candidate in list(node):
        if candidate.tag.rsplit("}", 1)[-1].lower() == "link":
            return str(candidate.attrib.get("href", ""))
    return ""


class HttpClient:
    def __init__(self, user_agent: str, timeout: int = 18, attempts: int = 2):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": (
                    "application/rss+xml, application/atom+xml, "
                    "application/json, text/xml;q=0.9"
                ),
            }
        )
        self.timeout = timeout
        self.attempts = max(1, attempts)

    def request(self, url: str, **kwargs: Any) -> requests.Response:
        """GET with a short retry, because one dropped TCP connection to a
        central-bank feed otherwise costs a whole polling cycle of coverage."""
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                return self.session.get(url, timeout=self.timeout, **kwargs)
            except requests.RequestException as exc:
                last = exc
                if attempt + 1 < self.attempts:
                    LOGGER.warning("http_retry url=%s attempt=%s", url, attempt + 1)
        raise last if last else RuntimeError("request failed")

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        response = self.request(url, **kwargs)
        response.raise_for_status()
        return response


class RssCollector:
    def __init__(self, client: HttpClient, store: RadarStore):
        self.client = client
        self.store = store
        # An inspection run must not consume the daemon's conditional-GET
        # cache: storing an ETag here would make the next real poll a 304 and
        # silently drop that batch of headlines.
        self.readonly = False

    def collect(
        self,
        *,
        source_id: str,
        source_name: str,
        url: str,
        source_tier: str,
        region: str,
        category_hint: str,
        now: datetime,
    ) -> list[NewsItem]:
        headers: dict[str, str] = {}
        if not self.readonly:
            etag = self.store.get_meta(f"etag:{source_id}")
            modified = self.store.get_meta(f"modified:{source_id}")
            if etag:
                headers["If-None-Match"] = etag
            if modified:
                headers["If-Modified-Since"] = modified
        response = self.client.request(url, headers=headers)
        if response.status_code == 304:
            return []
        response.raise_for_status()
        if not self.readonly:
            if response.headers.get("ETag"):
                self.store.set_meta(f"etag:{source_id}", response.headers["ETag"], now)
            if response.headers.get("Last-Modified"):
                self.store.set_meta(
                    f"modified:{source_id}", response.headers["Last-Modified"], now
                )
        root = SafeET.fromstring(response.content)
        entries = [
            node
            for node in root.iter()
            if node.tag.rsplit("}", 1)[-1].lower() in {"item", "entry"}
        ]
        result: list[NewsItem] = []
        for entry in entries:
            title = _child_text(entry, "title")
            link = _link(entry)
            published = parse_datetime(
                _child_text(entry, "pubDate", "published", "updated", "date")
            )
            if not title or not link or published is None:
                continue
            item_source = _child_text(entry, "source") or source_name
            summary = _child_text(entry, "description", "summary", "content")
            # Identity stays keyed on the raw title: recomputing it over a
            # cleaned title would make every already-seen item look new on the
            # first poll after deploy and replay the whole backlog as alerts.
            identity = stable_hash(
                f"{source_id}|{_child_text(entry, 'guid', 'id') or link}|{title}", 24
            )
            result.append(
                NewsItem(
                    identity=identity,
                    title=strip_source_suffix(title, (item_source, source_name)),
                    url=link,
                    source=item_source,
                    source_id=source_id,
                    source_tier=source_tier,
                    published_at=published,
                    fetched_at=now,
                    summary=summary,
                    region=region,
                    category_hint=category_hint,
                )
            )
        return result


def discovery_queries(now_sgt: datetime) -> list[tuple[str, str]]:
    queries = [
        ("breaking_finance", "breaking financial news when:1h"),
        ("crypto", "crypto Bitcoin regulation breaking news when:1h"),
    ]
    hour = now_sgt.hour
    if 0 <= hour < 8:
        queries.append(("asia", "China Japan Korea market breaking news when:1h"))
    elif 8 <= hour < 16:
        queries.append(
            ("europe", "ECB Europe energy geopolitical breaking news when:1h")
        )
    else:
        queries.append(("us", "Fed Treasury US market breaking news when:1h"))
    queries.append(
        (
            "world",
            "earthquake OR tsunami OR coup OR ceasefire OR airstrike "
            "OR outbreak when:1h",
        )
    )
    return queries


class GoogleNewsCollector:
    def __init__(self, rss: RssCollector, max_queries: int = 3):
        self.rss = rss
        self.max_queries = max(1, max_queries)

    def collect(self, now: datetime, now_sgt: datetime) -> list[NewsItem]:
        result: list[NewsItem] = []
        for query_id, query in discovery_queries(now_sgt)[: self.max_queries]:
            url = (
                "https://news.google.com/rss/search?q="
                f"{quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
            )
            items = self.rss.collect(
                source_id=f"gnews:{query_id}",
                source_name="Google News",
                url=url,
                source_tier="secondary",
                region=query_id if query_id in {"asia", "europe", "us"} else "global",
                category_hint="discovery",
                now=now,
            )
            result.extend(items)
        return result


class GdeltCollector:
    ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
    # GDELT is theme-indexed: the keyword queries above are Google News
    # syntax whose implicit AND matches next to nothing here. World-event
    # themes give this fallback tier its own beat — disasters, conflicts,
    # health crises — instead of echoing the regional market query.
    WORLD_QUERY = (
        "(theme:NATURAL_DISASTER OR theme:ARMEDCONFLICT"
        " OR theme:HEALTH_PANDEMIC OR theme:TERROR)"
    )

    def __init__(self, client: HttpClient):
        self.client = client

    def collect(self, now: datetime, now_sgt: datetime) -> list[NewsItem]:
        response = self.client.get(
            self.ENDPOINT,
            params={
                "query": self.WORLD_QUERY,
                "mode": "artlist",
                "format": "json",
                "maxrecords": 50,
                "timespan": "60min",
                "sort": "HybridRel",
            },
        )
        payload = response.json()
        result: list[NewsItem] = []
        for row in payload.get("articles", []):
            title = str(row.get("title") or "").strip()
            url = str(row.get("url") or "").strip()
            published = parse_datetime(str(row.get("seendate") or ""))
            if not title or not url or published is None:
                continue
            source = str(row.get("domain") or "GDELT")
            result.append(
                NewsItem(
                    identity=stable_hash(f"gdelt|{url}|{title}", 24),
                    title=strip_source_suffix(title, (source,)),
                    url=url,
                    source=source,
                    source_id="gdelt",
                    source_tier="fallback",
                    published_at=published,
                    fetched_at=now,
                    region="global",
                    category_hint="discovery",
                    raw={
                        "language": row.get("language"),
                        "sourcecountry": row.get("sourcecountry"),
                    },
                )
            )
        return result


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").replace("%", ""))
    except ValueError:
        return None


def _has_clock(value: Any) -> bool:
    text = str(value or "").strip()
    return "T" in text or bool(re.search(r"\s\d{1,2}:\d{2}(?::\d{2})?", text))


class FmpCollector:
    BASE = "https://financialmodelingprep.com/stable"

    def __init__(self, client: HttpClient, api_key: str, config: dict[str, Any]):
        self.client = client
        self.api_key = api_key
        self.config = config

    def _get(self, endpoint: str, now: datetime) -> list[dict[str, Any]]:
        start = (now - timedelta(days=1)).date().isoformat()
        end = now.date().isoformat()
        response = self.client.get(
            f"{self.BASE}/{endpoint}",
            params={"from": start, "to": end, "apikey": self.api_key},
        )
        payload = response.json()
        if not isinstance(payload, list):
            detail = (
                sorted(payload)[:8]
                if isinstance(payload, dict)
                else type(payload).__name__
            )
            raise RuntimeError(f"FMP returned unexpected payload: {detail}")
        return payload

    def collect_macro(self, now: datetime) -> list[NewsItem]:
        if not self.api_key:
            return []
        watch = [item.lower() for item in self.config["macro_events"]]
        result: list[NewsItem] = []
        for row in self._get("economic-calendar", now):
            name = str(row.get("event") or row.get("name") or "").strip()
            if not name or not any(term in name.lower() for term in watch):
                continue
            actual = _float(row.get("actual"))
            estimate = _float(row.get("estimate") or row.get("consensus"))
            raw_date = row.get("date")
            if not _has_clock(raw_date):
                continue
            published = parse_datetime(str(raw_date or ""))
            if actual is None or estimate is None or published is None:
                continue
            country = str(row.get("country") or "").strip()
            unit = str(row.get("unit") or row.get("impact") or "")
            raw_identity = f"macro|{name}|{published.isoformat()}|{actual}|{estimate}"
            result.append(
                NewsItem(
                    identity=stable_hash(raw_identity, 24),
                    title=f"{country} {name}: actual {actual:g}, estimate {estimate:g}",
                    url="https://site.financialmodelingprep.com/developer/docs/stable/economics-calendar",
                    source="FMP Economic Calendar",
                    source_id="fmp_macro",
                    source_tier="structured",
                    published_at=published,
                    fetched_at=now,
                    region="global",
                    category_hint="macro",
                    actual=actual,
                    estimate=estimate,
                    unit=unit,
                    raw=row,
                )
            )
        return result
