from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class NewsItem:
    identity: str
    title: str
    url: str
    source: str
    source_id: str
    source_tier: str
    published_at: datetime
    fetched_at: datetime
    summary: str = ""
    region: str = "global"
    category_hint: str = ""
    actual: float | None = None
    estimate: float | None = None
    unit: str = ""
    symbol: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["published_at"] = self.published_at.isoformat()
        data["fetched_at"] = self.fetched_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NewsItem:
        payload = dict(data)
        payload["published_at"] = datetime.fromisoformat(payload["published_at"])
        payload["fetched_at"] = datetime.fromisoformat(payload["fetched_at"])
        return cls(**payload)


@dataclass(slots=True)
class Assessment:
    level: str
    category: str
    event_key: str
    material_hash: str
    reason: str
    requires_corroboration: bool
    region: str
    action: str
    entities: tuple[str, ...]
    topic_anchor: str = ""
    lineage_day: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entities"] = list(self.entities)
        return data


@dataclass(slots=True)
class AlertEvent:
    assessment: Assessment
    items: list[NewsItem]

    @property
    def primary(self) -> NewsItem:
        return sorted(
            self.items,
            key=lambda item: (
                item.source_tier != "primary",
                -item.published_at.timestamp(),
            ),
        )[0]

    @property
    def sources(self) -> list[str]:
        return sorted({item.source for item in self.items})


@dataclass(slots=True)
class RenderedMessage:
    level: str
    html: str
    plain_text: str
    event_keys: list[str]
    content_hash: str
    evidence: list[dict[str, Any]]
    created_at: datetime
    visual_spec: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RenderedMessage:
        payload = dict(data)
        payload["created_at"] = datetime.fromisoformat(payload["created_at"])
        return cls(**payload)
