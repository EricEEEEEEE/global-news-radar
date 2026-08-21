"""Optional semantic backstop for story clustering.

Token overlap cannot see that "6.1 quake off Hokkaido" and "Tsunami advisory
lifted for northern Japan" are one story: they share no content word. A static
sentence embedding can, and does it without a network call — model2vec
distills a transformer into a lookup table, so encoding a headline is a mean
of word vectors, microseconds on CPU, no GPU and no API.

Everything here is optional and fail-open. The package may be absent, the
model may fail to download, encoding may raise; each case disables the
embedder and leaves clustering exactly as it was. The radar must never go
quiet because an enhancement broke.
"""

from __future__ import annotations

import logging
import math
from typing import Any

LOGGER = logging.getLogger(__name__)

# A static model is a read-only lookup table, so loading the same name twice
# buys nothing and costs a few seconds of disk and, on a cold cache, a
# download. One entry per model name for the life of the process.
_MODELS: dict[str, Any] = {}


class TitleEmbedder:
    """Cosine similarity between headlines, or nothing at all."""

    def __init__(
        self,
        *,
        enabled: bool,
        model_name: str,
        threshold: float,
        max_cache: int = 4096,
    ):
        self.threshold = float(threshold)
        self.model_name = str(model_name or "")
        self._max_cache = max(64, int(max_cache))
        self._cache: dict[str, tuple[float, ...] | None] = {}
        self._model: Any = None
        self._broken = False
        if enabled and self.model_name:
            self._model = self._load()

    def _load(self) -> Any:
        if self.model_name in _MODELS:
            return _MODELS[self.model_name]
        model = self._load_uncached()
        _MODELS[self.model_name] = model
        return model

    def _load_uncached(self) -> Any:
        try:
            from model2vec import StaticModel  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            LOGGER.info("embedder_unavailable reason=import error=%s", exc)
            return None
        try:
            model = StaticModel.from_pretrained(self.model_name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "embedder_unavailable reason=load model=%s error=%s",
                self.model_name,
                exc,
            )
            return None
        LOGGER.info("embedder_ready model=%s", self.model_name)
        return model

    @property
    def available(self) -> bool:
        return self._model is not None and not self._broken

    def encode(self, text: str) -> tuple[float, ...] | None:
        """Unit-length vector for one headline, or None if that is impossible."""
        if not self.available:
            return None
        key = " ".join(str(text or "").split()).lower()
        if not key:
            return None
        if key in self._cache:
            return self._cache[key]
        try:
            raw = self._model.encode([key])[0]
            values = [float(value) for value in raw]
        except Exception as exc:  # noqa: BLE001
            # One failure disables the backstop for the process rather than
            # raising on every subsequent headline.
            self._broken = True
            LOGGER.warning("embedder_failed error=%s", exc)
            return None
        norm = math.sqrt(sum(value * value for value in values))
        vector = tuple(value / norm for value in values) if norm else None
        if len(self._cache) >= self._max_cache:
            self._cache.clear()
        self._cache[key] = vector
        return vector

    def similarity(self, left: str, right: str) -> float | None:
        first = self.encode(left)
        second = self.encode(right)
        if first is None or second is None or len(first) != len(second):
            return None
        return sum(a * b for a, b in zip(first, second, strict=True))
