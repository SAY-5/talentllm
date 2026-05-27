"""Embedding index and retrieval over the structured records."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .dataset import Record
from .provider import HashingProvider, Provider


@dataclass(frozen=True)
class Retrieved:
    """A record paired with its relevance score for a query."""

    record: Record
    score: float


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class RecordIndex:
    """In-memory vector index that retrieves records by relevance."""

    def __init__(self, records: list[Record], provider: Provider | None = None) -> None:
        self.provider = provider or HashingProvider()
        self.records = records
        fit = getattr(self.provider, "fit", None)
        if callable(fit):
            fit([record.text for record in records])
        self._vectors = [self.provider.embed(record.text) for record in records]

    def search(self, query: str, top_k: int = 5) -> list[Retrieved]:
        """Return up to top_k records ranked by descending relevance."""
        query_vec = self.provider.embed(query)
        scored = [
            Retrieved(record=record, score=_cosine(query_vec, vector))
            for record, vector in zip(self.records, self._vectors, strict=True)
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
