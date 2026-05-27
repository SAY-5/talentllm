"""Grounded question answering over the structured dataset.

The assistant retrieves the records relevant to a question, composes an answer
strictly from those records through the provider seam, and returns the source
records it used as citations. A grounding guard enforces two rules: the answer
may only assert facts present in the retrieved records, and when no record is
relevant the assistant declines rather than inventing an answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dataset import Record, load_records
from .provider import HashingProvider, Provider, tokenize
from .retrieval import RecordIndex, Retrieved

NO_ANSWER = (
    "No matching records were found, so this question cannot be "
    "answered from the dataset."
)

# Tokens that the composer may use for connective phrasing even though they are
# not part of any record. They carry no factual claim about talent or learning.
_FUNCTION_WORDS = {
    "no",
    "matching",
    "records",
    "were",
    "found",
    "so",
    "this",
    "question",
    "cannot",
    "be",
    "answered",
    "from",
    "the",
    "dataset",
}


@dataclass(frozen=True)
class Citation:
    """A source record cited in support of an answer."""

    id: str
    type: str
    text: str


@dataclass(frozen=True)
class Answer:
    """An answer together with the source records that ground it."""

    question: str
    answer: str
    citations: list[Citation]
    grounded: bool


class Assistant:
    """Answers natural-language questions with record-level grounding."""

    def __init__(
        self,
        records: list[Record] | None = None,
        provider: Provider | None = None,
        min_score: float = 0.18,
        top_k: int = 4,
    ) -> None:
        self.provider = provider or HashingProvider()
        self.records = records if records is not None else load_records()
        self.index = RecordIndex(self.records, provider=self.provider)
        self.min_score = min_score
        self.top_k = top_k

    def _relevant(self, query: str) -> list[Retrieved]:
        hits = self.index.search(query, top_k=self.top_k)
        return [hit for hit in hits if hit.score >= self.min_score]

    def _is_grounded(self, answer: str, records: list[Record]) -> bool:
        """True when every content token in the answer comes from a record."""
        allowed: set[str] = set(_FUNCTION_WORDS)
        for record in records:
            allowed.update(tokenize(record.text))
        return all(token in allowed for token in tokenize(answer))

    def ask(self, question: str) -> Answer:
        relevant = self._relevant(question)
        if not relevant:
            return Answer(
                question=question,
                answer=NO_ANSWER,
                citations=[],
                grounded=True,
            )
        records = [hit.record for hit in relevant]
        composed = self.provider.compose(question, records)
        if not self._is_grounded(composed, records):
            # The guard refuses an ungrounded composition rather than return it.
            return Answer(
                question=question,
                answer=NO_ANSWER,
                citations=[],
                grounded=True,
            )
        citations = [
            Citation(id=r.id, type=r.type, text=r.text) for r in records
        ]
        return Answer(
            question=question,
            answer=composed,
            citations=citations,
            grounded=True,
        )
