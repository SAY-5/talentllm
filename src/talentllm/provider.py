"""Model provider seam.

The retrieval index and answer composer depend on a provider interface for two
operations: embedding text into a vector, and composing an answer from a set of
retrieved records. The default provider is fully deterministic and offline so
that tests and continuous integration are hermetic. A real hosted provider can
be substituted behind the same interface without touching the rest of the code.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from .dataset import Record

_TOKEN = re.compile(r"[a-z0-9]+")

# Terms that carry no retrieval signal. Dropping them keeps cosine similarity
# focused on the content words that distinguish one record from another.
_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "he",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "the",
    "to",
    "who",
    "what",
    "which",
    "with",
    "does",
    "do",
    "has",
    "have",
    "hold",
    "holds",
    "works",
    "work",
}


# Maps common rewordings onto the vocabulary used in the records so that a
# question phrased with a synonym still overlaps the right records. This keeps
# retrieval robust to varied phrasings without any external model.
_SYNONYMS = {
    "credential": "certification",
    "credentials": "certification",
    "certified": "certification",
    "cert": "certification",
    "earned": "holds",
    "earn": "holds",
    "obtained": "holds",
    "class": "course",
    "classes": "course",
    "training": "course",
    "finished": "completed",
    "passed": "completed",
    "team": "department",
    "group": "department",
    "based": "located",
    "location": "located",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens used by both embedding and grounding checks."""
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stop words removed and synonyms normalized.

    Used to build the embedding vocabulary and to embed queries, so a reworded
    question maps onto the same terms as the records it should retrieve.
    """
    result = []
    for token in tokenize(text):
        if token in _STOP:
            continue
        result.append(_SYNONYMS.get(token, token))
    return result


class Provider(Protocol):
    """Interface every model provider must satisfy."""

    def embed(self, text: str) -> list[float]:
        """Return a fixed-length vector for the given text."""

    def compose(self, question: str, records: list[Record]) -> str:
        """Return an answer composed only from the supplied records."""


class HashingProvider:
    """Deterministic offline provider.

    Embeddings are term-frequency vectors over a fixed vocabulary fit on the
    record corpus, with inverse-document-frequency weighting. The vocabulary is
    derived deterministically from the records, so embeddings are stable across
    runs and machines and similarity reflects genuine term overlap rather than
    hash collisions. Answer composition is extractive: it stitches together the
    text of the supplied records and never introduces tokens that are not present
    in them, which keeps every answer grounded by construction.
    """

    def __init__(self, corpus: list[str] | None = None) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []
        if corpus:
            self.fit(corpus)

    def fit(self, corpus: list[str]) -> None:
        """Build the vocabulary and idf weights from a corpus of documents."""
        doc_freq: Counter[str] = Counter()
        vocab: dict[str, int] = {}
        for document in corpus:
            seen = set(content_tokens(document))
            for token in seen:
                doc_freq[token] += 1
                if token not in vocab:
                    vocab[token] = len(vocab)
        self._vocab = vocab
        n_docs = max(len(corpus), 1)
        idf = [0.0] * len(vocab)
        for token, index in vocab.items():
            idf[index] = math.log((1 + n_docs) / (1 + doc_freq[token])) + 1.0
        self._idf = idf

    def embed(self, text: str) -> list[float]:
        if not self._vocab:
            # Without a fitted corpus, fall back to fitting on the single text so
            # the provider still returns a usable vector.
            self.fit([text])
        vec = [0.0] * len(self._vocab)
        counts = Counter(content_tokens(text))
        for token, count in counts.items():
            index = self._vocab.get(token)
            if index is not None:
                vec[index] = count * self._idf[index]
        norm = math.sqrt(sum(component * component for component in vec))
        if norm == 0.0:
            return vec
        return [component / norm for component in vec]

    def compose(self, question: str, records: list[Record]) -> str:
        if not records:
            return (
                "No matching records were found, so this question cannot be "
                "answered from the dataset."
            )
        lines = [record.text for record in records]
        return " ".join(lines)
