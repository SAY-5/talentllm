"""Citation precision and anti-hallucination tests.

These assert the two properties that distinguish this assistant: the records it
cites are the ones that actually support the question, and it refuses to answer
when the retrieved records do not support a claim.
"""

import pytest

from talentllm import Assistant
from talentllm.assistant import NO_ANSWER, Answer
from talentllm.dataset import Record
from talentllm.provider import HashingProvider, tokenize

# Questions paired with the record id that genuinely supports the answer.
SUPPORTED = [
    ("what certification does Noah Kim hold", "cert-301"),
    ("which course did Sofia Alvarez complete", "comp-203"),
    ("what is Liam Carter's certification", "cert-302"),
    ("which course covers accessible frontend development", "course-103"),
]


@pytest.mark.parametrize("question,expected_id", SUPPORTED)
def test_cited_record_is_a_correct_supporting_record(question, expected_id):
    assistant = Assistant()
    result = assistant.ask(question)
    cited_ids = {c.id for c in result.citations}
    assert expected_id in cited_ids


def test_top_citation_precision_over_supported_set():
    assistant = Assistant()
    correct = 0
    for question, expected_id in SUPPORTED:
        result = assistant.ask(question)
        if result.citations and result.citations[0].id == expected_id:
            correct += 1
    assert correct / len(SUPPORTED) >= 0.75


def test_refuses_when_no_record_supports_the_question():
    assistant = Assistant()
    result = assistant.ask("what is the population of Mars in the year 3000")
    assert result.answer == NO_ANSWER
    assert result.citations == []


class _FabricatingProvider:
    """A provider that invents a fact not present in any record.

    It reuses a real embedding so retrieval still surfaces records, but its
    composition asserts a salary that appears in no record.
    """

    def __init__(self) -> None:
        self._embedder = HashingProvider()

    def fit(self, corpus: list[str]) -> None:
        self._embedder.fit(corpus)

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed(text)

    def compose(self, question: str, records: list[Record]) -> str:
        return "Noah Kim earns a salary of 999999 dollars per year."


def test_guard_rejects_an_invented_fact():
    # The provider tries to assert a salary that appears in no record. The guard
    # must reject it and fall back to the no-answer response.
    assistant = Assistant(provider=_FabricatingProvider())
    result = assistant.ask("what certification does Noah Kim hold")
    assert result.answer == NO_ANSWER
    assert result.citations == []


def test_answer_tokens_are_a_subset_of_cited_record_tokens():
    assistant = Assistant()
    result: Answer = assistant.ask("what certification does Noah Kim hold")
    cited_tokens: set[str] = set()
    for citation in result.citations:
        cited_tokens.update(tokenize(citation.text))
    answer_tokens = set(tokenize(result.answer))
    leaked = answer_tokens - cited_tokens
    # Anything left over must be a connective word, never a fact.
    assert leaked <= {"no", "matching", "records", "the", "and"}
