"""Contract tests for retrieval ranking and answer grounding."""

from talentllm import Assistant, RecordIndex, load_records
from talentllm.assistant import NO_ANSWER
from talentllm.provider import tokenize


def test_retrieval_ranks_by_relevance():
    index = RecordIndex(load_records())
    hits = index.search("Kubernetes operations course", top_k=4)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert "Kubernetes" in hits[0].record.text


def test_answer_cites_only_retrieved_records():
    assistant = Assistant()
    result = assistant.ask("which courses cover Spark")
    cited_ids = {c.id for c in result.citations}
    retrieved_ids = {hit.record.id for hit in assistant._relevant(result.question)}
    assert cited_ids
    assert cited_ids <= retrieved_ids


def test_every_answer_fact_traces_to_a_citation():
    assistant = Assistant()
    result = assistant.ask("what certification does Noah Kim hold")
    cited_tokens: set[str] = set()
    for citation in result.citations:
        cited_tokens.update(tokenize(citation.text))
    content = [t for t in tokenize(result.answer) if t not in {"and", "the", "a"}]
    assert all(token in cited_tokens for token in content)


def test_out_of_scope_question_returns_no_answer():
    assistant = Assistant()
    result = assistant.ask("what is the boiling point of helium on Jupiter")
    assert result.answer == NO_ANSWER
    assert result.citations == []
    assert result.grounded is True


def test_golden_answer_for_seeded_query():
    assistant = Assistant()
    result = assistant.ask("what certification does Noah Kim hold")
    assert result.citations[0].id == "cert-301"
    assert (
        "Noah Kim holds the Certified Kubernetes Administrator certification"
        in result.answer
    )
