from fastapi.testclient import TestClient

from talentllm import Assistant, RecordIndex, load_records
from talentllm.service import app


def test_dataset_loads_records():
    records = load_records()
    assert len(records) > 10
    assert all(r.id and r.text for r in records)


def test_retrieval_returns_records():
    records = load_records()
    index = RecordIndex(records)
    hits = index.search("who works with Kubernetes", top_k=3)
    assert len(hits) == 3
    assert any("Kubernetes" in hit.record.text for hit in hits)


def test_assistant_answers_with_citations():
    assistant = Assistant()
    answer = assistant.ask("what certification does Noah Kim hold")
    assert answer.citations
    assert any("Noah Kim" in c.text for c in answer.citations)
    assert "Noah Kim" in answer.answer


def test_service_ask_endpoint():
    client = TestClient(app)
    response = client.post("/ask", json={"question": "which courses cover Spark"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"]
    assert body["grounded"] is True


def test_service_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
