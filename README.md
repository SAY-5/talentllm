# TalentLLM

TalentLLM is an NLP assistant that answers natural-language questions over a
structured talent and learning dataset. It retrieves the records relevant to a
question, composes an answer strictly from those records, and returns the source
records it used as citations. Every answer is grounded at the record level: the
assistant only asserts facts that appear in the cited records, and when no record
supports a question it declines instead of inventing an answer.

## What it does

- Loads a structured dataset of talent and learning records: employees, skills,
  courses, completions, and certifications.
- Indexes the records with a deterministic local embedding behind a provider seam
  and retrieves the records relevant to a question by cosine similarity.
- Composes a grounded answer from the retrieved records through the same provider
  seam, and returns the cited source records alongside the answer.
- Exposes the assistant as a FastAPI service.

## How this differs

This repository is one of three related projects with deliberately different
designs:

- **talentllm** (this repo) answers questions by retrieving records and citing
  the source records that support the answer. Grounding is at the record level:
  the response carries the exact records it used.
- **insightllm** translates a natural-language question into a query over tabular
  data and runs it. It produces a computed result rather than retrieved records.
  In short, talentllm retrieves and cites records, while insightllm translates a
  question into a query.
- **talentagent** matches candidates to roles rather than answering questions.

## Provider seam

The retrieval index and the answer composer depend on a small provider interface
with two operations: embed text into a vector, and compose an answer from a set
of records. The default provider is fully deterministic and offline, so tests and
continuous integration run without network access. A hosted model provider can be
substituted behind the same interface without changing the rest of the code.

## Grounding guard

The assistant enforces two rules:

1. The composed answer may only assert facts present in the retrieved records.
   A guard checks that every content token in the answer comes from a cited
   record; if not, the answer is rejected.
2. When no retrieved record clears the relevance threshold, the assistant returns
   a no-answer response with no citations rather than fabricating an answer.

## Phrasing robustness

Questions can be asked in different words. The query embedding normalizes common
rewordings onto the vocabulary used in the records, for example treating
"credential" and "certified" as "certification". Several phrasings of the same
question therefore surface the same supporting records above the relevance
threshold, which the phrasing tests assert.

## Usage

```bash
pip install -e ".[dev]"
```

```python
from talentllm import Assistant

assistant = Assistant()
result = assistant.ask("what certification does Noah Kim hold")
print(result.answer)
for citation in result.citations:
    print(citation.id, citation.text)
```

Run the service:

```bash
uvicorn talentllm.service:app --reload
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "which courses cover Spark"}'
```

## Development

```bash
ruff check .
mypy
pytest
```

## Benchmark

`bench/run_bench.py` generates a deterministic dataset of roughly 960 records and
measures retrieval quality and latency. On the most recent run over 367 queries
the assistant reached recall@3 of 0.978 with a mean latency of 11.6 ms per query.
CI runs the same benchmark as a smoke gate and fails if recall@3 drops below 0.30.

```bash
python bench/run_bench.py
```

## Deployment

The intended target is AWS. See [deploy/README.md](deploy/README.md) for an
honest description of the container image and the AWS resources required. No live
deployment is included in this repository.

## License

MIT. See [LICENSE](LICENSE).
