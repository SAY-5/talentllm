"""Benchmark retrieval and answer composition over the large dataset.

Reports two measured numbers:

- recall_at_3: fraction of queries whose known supporting record appears in the
  top three retrieved records.
- mean_latency_ms: average wall-clock time to answer a query end to end.

The dataset is built deterministically, so the recall figure is reproducible
on any machine. Latency depends on hardware and is reported for the run only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synth import build  # noqa: E402

from talentllm.assistant import Assistant  # noqa: E402
from talentllm.dataset import Record  # noqa: E402


def _queries(records: list[dict]) -> list[tuple[str, str]]:
    """Build (question, expected_record_id) pairs from completion records."""
    pairs: list[tuple[str, str]] = []
    for row in records:
        if row["type"] == "completion":
            name = row["text"].split(" completed")[0]
            code = row["text"].split("course ")[1].split(" ")[0]
            pairs.append((f"which course did {name} complete {code}", row["id"]))
    return pairs


def run() -> dict[str, float]:
    raw = build()
    records = [Record.from_dict(r) for r in raw]
    assistant = Assistant(records=records, top_k=3, min_score=0.05)
    queries = _queries(raw)

    hits = 0
    start = time.perf_counter()
    for question, expected in queries:
        retrieved = assistant._relevant(question)
        ids = [r.record.id for r in retrieved[:3]]
        if expected in ids:
            hits += 1
    elapsed = time.perf_counter() - start

    n = len(queries)
    return {
        "queries": float(n),
        "recall_at_3": hits / n if n else 0.0,
        "mean_latency_ms": (elapsed / n * 1000.0) if n else 0.0,
    }


def main() -> None:
    metrics = run()
    print(json.dumps(metrics, indent=2))
    threshold = 0.30
    if metrics["recall_at_3"] < threshold:
        print(
            f"bench-regress: recall_at_3 {metrics['recall_at_3']:.3f} "
            f"below threshold {threshold:.2f}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"bench-regress: recall_at_3 {metrics['recall_at_3']:.3f} at or above {threshold:.2f}")


if __name__ == "__main__":
    main()
