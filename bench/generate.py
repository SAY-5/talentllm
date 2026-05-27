"""Generate a large deterministic dataset for benchmarking.

The records mirror the shape of the packaged dataset but are produced by a fixed
pseudo-random sequence so the benchmark is reproducible. Each generated query has
a known supporting record, which lets the benchmark measure retrieval quality.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

FIRST = ["Ava", "Liam", "Sofia", "Noah", "Maya", "Omar", "Ines", "Theo", "Lena", "Ravi"]
LAST = ["Bennett", "Carter", "Alvarez", "Kim", "Singh", "Haddad", "Moore", "Reyes"]
SKILLS = [
    "Python",
    "Spark",
    "PyTorch",
    "Kubernetes",
    "Terraform",
    "React",
    "TypeScript",
    "SQL",
    "Go",
    "Statistics",
]
DEPARTMENTS = ["Data Platform", "Applied Research", "Infrastructure", "Product Engineering"]
LOCATIONS = ["Austin", "Seattle", "Remote", "Boston", "Denver"]
COURSES = [
    ("DATA-101", "Distributed Data Processing with Spark"),
    ("ML-201", "Applied Deep Learning"),
    ("WEB-110", "Accessible Frontend Development"),
    ("OPS-150", "Kubernetes Operations"),
    ("SQL-120", "Advanced Query Optimization"),
    ("GO-130", "Concurrent Programming in Go"),
]


def build(n_employees: int = 600, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    for i in range(n_employees):
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        dept = rng.choice(DEPARTMENTS)
        loc = rng.choice(LOCATIONS)
        skills = rng.sample(SKILLS, 3)
        eid = f"emp-{i:04d}"
        skill_text = ", ".join(skills)
        records.append(
            {
                "id": eid,
                "type": "employee",
                "text": (
                    f"{name} is a member of the {dept} department based in {loc} "
                    f"and works with {skill_text}."
                ),
            }
        )
        if rng.random() < 0.6:
            code, title = rng.choice(COURSES)
            score = rng.randint(70, 99)
            cid = f"comp-{i:04d}"
            records.append(
                {
                    "id": cid,
                    "type": "completion",
                    "text": (
                        f"{name} completed course {code} {title} with a score of {score}."
                    ),
                }
            )
    return records


def main() -> None:
    out = Path(__file__).with_name("dataset_large.json")
    records = build()
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
