"""Loader for the structured talent and learning dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class Record:
    """A single structured talent or learning record."""

    id: str
    type: str
    text: str
    fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Record:
        extra = {k: v for k, v in raw.items() if k not in {"id", "type", "text"}}
        return cls(id=raw["id"], type=raw["type"], text=raw["text"], fields=extra)


def load_records() -> list[Record]:
    """Load the packaged dataset of talent and learning records."""
    with resources.files("talentllm.data").joinpath("records.json").open(
        "r", encoding="utf-8"
    ) as handle:
        rows = json.load(handle)
    return [Record.from_dict(row) for row in rows]


def load_records_from_path(path: str) -> list[Record]:
    """Load records from an explicit JSON file path."""
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    return [Record.from_dict(row) for row in rows]
