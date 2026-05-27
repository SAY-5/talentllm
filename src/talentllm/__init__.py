"""TalentLLM: a grounded question-answering assistant over talent and learning records."""

from .assistant import Answer, Assistant, Citation
from .dataset import Record, load_records
from .retrieval import RecordIndex, Retrieved

__all__ = [
    "Answer",
    "Assistant",
    "Citation",
    "Record",
    "RecordIndex",
    "Retrieved",
    "load_records",
]

__version__ = "0.1.0"
