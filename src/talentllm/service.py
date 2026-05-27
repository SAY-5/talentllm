"""FastAPI service exposing the grounded assistant."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .assistant import Assistant


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class CitationModel(BaseModel):
    id: str
    type: str
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    citations: list[CitationModel]


@lru_cache(maxsize=1)
def get_assistant() -> Assistant:
    return Assistant()


app = FastAPI(title="TalentLLM", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = get_assistant().ask(request.question)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        grounded=result.grounded,
        citations=[
            CitationModel(id=c.id, type=c.type, text=c.text) for c in result.citations
        ],
    )
