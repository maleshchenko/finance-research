"""Structured answer schema and JSON serialization helpers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    classification: Literal["REPORTED", "CALCULATED", "INTERPRETATION", "WEB SOURCE"]
    citations: list[str] = Field(min_length=1)


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    claims: list[AnswerClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def parse_answer(answer: str) -> StructuredAnswer:
    return StructuredAnswer.model_validate_json(answer)


def serialize_answer(value: object) -> str:
    if isinstance(value, str):
        parsed = parse_answer(value)
    else:
        parsed = StructuredAnswer.model_validate(value)
    return parsed.model_dump_json()
