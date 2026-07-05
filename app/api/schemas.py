"""Pydantic models for API request and response payloads."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CheckRequest(BaseModel):
    """Payload for POST /v1/check."""

    text: str = Field(..., min_length=1, max_length=10_000)
    circuit_id: str = Field(..., min_length=1, max_length=100)


class CheckResponse(BaseModel):
    """Returned after every /v1/check evaluation."""

    status: Literal["pass", "blocked"]
    circuit_state: str
    detail: str | None = None
    detector: Literal["fast_filter", "deep_classifier", None] = None


class CircuitStateResponse(BaseModel):
    """Returned by the debug state endpoint."""

    circuit_id: str
    state: str
    failure_count: int
