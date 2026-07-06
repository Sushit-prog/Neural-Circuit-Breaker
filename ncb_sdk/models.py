"""Data models for NCB SDK responses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    """Result of a content safety check against the NCB backend.

    Attributes:
        allowed: Whether the text was allowed through (True) or blocked (False).
        circuit_state: Current state of the circuit breaker (e.g. "closed", "open").
        detail: Optional human-readable explanation of why the request was blocked.
        detector: Which detector flagged the content, if any ("fast_filter" or "deep_classifier").
        fallback_message: Optional safe fallback message the backend provides when
            blocking a request, so callers have something usable to show the user.
    """

    allowed: bool
    circuit_state: str
    detail: str | None = None
    detector: str | None = None
    fallback_message: str | None = None
