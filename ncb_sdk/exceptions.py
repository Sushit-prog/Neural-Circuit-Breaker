"""Exception hierarchy for the NCB SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ncb_sdk.models import CheckResult


class NCBError(Exception):
    """Base exception for all NCB SDK errors."""


class NCBConnectionError(NCBError):
    """Raised when the NCB backend is unreachable.

    This is a safety-critical failure mode. When the backend cannot be reached,
    the SDK raises rather than silently allowing requests through. A broken
    connection to the safety backend must never result in unprotected calls.
    """


class NCBTimeoutError(NCBError):
    """Raised when a request to the NCB backend times out.

    Same fail-closed principle as NCBConnectionError: timeouts are treated as
    safety failures, not as silent pass-throughs.
    """


class NCBBlockedError(NCBError):
    """Raised when the NCB backend blocks a request.

    Carries the full CheckResult so the caller can inspect the block reason,
    circuit state, and any fallback message.
    """

    def __init__(self, message: str, result: CheckResult) -> None:
        super().__init__(message)
        self.result = result
