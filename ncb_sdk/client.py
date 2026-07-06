"""NeuralCircuitBreaker client — sync SDK for the NCB API."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import httpx

from ncb_sdk.exceptions import NCBBlockedError, NCBConnectionError, NCBTimeoutError
from ncb_sdk.models import CheckResult

F = TypeVar("F", bound=Callable[..., Any])


class NeuralCircuitBreaker:
    """Client for the Neural Circuit Breaker safety API.

    Provides two ways to protect LLM calls:

    1. Explicit check::

        result = ncb.check("some user text")
        if not result.allowed:
            # handle block

    2. Decorator::

        @ncb.protect
        def call_llm(prompt: str) -> str:
            return llm.generate(prompt)

        # Returns the LLM result if allowed, raises NCBBlockedError if not.

    **Fail-closed design**: If the backend is unreachable or times out, the SDK
    raises an exception rather than silently returning "allowed". This is
    deliberate — a broken connection to the safety backend must never result in
    unprotected calls. The caller must handle these network errors explicitly.
    """

    def __init__(
        self,
        base_url: str,
        circuit_id: str,
        api_key: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        """Initialize the NCB client.

        Args:
            base_url: Base URL of the NCB backend (e.g. "http://localhost:8000").
            circuit_id: Circuit breaker identifier to check against.
            api_key: Optional API key for authentication. Forward-compatible
                with future auth enforcement on the backend.
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.circuit_id = circuit_id
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> NeuralCircuitBreaker:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def check(self, text: str) -> CheckResult:
        """Check whether *text* is allowed through the circuit breaker.

        Posts the text to the NCB backend and returns a structured result.

        **Safety note**: On network failure or timeout, this method raises
        :class:`NCBConnectionError` or :class:`NCBTimeoutError` — it does
        *not* return a default "allowed" result. A broken connection to the
        safety backend must be a loud, explicit failure the caller has to
        handle, never a silent bypass.

        Args:
            text: The text content to evaluate.

        Returns:
            A CheckResult with the safety evaluation.

        Raises:
            NCBConnectionError: Backend is unreachable.
            NCBTimeoutError: Request timed out.
        """
        url = f"{self.base_url}/v1/check"
        payload = {"text": text, "circuit_id": self.circuit_id}

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = self._client.post(url, json=payload, headers=headers)
        except httpx.ConnectError as exc:
            raise NCBConnectionError(
                f"Cannot reach NCB backend at {self.base_url}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise NCBTimeoutError(
                f"NCB backend timed out after {self._client.timeout}"
            ) from exc

        response.raise_for_status()
        data = response.json()

        return CheckResult(
            allowed=data["status"] == "pass",
            circuit_state=data["circuit_state"],
            detail=data.get("detail"),
            detector=data.get("detector"),
            fallback_message=data.get("fallback", {}).get("message") if data.get("fallback") else None,
        )

    def protect(self, func: F) -> F:
        """Decorator that gates a function behind an NCB safety check.

        Assumes the first positional argument, or a ``text``/``prompt`` keyword
        argument, is the user-provided text to check.

        **Fail-closed behavior**: If the backend is unreachable, the exception
        propagates — the wrapped function is *not* called. This prevents
        silently disabling protection on infrastructure failure.

        If the backend blocks the request, raises :class:`NCBBlockedError`
        instead of calling the wrapped function. The error carries the full
        :class:`CheckResult` so the caller can inspect the block reason.

        Usage::

            @ncb.protect
            def generate(prompt: str) -> str:
                return llm.complete(prompt)

            # Raises NCBBlockedError if blocked, NCBConnectionError if backend down
            response = generate("Hello, world!")
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            text = _extract_text(args, kwargs)
            result = self.check(text)

            if not result.allowed:
                raise NCBBlockedError(
                    f"Request blocked by NCB: {result.detail or 'content not allowed'}",
                    result=result,
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


def _extract_text(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Extract the text to check from function arguments.

    Checks positional args first (takes the first one), then looks for
    ``text`` or ``prompt`` keyword arguments.

    Raises ValueError if no suitable text argument is found.
    """
    if args:
        val = args[0]
        if isinstance(val, str):
            return val

    for key in ("text", "prompt"):
        if key in kwargs and isinstance(kwargs[key], str):
            return kwargs[key]

    raise ValueError(
        "Could not extract text to check. Pass text as the first positional "
        "argument, or as a 'text' or 'prompt' keyword argument."
    )
