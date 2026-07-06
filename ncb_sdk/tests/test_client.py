"""Tests for the NCB SDK client."""

from __future__ import annotations

import pytest
import httpx

from ncb_sdk import (
    NeuralCircuitBreaker,
    CheckResult,
    NCBBlockedError,
    NCBConnectionError,
    NCBTimeoutError,
)


# ---------------------------------------------------------------------------
# Helpers — mock transport for httpx
# ---------------------------------------------------------------------------

def _json_transport(data: dict, status_code: int = 200) -> httpx.MockTransport:
    """Return an httpx MockTransport that responds with a JSON body."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=data)
    return httpx.MockTransport(handler)


def _pass_response() -> dict:
    return {
        "status": "pass",
        "circuit_state": "closed",
        "detail": None,
        "detector": None,
        "fallback": None,
    }


def _blocked_response() -> dict:
    return {
        "status": "blocked",
        "circuit_state": "open",
        "detail": "Circuit is OPEN — requests blocked during cooldown",
        "detector": "fast_filter",
        "fallback": {
            "message": "I'm sorry, I can't help with that.",
            "strategy": "static",
            "fallback_triggered": True,
        },
    }


def _error_transport(exc_cls: type[httpx.TransportError]) -> httpx.MockTransport:
    """Return a MockTransport that always raises the given exception."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc_cls("mock error")
    return httpx.MockTransport(handler)


def _make_client(transport: httpx.MockTransport) -> NeuralCircuitBreaker:
    """Create a NeuralCircuitBreaker wired to a mock transport."""
    ncb = NeuralCircuitBreaker(
        base_url="http://testserver",
        circuit_id="test-circuit",
    )
    ncb._client = httpx.Client(transport=transport, timeout=5.0)
    return ncb


# ---------------------------------------------------------------------------
# check() tests
# ---------------------------------------------------------------------------

class TestCheck:
    def test_allowed_response(self) -> None:
        ncb = _make_client(_json_transport(_pass_response()))
        result = ncb.check("safe text")
        assert result.allowed is True
        assert result.circuit_state == "closed"
        assert result.detail is None
        assert result.detector is None
        assert result.fallback_message is None

    def test_blocked_response(self) -> None:
        ncb = _make_client(_json_transport(_blocked_response()))
        result = ncb.check("unsafe text")
        assert result.allowed is False
        assert result.circuit_state == "open"
        assert result.detail == "Circuit is OPEN — requests blocked during cooldown"
        assert result.detector == "fast_filter"
        assert result.fallback_message == "I'm sorry, I can't help with that."

    def test_connection_failure_raises(self) -> None:
        ncb = _make_client(_error_transport(httpx.ConnectError))
        with pytest.raises(NCBConnectionError):
            ncb.check("any text")

    def test_connection_failure_not_silent_allowed(self) -> None:
        """Verify connection failure does NOT return a default 'allowed' result."""
        ncb = _make_client(_error_transport(httpx.ConnectError))
        with pytest.raises(NCBConnectionError):
            ncb.check("any text")

    def test_timeout_raises(self) -> None:
        ncb = _make_client(_error_transport(httpx.TimeoutException))
        with pytest.raises(NCBTimeoutError):
            ncb.check("any text")

    def test_api_key_sent_in_header(self) -> None:
        """Verify the API key is included in the Authorization header."""
        captured_headers: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_headers.update(dict(request.headers))
            return httpx.Response(200, json=_pass_response())

        transport = httpx.MockTransport(handler)
        ncb = _make_client(transport)
        ncb.api_key = "test-key-123"
        ncb.check("text")
        assert captured_headers.get("authorization") == "Bearer test-key-123"


# ---------------------------------------------------------------------------
# protect() decorator tests
# ---------------------------------------------------------------------------

class TestProtect:
    def test_calls_function_when_allowed(self) -> None:
        ncb = _make_client(_json_transport(_pass_response()))

        @ncb.protect
        def my_func(text: str) -> str:
            return f"processed: {text}"

        assert my_func("hello") == "processed: hello"

    def test_raises_blocked_error_when_blocked(self) -> None:
        ncb = _make_client(_json_transport(_blocked_response()))
        called = False

        @ncb.protect
        def my_func(text: str) -> str:
            nonlocal called
            called = True
            return "should not reach"

        with pytest.raises(NCBBlockedError) as exc_info:
            my_func("bad text")

        assert called is False
        assert exc_info.value.result.allowed is False
        assert exc_info.value.result.fallback_message == "I'm sorry, I can't help with that."

    def test_propagates_connection_error(self) -> None:
        """protect() must NOT swallow backend-down errors."""
        ncb = _make_client(_error_transport(httpx.ConnectError))

        @ncb.protect
        def my_func(text: str) -> str:
            return "should not reach"

        with pytest.raises(NCBConnectionError):
            my_func("text")

    def test_extracts_text_from_first_positional_arg(self) -> None:
        ncb = _make_client(_json_transport(_pass_response()))

        @ncb.protect
        def my_func(context: int, text: str) -> str:
            return text

        # First positional arg is not a string — should fall through to kwargs
        # But no 'text' or 'prompt' kwarg provided either, so raises ValueError
        with pytest.raises(ValueError, match="Could not extract text"):
            my_func(42, "actual text")

    def test_extracts_text_from_prompt_kwarg(self) -> None:
        ncb = _make_client(_json_transport(_pass_response()))

        @ncb.protect
        def my_func(**kwargs: str) -> str:
            return kwargs.get("prompt", "")

        result = my_func(prompt="hello world")
        assert result == "hello world"

    def test_extracts_text_from_text_kwarg(self) -> None:
        ncb = _make_client(_json_transport(_pass_response()))

        @ncb.protect
        def my_func(**kwargs: str) -> str:
            return kwargs.get("text", "")

        result = my_func(text="hello world")
        assert result == "hello world"

    def test_raises_valueerror_when_no_text_found(self) -> None:
        ncb = _make_client(_json_transport(_pass_response()))

        @ncb.protect
        def my_func(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="Could not extract text"):
            my_func(42)

    def test_preserves_function_metadata(self) -> None:
        ncb = _make_client(_json_transport(_pass_response()))

        @ncb.protect
        def my_func(text: str) -> str:
            """My docstring."""
            return text

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_closes_client(self) -> None:
        ncb = NeuralCircuitBreaker(
            base_url="http://testserver",
            circuit_id="test",
        )
        with ncb:
            pass
        # After context exit, client should be closed
        assert ncb._client.is_closed
