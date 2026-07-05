"""Unit tests for FallbackRouter — static and model strategies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.routing.fallback_router import FallbackRouter
from app.routing.schemas import FallbackResponse


# ---------------------------------------------------------------------------
# Static strategy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_static_returns_configured_message() -> None:
    router = FallbackRouter(
        strategy="static",
        static_message="Safe default message",
    )
    result = await router.get_fallback(
        circuit_id="test",
        original_text="malicious input",
        block_reason="matched pattern",
    )
    assert result.message == "Safe default message"
    assert result.strategy == "static"
    assert result.fallback_triggered is True


@pytest.mark.asyncio
async def test_static_ignores_model_config() -> None:
    """Static strategy works even if model fields are populated."""
    router = FallbackRouter(
        strategy="static",
        static_message="Blocked",
        model_provider="groq",
        model_name="some-model",
        model_api_key="key",
    )
    result = await router.get_fallback("c1", "text", "reason")
    assert result.strategy == "static"
    assert result.message == "Blocked"


# ---------------------------------------------------------------------------
# Model strategy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_calls_api_and_returns_response() -> None:
    router = FallbackRouter(
        strategy="model",
        static_message="fallback",
        model_provider="groq",
        model_name="llama-3.1-8b-instant",
        model_api_key="test-key",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "I cannot help with that request."}}]
    }

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.routing.fallback_router.httpx.AsyncClient", return_value=mock_client):
        result = await router.get_fallback("c1", "bad text", "injection detected")

    assert result.strategy == "model"
    assert "cannot help" in result.message
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_model_falls_back_to_static_on_api_error() -> None:
    """API failure -> silently use static fallback."""
    router = FallbackRouter(
        strategy="model",
        static_message="Static fallback message",
        model_provider="groq",
        model_name="llama-3.1-8b-instant",
        model_api_key="test-key",
    )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.routing.fallback_router.httpx.AsyncClient", return_value=mock_client):
        result = await router.get_fallback("c1", "text", "reason")

    assert result.strategy == "static"
    assert result.message == "Static fallback message"


@pytest.mark.asyncio
async def test_model_falls_back_on_empty_response() -> None:
    """Empty model content -> use static fallback."""
    router = FallbackRouter(
        strategy="model",
        static_message="Static fallback",
        model_provider="groq",
        model_name="llama-3.1-8b-instant",
        model_api_key="test-key",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.routing.fallback_router.httpx.AsyncClient", return_value=mock_client):
        result = await router.get_fallback("c1", "text", "reason")

    assert result.strategy == "static"
    assert result.message == "Static fallback"


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_strategy_without_api_key_uses_static() -> None:
    """Model strategy with empty API key -> silently operates as STATIC."""
    router = FallbackRouter(
        strategy="model",
        static_message="Static fallback",
        model_provider="groq",
        model_name="llama-3.1-8b-instant",
        model_api_key="",  # missing
    )
    assert router._strategy == "static"
    result = await router.get_fallback("c1", "text", "reason")
    assert result.strategy == "static"


# ---------------------------------------------------------------------------
# Interface completeness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_original_text_not_forwarded_to_model() -> None:
    """Verify original_text is never passed to the model API call."""
    router = FallbackRouter(
        strategy="model",
        static_message="fallback",
        model_provider="groq",
        model_name="test-model",
        model_api_key="key",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": "Blocked."}}]}

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("app.routing.fallback_router.httpx.AsyncClient", return_value=mock_client):
        await router.get_fallback("c1", "INJECTED malicious text", "injection")

    # Inspect the payload sent to the API
    call_args = mock_client.post.call_args
    payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
    user_message = payload["messages"][1]["content"]
    assert "INJECTED malicious text" not in user_message
    assert "injection" in user_message
