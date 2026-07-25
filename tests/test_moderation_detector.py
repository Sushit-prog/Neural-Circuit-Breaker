"""Unit tests for ModerationDetector — uses httpx.MockTransport, no live API calls."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.detectors.moderation_detector import ModerationDetector


def _make_settings(**overrides: Any) -> Settings:
    defaults = dict(
        REDIS_URL="redis://localhost:6379/0",
        CIRCUIT_FAILURE_THRESHOLD=5,
        CIRCUIT_WINDOW_SECONDS=60,
        CIRCUIT_COOLDOWN_SECONDS=30,
        MODERATION_DETECTOR_ENABLED=True,
        MODERATION_API_KEY="fake-key",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _client_for(data: dict, status_code: int = 200) -> httpx.AsyncClient:
    """Build an AsyncClient wired to a MockTransport returning a fixed JSON body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=data)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _moderation_body(flagged: bool, category_scores: dict[str, float]) -> dict:
    return {"results": [{"flagged": flagged, "category_scores": category_scores}]}


@pytest.mark.asyncio
async def test_flagged_content_is_unsafe() -> None:
    """API returns flagged=True -> is_safe=False, reason names top category."""
    cfg = _make_settings()
    client = _client_for(_moderation_body(True, {"harassment": 0.91, "violence": 0.12}))
    detector = ModerationDetector(config=cfg, client=client)

    result = await detector.check("something flagged")

    assert result.is_safe is False
    assert result.score == pytest.approx(0.91, abs=0.001)
    assert "harassment" in result.reason.lower()


@pytest.mark.asyncio
async def test_safe_content_passes() -> None:
    """API returns flagged=False -> is_safe=True."""
    cfg = _make_settings()
    client = _client_for(_moderation_body(False, {"harassment": 0.01, "violence": 0.02}))
    detector = ModerationDetector(config=cfg, client=client)

    result = await detector.check("what's the weather today?")

    assert result.is_safe is True
    assert result.reason == ""


@pytest.mark.asyncio
async def test_missing_api_key_fails_closed() -> None:
    """No API key configured -> is_ready is False, check() fails closed without a network call."""
    cfg = _make_settings(MODERATION_API_KEY="")
    detector = ModerationDetector(config=cfg)

    assert detector.is_ready is False
    result = await detector.check("anything")

    assert result.is_safe is False
    assert "no api key" in result.reason.lower()


@pytest.mark.asyncio
async def test_timeout_fails_closed() -> None:
    """Request timeout -> is_safe=False, never raises out of check()."""
    cfg = _make_settings()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    detector = ModerationDetector(config=cfg, client=client)

    result = await detector.check("anything")

    assert result.is_safe is False
    assert "timeout" in result.reason.lower()


@pytest.mark.asyncio
async def test_non_2xx_response_fails_closed() -> None:
    """A 500 from the API -> is_safe=False, exception is swallowed not raised."""
    cfg = _make_settings()
    client = _client_for({"error": "internal"}, status_code=500)
    detector = ModerationDetector(config=cfg, client=client)

    result = await detector.check("anything")

    assert result.is_safe is False


@pytest.mark.asyncio
async def test_unexpected_payload_shape_fails_closed() -> None:
    """Malformed response body (missing expected keys) -> is_safe=False, no crash."""
    cfg = _make_settings()
    client = _client_for({"unexpected": "shape"})
    detector = ModerationDetector(config=cfg, client=client)

    result = await detector.check("anything")

    assert result.is_safe is False
    assert "unexpected" in result.reason.lower()


@pytest.mark.asyncio
async def test_empty_category_scores_uses_flagged_as_score() -> None:
    """If category_scores is empty/missing, fall back to a binary score from flagged."""
    cfg = _make_settings()
    client = _client_for({"results": [{"flagged": True, "category_scores": {}}]})
    detector = ModerationDetector(config=cfg, client=client)

    result = await detector.check("anything")

    assert result.is_safe is False
    assert result.score == 1.0
