"""Shared test fixtures for the Neural Circuit Breaker test suite."""

from __future__ import annotations

from typing import AsyncIterator

import fakeredis.aioredis
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.redis_client import get_redis
from app.routing.fallback_router import FallbackRouter


@pytest_asyncio.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    """Isolated fakeredis instance per test (no real Redis needed)."""
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def test_client(fake_redis: fakeredis.aioredis.FakeRedis) -> AsyncIterator[AsyncClient]:
    """Async HTTP client wired to the FastAPI app with a fake Redis backend.

    Overrides the module-level ``settings`` used by the app so that the
    circuit breaker uses short thresholds/cooldowns for fast test execution.
    """
    import app.core.config as config_mod
    import app.api.routes as routes_mod

    test_settings = Settings(
        REDIS_URL="redis://localhost:6379/0",
        CIRCUIT_FAILURE_THRESHOLD=3,
        CIRCUIT_WINDOW_SECONDS=10,
        CIRCUIT_COOLDOWN_SECONDS=1,
        DEEP_CLASSIFIER_ENABLED=False,
    )

    # Patch the settings object so routes + circuit breaker pick them up
    original = config_mod.settings
    config_mod.settings = test_settings
    routes_mod.settings = test_settings

    from app.main import app

    async def _override_get_redis() -> fakeredis.aioredis.FakeRedis:
        return fake_redis

    # Ensure fallback router is on app.state (normally set in lifespan)
    if not hasattr(app.state, "fallback_router") or app.state.fallback_router is None:
        app.state.fallback_router = FallbackRouter(
            strategy="static",
            static_message="Test fallback message",
        )

    app.dependency_overrides[get_redis] = _override_get_redis
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()

    # Restore original settings
    config_mod.settings = original
    routes_mod.settings = original
