"""Unit tests for CircuitBreaker state transitions."""

from __future__ import annotations

import asyncio

import pytest
import fakeredis.aioredis

from app.core.circuit_breaker import CircuitBreaker, CircuitState
from app.core.config import Settings


@pytest.mark.asyncio
async def test_defaults_to_closed(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """New circuit_id with no prior state should be CLOSED."""
    cfg = Settings(CIRCUIT_FAILURE_THRESHOLD=5, CIRCUIT_WINDOW_SECONDS=60, CIRCUIT_COOLDOWN_SECONDS=30)
    breaker = CircuitBreaker(fake_redis, "test1", cfg)
    assert await breaker.get_state() == CircuitState.CLOSED
    assert await breaker.get_failure_count() == 0


@pytest.mark.asyncio
async def test_closed_allows_requests(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """CLOSED circuits always allow requests."""
    breaker = CircuitBreaker(fake_redis, "test2")
    assert await breaker.should_allow() is True


@pytest.mark.asyncio
async def test_opens_after_threshold(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """Circuit transitions CLOSED -> OPEN after reaching failure threshold."""
    cfg = Settings(CIRCUIT_FAILURE_THRESHOLD=3, CIRCUIT_WINDOW_SECONDS=60, CIRCUIT_COOLDOWN_SECONDS=30)
    breaker = CircuitBreaker(fake_redis, "test3", cfg)

    assert await breaker.record_failure() == CircuitState.CLOSED
    assert await breaker.record_failure() == CircuitState.CLOSED
    state = await breaker.record_failure()
    assert state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_open_blocks_requests(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """OPEN circuits reject requests during cooldown."""
    cfg = Settings(CIRCUIT_FAILURE_THRESHOLD=2, CIRCUIT_WINDOW_SECONDS=60, CIRCUIT_COOLDOWN_SECONDS=30)
    breaker = CircuitBreaker(fake_redis, "test4", cfg)

    await breaker.record_failure()
    await breaker.record_failure()  # -> OPEN

    assert await breaker.should_allow() is False


@pytest.mark.asyncio
async def test_half_open_after_cooldown(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """After cooldown expires, OPEN -> HALF_OPEN and allows one probe."""
    cfg = Settings(
        CIRCUIT_FAILURE_THRESHOLD=2,
        CIRCUIT_WINDOW_SECONDS=60,
        CIRCUIT_COOLDOWN_SECONDS=1,  # 1 second cooldown
    )
    breaker = CircuitBreaker(fake_redis, "test5", cfg)

    await breaker.record_failure()
    await breaker.record_failure()  # -> OPEN

    # Still in cooldown
    assert await breaker.should_allow() is False

    # Wait for cooldown (slight buffer for CI)
    import asyncio
    await asyncio.sleep(1.1)

    # Should now allow (transition to HALF_OPEN)
    assert await breaker.should_allow() is True
    assert await breaker.get_state() == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_half_open_success_closes_circuit(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """HALF_OPEN + success -> CLOSED, failure count reset."""
    cfg = Settings(CIRCUIT_FAILURE_THRESHOLD=2, CIRCUIT_WINDOW_SECONDS=60, CIRCUIT_COOLDOWN_SECONDS=1)
    breaker = CircuitBreaker(fake_redis, "test6", cfg)

    await breaker.record_failure()
    await breaker.record_failure()  # -> OPEN

    import asyncio
    await asyncio.sleep(1.1)
    await breaker.should_allow()  # -> HALF_OPEN

    state = await breaker.record_success()
    assert state == CircuitState.CLOSED
    assert await breaker.get_failure_count() == 0


@pytest.mark.asyncio
async def test_half_open_failure_reopens_circuit(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """HALF_OPEN + failure -> OPEN (cooldown restarts)."""
    cfg = Settings(CIRCUIT_FAILURE_THRESHOLD=2, CIRCUIT_WINDOW_SECONDS=60, CIRCUIT_COOLDOWN_SECONDS=5)
    breaker = CircuitBreaker(fake_redis, "test7", cfg)

    await breaker.record_failure()
    await breaker.record_failure()  # -> OPEN

    import asyncio
    await asyncio.sleep(1.1)
    await breaker.should_allow()  # -> HALF_OPEN

    state = await breaker.record_failure()
    assert state == CircuitState.OPEN
    assert await breaker.should_allow() is False  # cooldown restarted


@pytest.mark.asyncio
async def test_half_open_allows_only_one_probe(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """While in HALF_OPEN, only one probe is allowed through."""
    cfg = Settings(CIRCUIT_FAILURE_THRESHOLD=2, CIRCUIT_WINDOW_SECONDS=60, CIRCUIT_COOLDOWN_SECONDS=1)
    breaker = CircuitBreaker(fake_redis, "test8", cfg)

    await breaker.record_failure()
    await breaker.record_failure()  # -> OPEN

    import asyncio
    await asyncio.sleep(1.1)
    assert await breaker.should_allow() is True   # -> HALF_OPEN, first probe
    assert await breaker.should_allow() is False   # second probe blocked
    assert await breaker.should_allow() is False   # still blocked


@pytest.mark.asyncio
async def test_concurrent_probe_claim(fake_redis: fakeredis.aioredis.FakeRedis) -> None:
    """When cooldown expires, exactly one concurrent request claims the probe."""
    cfg = Settings(
        CIRCUIT_FAILURE_THRESHOLD=2,
        CIRCUIT_WINDOW_SECONDS=60,
        CIRCUIT_COOLDOWN_SECONDS=1,
    )
    breaker = CircuitBreaker(fake_redis, "test_concurrent", cfg)

    await breaker.record_failure()
    await breaker.record_failure()  # -> OPEN

    await asyncio.sleep(1.1)  # wait for cooldown to expire

    # Fire 5 concurrent should_allow() calls
    results = await asyncio.gather(*[breaker.should_allow() for _ in range(5)])

    true_count = sum(1 for r in results if r is True)
    false_count = sum(1 for r in results if r is False)

    assert true_count == 1, f"Expected exactly 1 probe, got {true_count}"
    assert false_count == 4, f"Expected 4 rejections, got {false_count}"
    assert await breaker.get_state() == CircuitState.HALF_OPEN
