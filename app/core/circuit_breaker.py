"""Circuit breaker state machine backed by Redis.

Tracks per-circuit-id failure counts and state transitions using atomic Redis
operations (WATCH/MULTI) so the system remains correct under concurrent
requests and across multiple uvicorn workers.
"""

from __future__ import annotations

import enum
import logging

import redis.asyncio as aioredis

from app.core.config import Settings, settings

logger = logging.getLogger(__name__)

_STATE_KEY = "circuit:{cid}:state"
_FAILURES_KEY = "circuit:{cid}:failures"
_COOLDOWN_KEY = "circuit:{cid}:cooldown"


class CircuitState(str, enum.Enum):
    """Finite states of the circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Pure-logic circuit breaker with Redis-backed persistence.

    Uses WATCH/MULTI for atomic state transitions so concurrent requests for
    the same circuit_id remain consistent without a Lua script.
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        redis: aioredis.Redis,
        circuit_id: str,
        config: Settings | None = None,
    ) -> None:
        self._redis = redis
        self._cid = circuit_id
        self._cfg = config or settings

    def _key(self, template: str) -> str:
        return template.format(cid=self._cid)

    # State reads ---------------------------------------------------------------

    async def get_state(self) -> CircuitState:
        """Read the current circuit state from Redis (defaults to CLOSED)."""
        raw = await self._redis.get(self._key(_STATE_KEY))
        if raw is None:
            return CircuitState.CLOSED
        return CircuitState(raw)

    async def get_failure_count(self) -> int:
        """Read the current failure count from Redis."""
        raw = await self._redis.get(self._key(_FAILURES_KEY))
        return int(raw) if raw is not None else 0

    async def get_cooldown_remaining(self) -> float:
        """Return seconds left on the cooldown timer, or 0 if inactive."""
        ttl = await self._redis.ttl(self._key(_COOLDOWN_KEY))
        return max(float(ttl), 0.0)

    # Decision ------------------------------------------------------------------

    async def should_allow(self) -> bool:
        """Determine whether the current request should proceed.

        CLOSED  -> always allow.
        OPEN    -> block if cooldown still active; otherwise atomically claim
                  the probe slot via WATCH/MULTI. Exactly one concurrent
                  request wins the transition to HALF_OPEN; all others are
                  rejected as if the circuit were still OPEN.
        HALF_OPEN -> block (only one probe allowed, which is already in flight).
        """
        state = await self.get_state()

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            remaining = await self._redis.ttl(self._key(_COOLDOWN_KEY))
            if remaining > 0:
                return False

            # Cooldown expired — atomically claim the probe slot
            state_key = self._key(_STATE_KEY)
            for _attempt in range(self.MAX_RETRIES):
                try:
                    async with self._redis.pipeline(transaction=True) as pipe:
                        await pipe.watch(state_key)
                        current = await pipe.get(state_key)

                        # Another request may have already transitioned
                        if current != CircuitState.OPEN:
                            return False

                        pipe.multi()
                        pipe.set(state_key, CircuitState.HALF_OPEN)
                        await pipe.execute()
                        logger.info(
                            "Circuit %s -> HALF_OPEN (probe claimed)", self._cid
                        )
                        return True
                except aioredis.WatchError:
                    continue

            # All retries exhausted — fail closed
            return False

        # HALF_OPEN — a probe is already in flight
        return False

    # Recording outcomes --------------------------------------------------------

    async def record_success(self) -> CircuitState:
        """Record a successful request. Resets failure count.

        HALF_OPEN -> CLOSED (circuit healed).
        CLOSED    -> stays CLOSED (resets window).
        """
        state = await self.get_state()
        await self._redis.delete(self._key(_FAILURES_KEY))

        if state == CircuitState.HALF_OPEN:
            await self._transition_to_closed()
            return CircuitState.CLOSED

        return state

    async def record_failure(self) -> CircuitState:
        """Record a failed request atomically using WATCH/MULTI.

        CLOSED + threshold reached -> OPEN.
        HALF_OPEN -> OPEN (reset cooldown).
        Retries on optimistic lock conflict.
        """
        state_key = self._key(_STATE_KEY)
        failures_key = self._key(_FAILURES_KEY)
        cooldown_key = self._key(_COOLDOWN_KEY)

        for _attempt in range(self.MAX_RETRIES):
            try:
                async with self._redis.pipeline(transaction=True) as pipe:
                    await pipe.watch(state_key, failures_key)
                    current_state = await pipe.get(state_key) or CircuitState.CLOSED
                    current_failures = int(await pipe.get(failures_key) or 0)

                    pipe.multi()
                    new_failures = current_failures + 1
                    pipe.set(failures_key, new_failures, ex=self._cfg.CIRCUIT_WINDOW_SECONDS)

                    new_state = current_state
                    if current_state == CircuitState.HALF_OPEN:
                        new_state = CircuitState.OPEN
                        pipe.set(state_key, new_state)
                        pipe.set(cooldown_key, "1", ex=self._cfg.CIRCUIT_COOLDOWN_SECONDS)
                    elif current_state == CircuitState.CLOSED and new_failures >= self._cfg.CIRCUIT_FAILURE_THRESHOLD:
                        new_state = CircuitState.OPEN
                        pipe.set(state_key, new_state)
                        pipe.set(cooldown_key, "1", ex=self._cfg.CIRCUIT_COOLDOWN_SECONDS)

                    await pipe.execute()
                    return CircuitState(new_state)
            except aioredis.WatchError:
                continue

        # Last resort: fall through to simple read-then-write
        current_state = await self.get_state()
        new_failures = await self._redis.incr(failures_key)
        await self._redis.expire(failures_key, self._cfg.CIRCUIT_WINDOW_SECONDS)

        if current_state == CircuitState.HALF_OPEN:
            await self._transition_to_open()
            return CircuitState.OPEN
        if current_state == CircuitState.CLOSED and new_failures >= self._cfg.CIRCUIT_FAILURE_THRESHOLD:
            await self._transition_to_open()
            return CircuitState.OPEN

        return current_state

    # Internal transitions ------------------------------------------------------

    async def _transition_to_closed(self) -> None:
        """HALF_OPEN -> CLOSED after a successful probe."""
        await self._redis.set(self._key(_STATE_KEY), CircuitState.CLOSED)
        await self._redis.delete(self._key(_COOLDOWN_KEY))
        logger.info("Circuit %s -> CLOSED", self._cid)

    async def _transition_to_open(self) -> None:
        """Transition to OPEN state with cooldown timer."""
        await self._redis.set(self._key(_STATE_KEY), CircuitState.OPEN)
        await self._redis.set(
            self._key(_COOLDOWN_KEY), "1", ex=self._cfg.CIRCUIT_COOLDOWN_SECONDS
        )
        logger.info("Circuit %s -> OPEN", self._cid)
