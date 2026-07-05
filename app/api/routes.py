"""API routes for content checking and circuit-breaker inspection."""

from __future__ import annotations

import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import CheckRequest, CheckResponse, CircuitStateResponse
from app.core.circuit_breaker import CircuitBreaker, CircuitState
from app.core.config import settings
from app.core.redis_client import get_redis
from app.detectors.fast_filter import FastFilter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["v1"])

_detector = FastFilter()


@router.post("/check", response_model=CheckResponse)
async def check_text(
    payload: CheckRequest,
    redis: aioredis.Redis = Depends(get_redis),
) -> CheckResponse:
    """Evaluate *text* against the detector, gated by the circuit breaker.

    1. If the circuit is OPEN and cooldown hasn't expired, reject immediately
       (the whole point of the breaker — no detector work is done).
    2. Otherwise run the fast filter.
    3. Record the outcome and return the new state.
    """
    breaker = CircuitBreaker(redis, payload.circuit_id, settings)

    try:
        allowed = await breaker.should_allow()
    except Exception:
        logger.exception("Circuit breaker check failed — failing closed")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    if not allowed:
        state = await breaker.get_state()
        return CheckResponse(
            status="blocked",
            circuit_state=state.value,
            detail="Circuit is OPEN — requests blocked during cooldown",
        )

    result = await _detector.check(payload.text)

    if result.is_safe:
        new_state = await breaker.record_success()
    else:
        new_state = await breaker.record_failure()

    if result.is_safe:
        return CheckResponse(status="pass", circuit_state=new_state.value, detail=None)
    return CheckResponse(status="blocked", circuit_state=new_state.value, detail=result.reason)


@router.get("/circuit/{circuit_id}/state", response_model=CircuitStateResponse)
async def get_circuit_state(
    circuit_id: str,
    redis: aioredis.Redis = Depends(get_redis),
) -> CircuitStateResponse:
    """Debug endpoint: return current state and failure count for *circuit_id*."""
    breaker = CircuitBreaker(redis, circuit_id, settings)
    state = await breaker.get_state()
    count = await breaker.get_failure_count()

    return CircuitStateResponse(
        circuit_id=circuit_id,
        state=state.value,
        failure_count=count,
    )
