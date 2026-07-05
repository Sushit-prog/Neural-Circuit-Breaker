"""API routes for content checking and circuit-breaker inspection."""

from __future__ import annotations

import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.schemas import CheckRequest, CheckResponse, CircuitStateResponse
from app.core.circuit_breaker import CircuitBreaker, CircuitState
from app.core.config import settings
from app.core.redis_client import get_redis
from app.detectors.deep_classifier import DeepClassifier
from app.detectors.fast_filter import FastFilter
from app.detectors.base import Detector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["v1"])


def get_fast_filter() -> FastFilter:
    """Return the singleton fast filter instance."""
    return _fast_filter


def get_deep_classifier(request: Request) -> DeepClassifier | None:
    """Return the deep classifier from app state, or None if disabled."""
    if not settings.DEEP_CLASSIFIER_ENABLED:
        return None
    return getattr(request.app.state, "deep_classifier", None)


_fast_filter = FastFilter()


@router.post("/check", response_model=CheckResponse)
async def check_text(
    payload: CheckRequest,
    redis: aioredis.Redis = Depends(get_redis),
    fast_filter: FastFilter = Depends(get_fast_filter),
    deep_classifier: DeepClassifier | None = Depends(get_deep_classifier),
) -> CheckResponse:
    """Evaluate *text* through a two-pass detector pipeline, gated by the circuit breaker.

    1. Circuit breaker gate (existing logic).
    2. FastFilter: catches obvious patterns. If flagged → block immediately,
       do NOT run the deep classifier.
    3. DeepClassifier (if enabled): catches semantic evasion that slips past
       the regex. If flagged → block.
    4. Both pass → record success, return pass.
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

    # Pass 1: fast regex filter
    fast_result = await fast_filter.check(payload.text)
    if not fast_result.is_safe:
        new_state = await breaker.record_failure()
        return CheckResponse(
            status="blocked",
            circuit_state=new_state.value,
            detail=fast_result.reason,
            detector="fast_filter",
        )

    # Pass 2: deep ML classifier (if enabled)
    if deep_classifier is not None:
        # Fail closed when the classifier was requested but its model isn't
        # operational.  Silently passing would defeat the purpose of having a
        # second safety layer — if the infra isn't healthy, block rather than
        # guess.
        if not deep_classifier.is_ready:
            logger.warning(
                "Deep classifier enabled but model not ready — "
                "failing closed on this request"
            )
            new_state = await breaker.record_failure()
            return CheckResponse(
                status="blocked",
                circuit_state=new_state.value,
                detail="Deep classifier model not loaded — failing closed",
                detector="deep_classifier",
            )
        try:
            deep_result = await deep_classifier.check(payload.text)
        except Exception:
            logger.exception("Deep classifier raised unexpectedly — failing closed")
            new_state = await breaker.record_failure()
            return CheckResponse(
                status="blocked",
                circuit_state=new_state.value,
                detail="Deep classifier error — failing closed",
                detector="deep_classifier",
            )

        if not deep_result.is_safe:
            new_state = await breaker.record_failure()
            return CheckResponse(
                status="blocked",
                circuit_state=new_state.value,
                detail=deep_result.reason,
                detector="deep_classifier",
            )

    # Both layers passed
    new_state = await breaker.record_success()
    return CheckResponse(status="pass", circuit_state=new_state.value, detail=None)


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
