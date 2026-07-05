"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.redis_client import close_redis, get_redis

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: verify Redis, load deep classifier. Shutdown: close Redis."""
    await get_redis()

    # Load deep classifier once at startup (if enabled)
    app.state.deep_classifier = None
    if settings.DEEP_CLASSIFIER_ENABLED:
        try:
            from app.detectors.deep_classifier import DeepClassifier

            app.state.deep_classifier = DeepClassifier(settings)
            if app.state.deep_classifier.is_ready:
                logger.info("Deep classifier initialized and ready")
            else:
                logger.error(
                    "Deep classifier model failed to load — "
                    "deep classification is DISABLED for this session; "
                    "requests will only be checked by fast filter"
                )
                app.state.deep_classifier = None
        except Exception:
            logger.exception(
                "Failed to initialize deep classifier — "
                "requests will only be checked by fast filter"
            )
            app.state.deep_classifier = None
    else:
        logger.info("Deep classifier disabled via DEEP_CLASSIFIER_ENABLED=False")

    yield

    await close_redis()


app = FastAPI(
    title="Neural Circuit Breaker",
    description="Circuit-breaker state machine for LLM safety",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — confirms the process is running."""
    return {"status": "ok"}
