"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes import router
from app.core.redis_client import close_redis, get_redis

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: verify Redis connectivity. Shutdown: close the pool."""
    await get_redis()
    yield
    await close_redis()


app = FastAPI(
    title="Neural Circuit Breaker",
    description="Circuit-breaker state machine for LLM safety",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — confirms the process is running."""
    return {"status": "ok"}
