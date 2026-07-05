"""Shared async Redis client instance.

Uses redis.asyncio so all I/O stays non-blocking in the FastAPI request path.
A single module-level client is reused across the application lifetime.
"""

import logging

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    """Return the shared Redis connection, creating it on first call.

    Raises RuntimeError if Redis is unreachable so the caller can fail closed.
    """
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
        try:
            await _redis.ping()
            logger.info("Redis connection established to %s", settings.REDIS_URL)
        except Exception:
            _redis = None
            logger.exception("Failed to connect to Redis at %s", settings.REDIS_URL)
            raise RuntimeError("Redis is unreachable — failing closed")
    return _redis


async def close_redis() -> None:
    """Shut down the Redis connection pool (called on app teardown)."""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
        logger.info("Redis connection closed")
