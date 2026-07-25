"""Hosted-API detector using OpenAI's Moderation endpoint.

Demonstrates that the `Detector` interface isn't tied to local model inference —
this implementation calls a remote API instead of loading anything into memory.
Same contract as DeepClassifier: async `check(text) -> DetectionResult`,
fails closed on any error, never raises out of `check()`.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import Settings, settings
from app.detectors.base import Detector, DetectionResult

logger = logging.getLogger(__name__)

_MODERATION_URL = "https://api.openai.com/v1/moderations"
_REQUEST_TIMEOUT = 3.0  # seconds — a slow provider must not dominate request latency


class ModerationDetector(Detector):
    """Third-pass detector backed by OpenAI's hosted moderation API.

    Unlike DeepClassifier (loaded once, runs in-process), this detector holds
    no model state — every check is a network call. `is_ready` reflects
    whether an API key is configured, not whether a model loaded.
    """

    def __init__(
        self,
        config: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cfg = config or settings
        self._api_key = self._cfg.MODERATION_API_KEY
        self._client = client  # optional override for testing

        if not self._api_key:
            logger.warning(
                "ModerationDetector enabled but MODERATION_API_KEY is empty — "
                "this detector will fail closed on every request"
            )

    @property
    def is_ready(self) -> bool:
        """True only when an API key is configured."""
        return bool(self._api_key)

    async def check(self, text: str) -> DetectionResult:
        """Evaluate *text* via the hosted moderation API.

        Fails closed (is_safe=False) on missing config, network error,
        timeout, non-2xx response, or unexpected payload shape — mirrors
        DeepClassifier's fail-closed contract so routes.py can treat every
        detector identically.
        """
        if not self._api_key:
            return DetectionResult(
                is_safe=False,
                score=1.0,
                reason="Moderation detector unavailable (no API key) — failing closed",
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {"input": text}

        try:
            if self._client is not None:
                resp = await self._client.post(_MODERATION_URL, headers=headers, json=payload)
            else:
                async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                    resp = await client.post(_MODERATION_URL, headers=headers, json=payload)
            resp.raise_for_status()
        except httpx.TimeoutException:
            logger.warning("Moderation API timed out — failing closed")
            return DetectionResult(
                is_safe=False, score=1.0, reason="Moderation API timeout — failing closed"
            )
        except Exception:
            logger.exception("Moderation API call failed — failing closed")
            return DetectionResult(
                is_safe=False, score=1.0, reason="Moderation API error — failing closed"
            )

        try:
            body = resp.json()
            result = body["results"][0]
            flagged = bool(result["flagged"])
            # category_scores: take the max across categories as the overall score
            category_scores = result.get("category_scores", {})
            score = max(category_scores.values()) if category_scores else (1.0 if flagged else 0.0)
        except (KeyError, IndexError, ValueError):
            logger.exception("Moderation API returned unexpected payload shape — failing closed")
            return DetectionResult(
                is_safe=False,
                score=1.0,
                reason="Moderation API unexpected response format — failing closed",
            )

        if flagged:
            top_category = max(category_scores, key=category_scores.get) if category_scores else "unknown"
            return DetectionResult(
                is_safe=False,
                score=score,
                reason=f"Moderation API flagged content (top category: {top_category}, score: {score:.4f})",
            )

        return DetectionResult(is_safe=True, score=score, reason="")
