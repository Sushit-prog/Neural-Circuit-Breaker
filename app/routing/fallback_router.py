"""Graceful fallback routing for blocked requests.

When a request is blocked (by any detector or the circuit breaker), the caller
receives a safe, usable response instead of a bare rejection. Two strategies:

- **STATIC** (default): returns a fixed canned message. Zero cost, zero latency,
  zero external dependency risk.
- **MODEL**: calls a hosted inference API (Groq / OpenRouter) with a strict
  system prompt that only knows the request was blocked — the raw unsafe text
  is never forwarded to the fallback model, preventing re-exploitation.

Fail-closed: if the MODEL strategy's API call fails for any reason, the router
silently falls back to STATIC. A broken fallback provider never results in an
error response to the caller.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx

from app.routing.schemas import FallbackResponse

logger = logging.getLogger(__name__)

_FALLBACK_TIMEOUT = 3.0  # seconds — a slow provider must not dominate latency

_SYSTEM_PROMPT = (
    "You are a safety system. A user request was blocked because it was flagged "
    "as potentially unsafe. Do NOT attempt to fulfill the original request. "
    "Respond ONLY with a brief, safe, generic message declining to engage. "
    "Keep it to one short sentence."
)

_PROVIDER_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


class FallbackRouter:
    """Decides and delivers the safe fallback for blocked requests.

    Stateless — holds only configuration, no per-request mutable state.
    """

    def __init__(
        self,
        strategy: Literal["static", "model"] = "static",
        static_message: str = "",
        model_provider: Literal["groq", "openrouter"] = "groq",
        model_name: str = "",
        model_api_key: str = "",
    ) -> None:
        self._strategy = strategy
        self._static_message = static_message
        self._model_provider = model_provider
        self._model_name = model_name
        self._api_key = model_api_key

        if strategy == "model" and not model_api_key:
            logger.warning(
                "FALLBACK_STRATEGY is 'model' but FALLBACK_MODEL_API_KEY is "
                "empty — falling back to STATIC strategy silently"
            )
            self._strategy = "static"

    async def get_fallback(
        self,
        circuit_id: str,
        original_text: str,
        block_reason: str,
    ) -> FallbackResponse:
        """Return a safe fallback response for a blocked request.

        ``original_text`` is accepted for interface completeness but is
        intentionally never forwarded to the model — only ``block_reason``
        and ``circuit_id`` are used in the model prompt to prevent
        re-exploitation through the fallback path itself.
        """
        if self._strategy == "model":
            result = await self._try_model_fallback(circuit_id, block_reason)
            if result is not None:
                return result
            # Model call failed — fall through to static
            logger.warning("Model fallback failed, using static fallback")

        return FallbackResponse(
            message=self._static_message,
            strategy="static",
        )

    async def _try_model_fallback(
        self,
        circuit_id: str,
        block_reason: str,
    ) -> FallbackResponse | None:
        """Attempt the model-based fallback. Returns None on any failure."""
        try:
            url = _PROVIDER_URLS[self._model_provider]
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload: dict[str, Any] = {
                "model": self._model_name,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"A request to endpoint '{circuit_id}' was blocked. "
                            f"Reason: {block_reason}. "
                            f"Provide a brief, safe, generic declination."
                        ),
                    },
                ],
                "max_tokens": 60,
                "temperature": 0.0,
            }

            async with httpx.AsyncClient(timeout=_FALLBACK_TIMEOUT) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()

            body = resp.json()
            message = body["choices"][0]["message"]["content"].strip()

            if not message:
                logger.warning("Model fallback returned empty content")
                return None

            return FallbackResponse(
                message=message,
                strategy="model",
            )

        except Exception:
            logger.exception("Model fallback call failed for provider %s", self._model_provider)
            return None
