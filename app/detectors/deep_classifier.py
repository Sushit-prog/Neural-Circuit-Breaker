"""ML-based prompt-injection detector using protectai/deberta-v3-base-prompt-injection-v2.

Runs on CPU only. Model + tokenizer are loaded once at init time, not per-request.
Inference is offloaded to a thread via asyncio.to_thread() so the main event loop
stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from app.core.config import Settings, settings
from app.detectors.base import Detector, DetectionResult

logger = logging.getLogger(__name__)

_MODEL_MAX_LENGTH = 512  # deberta-v3-base context window


class DeepClassifier(Detector):
    """Second-pass ML classifier for prompt-injection detection.

    Wraps a HuggingFace text-classification pipeline. Must be instantiated
    once at app startup (not per-request) because model loading is expensive.
    """

    def __init__(
        self,
        config: Settings | None = None,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._cfg = config or settings
        self._pipeline = None
        self._load_model(pipeline_factory)

    @property
    def is_ready(self) -> bool:
        """True only when the model loaded successfully and inference is possible."""
        return self._pipeline is not None

    def _load_model(self, pipeline_factory: Callable[..., Any] | None = None) -> None:
        """Load the HuggingFace pipeline. Called once at init.

        ``pipeline_factory`` is an optional override for testing — when None,
        the real ``transformers.pipeline`` is used.
        """
        try:
            if pipeline_factory is not None:
                hf_pipeline = pipeline_factory
            else:
                from transformers import pipeline as hf_pipeline

            logger.info(
                "Loading deep classifier model: %s (this may take a moment on first run)",
                self._cfg.DEEP_CLASSIFIER_MODEL,
            )
            self._pipeline = hf_pipeline(
                "text-classification",
                model=self._cfg.DEEP_CLASSIFIER_MODEL,
                device=-1,  # force CPU
                top_k=None,
            )
            logger.info("Deep classifier model loaded successfully")
        except Exception:
            logger.exception(
                "Failed to load deep classifier model '%s' — "
                "deep classification will be unavailable",
                self._cfg.DEEP_CLASSIFIER_MODEL,
            )
            self._pipeline = None

    async def check(self, text: str) -> DetectionResult:
        """Evaluate *text* using the ML classifier.

        Returns DetectionResult with score 0.0 (safe) to 1.0 (injection).
        Offloads inference to a thread so the event loop is not blocked.
        If the model failed to load or inference throws, fails closed.
        """
        if self._pipeline is None:
            return DetectionResult(
                is_safe=False,
                score=1.0,
                reason="Deep classifier unavailable (model failed to load) — failing closed",
            )

        # Truncate to model max length
        truncated = False
        if len(text) > _MODEL_MAX_LENGTH:
            text = text[:_MODEL_MAX_LENGTH]
            truncated = True
            logger.debug("Truncated input to %d chars for deep classifier", _MODEL_MAX_LENGTH)

        try:
            t0 = time.perf_counter()
            results = await asyncio.to_thread(self._pipeline, text)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug("Deep classifier inference: %.1f ms", elapsed_ms)
        except Exception:
            logger.exception("Deep classifier inference failed — failing closed")
            return DetectionResult(
                is_safe=False,
                score=1.0,
                reason="Deep classifier inference error — failing closed",
            )

        # Parse pipeline output: list of dicts with 'label' and 'score'
        # pipeline returns [[{label, score}, ...]] for single input
        if isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], list):
                scores_dict = {item["label"]: item["score"] for item in results[0]}
            else:
                scores_dict = {item["label"]: item["score"] for item in results}
        else:
            return DetectionResult(
                is_safe=False,
                score=1.0,
                reason="Deep classifier returned unexpected output format",
            )

        # The protectai model labels: "INJECTION" (unsafe) vs "SAFE"
        injection_score = scores_dict.get("INJECTION", 0.0)
        safe_score = scores_dict.get("SAFE", 0.0)

        is_injection = injection_score > self._cfg.DEEP_CLASSIFIER_THRESHOLD

        if is_injection:
            reason = f"Deep classifier flagged as prompt injection (score: {injection_score:.4f})"
            if truncated:
                reason += " [input was truncated]"
            return DetectionResult(
                is_safe=False,
                score=injection_score,
                reason=reason,
            )

        return DetectionResult(
            is_safe=True,
            score=safe_score,
            reason="",
        )
