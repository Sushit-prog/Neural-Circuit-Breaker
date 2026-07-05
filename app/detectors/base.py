"""Abstract base class for all content detectors.

Every detector implements ``check(text) -> DetectionResult`` so that future
ML-based detectors can be swapped in without touching the circuit breaker
or API layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class DetectionResult(BaseModel):
    """Outcome of a single detector evaluation."""

    is_safe: bool
    score: float
    reason: str


class Detector(ABC):
    """Interface that all detectors must satisfy."""

    @abstractmethod
    async def check(self, text: str) -> DetectionResult:
        """Evaluate *text* and return whether it is safe.

        ``score`` is 0.0 (completely safe) to 1.0 (definitely unsafe).
        """
        ...
