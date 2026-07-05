"""Regex-based fast filter for obvious jailbreak patterns.

This detector runs in <10 ms and catches the most common prompt-injection
attempts. It is intentionally simple — future detectors (ML-based, etc.)
will implement the same ``Detector`` interface and can be composed together.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.detectors.base import Detector, DetectionResult

_PATTERNS_FILE = Path(__file__).parent / "patterns.json"


class FastFilter(Detector):
    """Regex scanner against a hardcoded list of jailbreak phrases.

    Loaded once at import time so repeated calls are fast.
    """

    def __init__(self) -> None:
        raw: list[str] = json.loads(_PATTERNS_FILE.read_text(encoding="utf-8"))
        # Pre-compile each pattern as a case-insensitive regex
        self._compiled = [re.compile(re.escape(p), re.IGNORECASE) for p in raw]

    async def check(self, text: str) -> DetectionResult:
        """Return DetectionResult indicating whether *text* is safe.

        Score is 1.0 if any pattern matches, 0.0 otherwise.
        """
        for pattern in self._compiled:
            if pattern.search(text):
                return DetectionResult(
                    is_safe=False,
                    score=1.0,
                    reason=f"Matched jailbreak pattern: {pattern.pattern}",
                )
        return DetectionResult(is_safe=True, score=0.0, reason="")
