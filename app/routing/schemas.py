"""Response models for the fallback routing layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class FallbackResponse(BaseModel):
    """Safe fallback returned alongside a blocked response.

    Gives the caller something usable instead of a bare rejection.
    """

    message: str
    strategy: Literal["static", "model"]
    fallback_triggered: bool = True
