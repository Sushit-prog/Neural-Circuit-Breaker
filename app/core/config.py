"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration sourced from .env or environment.

    Every tuneable knob lives here so nothing is hardcoded in business logic.
    """

    REDIS_URL: str = "redis://localhost:6379/0"
    CIRCUIT_FAILURE_THRESHOLD: int = 5
    CIRCUIT_WINDOW_SECONDS: int = 60
    CIRCUIT_COOLDOWN_SECONDS: int = 30

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
