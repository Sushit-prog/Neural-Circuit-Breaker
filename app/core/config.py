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

    DEEP_CLASSIFIER_ENABLED: bool = True
    DEEP_CLASSIFIER_MODEL: str = "protectai/deberta-v3-base-prompt-injection-v2"
    DEEP_CLASSIFIER_THRESHOLD: float = 0.5

    FALLBACK_STRATEGY: str = "static"
    FALLBACK_STATIC_MESSAGE: str = (
        "This request could not be processed safely. Please rephrase and try again."
    )
    FALLBACK_MODEL_PROVIDER: str = "groq"
    FALLBACK_MODEL_NAME: str = "llama-3.1-8b-instant"
    FALLBACK_MODEL_API_KEY: str = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


settings = Settings()
