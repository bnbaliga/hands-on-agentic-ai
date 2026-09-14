from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    app_name: str = "agent-platform"
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    database_url: str

    model_provider: str
    model_name: str
    model_base_url: str | None = None
    model_api_key: SecretStr

    # Operational limits
    model_timeout_seconds: float = Field(default=30, gt=0)
    model_max_retries: int = Field(default=2, ge=0, le=10)
    model_max_output_tokens: int = Field(default=1000, gt=0)

    otel_service_name: str = "agent-platform"
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()