from dataclasses import dataclass
from agent_platform.core.config import Settings

@dataclass(frozen=True, slots=True)
class ModelConfig:
    provider: str
    model: str
    base_url: str | None
    timeout_seconds: float
    max_retries: int
    max_output_tokens: int

def build_model_config(settings: Settings) -> ModelConfig:  return ModelConfig(provider=settings.model_provider,
                                                                                   model=settings.model_name,
                                                                                   base_url=settings.model_base_url,
                                                                               timeout_seconds=settings.model_timeout_seconds,
                                                                               max_retries=settings.model_max_retries,
                                                                               max_output_tokens=settings.model_max_output_tokens)

