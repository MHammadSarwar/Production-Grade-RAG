
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

	app_name: str = "Production Grade RAG"
	environment: Literal["development", "testing", "production"] = "development"

	groq_api_key: SecretStr | None = None
	primary_model: str = "openai/gpt-oss-120b"
	fallback_model: str = "openai/gpt-oss-20b"
	model_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
	model_max_retries: int = Field(default=2, ge=0)
	rate_limit: str = "60/minute"

	langsmith_tracing: bool = False
	langsmith_api_key: SecretStr | None = None
	langsmith_project: str = "production-grade-rag-dev"


@lru_cache
def get_settings() -> Settings:
	return Settings()
