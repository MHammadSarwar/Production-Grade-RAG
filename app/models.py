from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

from app.security import security_pipeline


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=10_000,
        description="The user's question to answer using the RAG application.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional identifier for continuing an existing conversation.",
    )

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, value: str) -> str:
        is_allowed, cleaned_text, security_notes = security_pipeline.check_input(value)
        if not is_allowed:
            raise ValueError(" ".join(security_notes))
        return cleaned_text


class ChatResponse(BaseModel):
    answer: str = Field(description="The generated answer to the user's question.")
    model: str = Field(description="The model used to generate the answer.")
    conversation_id: str | None = Field(
        default=None,
        description="Identifier for the conversation, when one is available.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Identifiers or references for source documents used in the answer.",
    )
    cached: bool = Field(
        default=False,
        description="Whether the answer was returned from cache.",
    )
    processing_time_ms: float = Field(
        ge=0,
        description="Time spent processing the request in milliseconds.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time when the response was created.",
    )
    blocked: bool = Field(
        default=False,
        description="Whether the generated response was blocked by output validation.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Safety or privacy warnings produced while validating the response.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_output(cls, value: object) -> object:
        if not isinstance(value, dict) or not isinstance(value.get("answer"), str):
            return value

        is_allowed, answer, result_warnings = security_pipeline.check_output(value["answer"])
        validated = value.copy()
        validated["answer"] = answer
        validated["blocked"] = bool(validated.get("blocked", False)) or not is_allowed
        warnings = validated.get("warnings", [])
        if isinstance(warnings, list):
            validated["warnings"] = list(dict.fromkeys(warnings + result_warnings))
        return validated


class MetricsResponse(BaseModel):
    requests_total: int = Field(ge=0, description="Total number of requests received.")
    requests_failed: int = Field(ge=0, description="Total number of failed requests.")
    average_latency_ms: float = Field(
        ge=0,
        description="Average request latency in milliseconds.",
    )
    uptime_seconds: float = Field(ge=0, description="Application uptime in seconds.")
    total_errors: int = Field(ge=0, description="Total number of errors recorded.")
    error_rate: float = Field(
        ge=0,
        le=1,
        description="Fraction of requests that failed, from 0 to 1.",
    )
    cache_hit_rate: float = Field(
        ge=0,
        le=1,
        description="Fraction of requests served from cache, from 0 to 1.",
    )
    total_input_tokens: int = Field(
        ge=0,
        description="Total input tokens processed.",
    )
    total_output_tokens: int = Field(
        ge=0,
        description="Total output tokens generated.",
    )


class ErrorResponse(BaseModel):
    code: str = Field(description="A stable machine-readable error code.")
    message: str = Field(description="A human-readable description of the error.")
    request_id: str | None = Field(
        default=None,
        description="Identifier that can be used to correlate this error with logs.",
    )
    detail: str = Field(description="Additional information about the error.")


class HealthResponse(BaseModel):
    status: Literal["healthy", "unhealthy"] = Field(
        description="Current health status of the application.",
    )
    service: str = Field(description="Name of the application or service.")
    environment: Literal["development", "testing", "production"] = Field(
        description="Application environment reported by the health endpoint.",
    )
    version: str = Field(
        default="1.0.0",
        description="Application version.",
    )
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Health check results keyed by component name.",
    )
