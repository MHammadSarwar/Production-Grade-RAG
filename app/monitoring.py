import logging
import sys
from time import monotonic
from typing import Any

import structlog
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class StructuredJSONLogger:
    def __init__(self, name: str = "production_grade_rag", level: int = logging.INFO) -> None:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.add_log_level,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
            cache_logger_on_first_use=True,
        )
        self._logger = structlog.get_logger(name)

    def info(self, event: str, **fields: Any) -> None:
        self._logger.info(event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._logger.warning(event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._logger.error(event, **fields)


class MetricsCollector:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self._started_at = monotonic()
        self.requests_total = Counter(
            "rag_requests_total",
            "Total HTTP requests handled by the RAG application.",
            ("method", "route", "status_code"),
            registry=self.registry,
        )
        self.requests_failed = Counter(
            "rag_requests_failed_total",
            "Total HTTP requests that returned an error status.",
            ("method", "route", "status_code"),
            registry=self.registry,
        )
        self.errors_total = Counter(
            "rag_errors_total",
            "Total application errors by bounded error type.",
            ("error_type",),
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "rag_request_duration_seconds",
            "HTTP request duration in seconds.",
            ("route",),
            registry=self.registry,
        )
        self.cache_requests = Counter(
            "rag_cache_requests_total",
            "Cache lookups by result.",
            ("result",),
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "rag_llm_tokens_total",
            "LLM tokens consumed by direction and model.",
            ("direction", "model"),
            registry=self.registry,
        )

    def record_request(self, method: str, route: str, status_code: int) -> None:
        labels = (method.upper(), route, str(status_code))
        self.requests_total.labels(*labels).inc()
        if status_code >= 400:
            self.requests_failed.labels(*labels).inc()

    def record_error(self, error_type: str) -> None:
        self.errors_total.labels(error_type=error_type).inc()

    def observe_latency(self, route: str, duration_seconds: float) -> None:
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        self.request_latency.labels(route=route).observe(duration_seconds)

    def record_cache_result(self, hit: bool) -> None:
        self.cache_requests.labels(result="hit" if hit else "miss").inc()

    def record_llm_tokens(self, model: str, *, input_tokens: int, output_tokens: int) -> None:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        self.llm_tokens.labels(direction="input", model=model).inc(input_tokens)
        self.llm_tokens.labels(direction="output", model=model).inc(output_tokens)

    def render_prometheus(self) -> bytes:
        return generate_latest(self.registry)

    def snapshot(self) -> dict[str, int | float]:
        samples = [sample for family in self.registry.collect() for sample in family.samples]

        def total(sample_name: str, **labels: str) -> float:
            return sum(
                sample.value
                for sample in samples
                if sample.name == sample_name
                and all(sample.labels.get(key) == value for key, value in labels.items())
            )

        requests_total = int(total("rag_requests_total"))
        requests_failed = int(total("rag_requests_failed_total"))
        latency_count = total("rag_request_duration_seconds_count")
        latency_sum = total("rag_request_duration_seconds_sum")
        cache_hits = total("rag_cache_requests_total", result="hit")
        cache_misses = total("rag_cache_requests_total", result="miss")
        return {
            "requests_total": requests_total,
            "requests_failed": requests_failed,
            "average_latency_ms": latency_sum * 1_000 / latency_count if latency_count else 0.0,
            "uptime_seconds": monotonic() - self._started_at,
            "total_errors": int(total("rag_errors_total")),
            "error_rate": requests_failed / requests_total if requests_total else 0.0,
            "cache_hit_rate": cache_hits / (cache_hits + cache_misses)
            if cache_hits + cache_misses
            else 0.0,
            "total_input_tokens": int(total("rag_llm_tokens_total", direction="input")),
            "total_output_tokens": int(total("rag_llm_tokens_total", direction="output")),
        }


logger = StructuredJSONLogger()
metrics = MetricsCollector()
