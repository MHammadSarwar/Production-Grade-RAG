from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from app.agent import ProductionAgent
from app.cache import InMemoryCache
from app.config import Settings, get_settings
from app.models import ChatRequest, ChatResponse, HealthResponse, MetricsResponse
from app.monitoring import MetricsCollector, StructuredJSONLogger


def create_app(settings: Settings | None = None) -> FastAPI:
	app_settings = settings or get_settings()
	limiter = Limiter(key_func=get_remote_address)

	@asynccontextmanager
	async def lifespan(app: FastAPI) -> AsyncIterator[None]:
		app_logger = StructuredJSONLogger()
		metrics_collector = MetricsCollector()
		response_cache: InMemoryCache[ChatResponse] = InMemoryCache()
		agent = ProductionAgent(
			settings=app_settings,
			metrics_collector=metrics_collector,
		)
		app.state.logger = app_logger
		app.state.metrics = metrics_collector
		app.state.cache = response_cache
		app.state.agent = agent
		app_logger.info("application_started", environment=app_settings.environment)
		try:
			yield
		finally:
			app_logger.info(
				"application_stopped",
				metrics=metrics_collector.snapshot(),
				cache=response_cache.stats(),
			)

	app = FastAPI(title=app_settings.app_name, lifespan=lifespan)
	app.state.limiter = limiter
	app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

	@app.middleware("http")
	async def record_http_metrics(request: Request, call_next):
		started_at = perf_counter()
		route = request.scope.get("route")
		route_name = getattr(route, "path", None) or "unmatched"
		try:
			response = await call_next(request)
		except Exception as error:
			collector: MetricsCollector = request.app.state.metrics
			collector.record_request(request.method, route_name, 500)
			collector.record_error(type(error).__name__)
			collector.observe_latency(route_name, perf_counter() - started_at)
			request.app.state.logger.error(
				"http_request_failed",
				route=route_name,
				error_type=type(error).__name__,
			)
			raise

		collector = request.app.state.metrics
		collector.record_request(request.method, route_name, response.status_code)
		collector.observe_latency(route_name, perf_counter() - started_at)
		return response

	@app.post("/chat", response_model=ChatResponse)
	@limiter.limit(app_settings.rate_limit)
	async def chat(request: Request, body: ChatRequest) -> ChatResponse:
		namespace = (
			f"{app_settings.primary_model}:{app_settings.fallback_model}:"
			f"{body.conversation_id or 'stateless'}"
		)
		cache: InMemoryCache[ChatResponse] = request.app.state.cache
		metrics_collector: MetricsCollector = request.app.state.metrics
		cached_response = cache.get(body.message, namespace=namespace)
		if cached_response is not None:
			metrics_collector.record_cache_result(True)
			return cached_response.model_copy(update={"cached": True})

		metrics_collector.record_cache_result(False)
		agent: ProductionAgent = request.app.state.agent
		response = await run_in_threadpool(
			agent.invoke,
			body.message,
			body.conversation_id,
		)
		if not response.blocked and response.answer != agent.ERROR_RESPONSE:
			cache.set(body.message, response, namespace=namespace)
		return response

	@app.get("/health", response_model=HealthResponse)
	async def health(request: Request) -> HealthResponse:
		return HealthResponse(
			status="healthy",
			service=app_settings.app_name,
			environment=app_settings.environment,
			checks={
				"agent": hasattr(request.app.state, "agent"),
				"cache": hasattr(request.app.state, "cache"),
				"metrics": hasattr(request.app.state, "metrics"),
			},
		)

	@app.get("/metrics", response_model=MetricsResponse)
	async def read_metrics(request: Request) -> MetricsResponse:
		return MetricsResponse(**request.app.state.metrics.snapshot())

	@app.get("/cache/stats")
	async def read_cache_stats(request: Request) -> dict[str, int | float]:
		return request.app.state.cache.stats()

	return app


app = create_app()
