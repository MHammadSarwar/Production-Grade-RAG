from time import perf_counter
from typing import Any, Literal, TypedDict

from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langsmith import Client, traceable
from langsmith.run_helpers import tracing_context

from app.config import Settings, get_settings
from app.models import ChatResponse
from app.monitoring import MetricsCollector, logger, metrics
from app.security import security_pipeline


class AgentState(TypedDict, total=False):
	message: str
	conversation_id: str | None
	input_checked: bool
	security_notes: list[str]
	cleaned_message: str
	input_allowed: bool
	answer: str
	blocked: bool
	warnings: list[str]
	error_stage: Literal["input", "primary", "fallback"] | None


def _trace_agent_inputs(inputs: dict[str, Any]) -> dict[str, int]:
	message = inputs.get("message")
	return {"message_length": len(message) if isinstance(message, str) else 0}


def _trace_agent_outputs(output: Any) -> dict[str, Any]:
	if not isinstance(output, dict):
		return {"result_type": type(output).__name__}
	answer = output.get("answer")
	return {
		"answer_length": len(answer) if isinstance(answer, str) else 0,
		"blocked": bool(output.get("blocked", False)),
	}


class ProductionAgent:
	ERROR_RESPONSE = "Sorry, I couldn't process your request right now. Please try again shortly."
	INPUT_BLOCK_RESPONSE = "Your message couldn't be processed because it didn't pass security checks."

	def __init__(
		self,
		settings: Settings | None = None,
		primary_model: Runnable[Any, Any] | None = None,
		fallback_model: Runnable[Any, Any] | None = None,
		metrics_collector: MetricsCollector | None = None,
	) -> None:
		self.settings = settings or get_settings()
		self.metrics = metrics_collector or metrics
		if primary_model is None or fallback_model is None:
			if self.settings.groq_api_key is None:
				raise ValueError("GROQ_API_KEY must be configured to initialize Groq models")
			api_key = self.settings.groq_api_key.get_secret_value()
			primary_model = primary_model or self._create_model(
				self.settings.primary_model,
				api_key,
			)
			fallback_model = fallback_model or self._create_model(
				self.settings.fallback_model,
				api_key,
			)

		retry_options = {
			"retry_if_exception_type": (Exception,),
			"stop_after_attempt": self.settings.model_max_retries + 1,
			"wait_exponential_jitter": True,
		}
		self.primary_model = primary_model.with_retry(**retry_options)
		self.fallback_model = fallback_model.with_retry(**retry_options)

		self._langsmith_client = (
			Client(
				api_key=(
					self.settings.langsmith_api_key.get_secret_value()
					if self.settings.langsmith_api_key
					else None
				)
			)
			if self.settings.langsmith_tracing
			else None
		)
		self._traced_graph_invoke = traceable(
			name="production_agent_invoke",
			process_inputs=_trace_agent_inputs,
			process_outputs=_trace_agent_outputs,
		)(self._invoke_graph)
		self.graph = self._build_graph()

	def _create_model(self, model_name: str, api_key: str) -> Runnable[Any, Any]:
		return ChatGroq(
			model=model_name,
			api_key=api_key,
			temperature=self.settings.model_temperature,
			max_retries=0,
		)

	def _build_graph(self):
		workflow = StateGraph(AgentState)
		workflow.add_node("process_message", self.process_message)
		workflow.add_node("try_fallback", self.try_fallback)
		workflow.add_node("handle_error", self.handle_error)
		workflow.add_edge(START, "process_message")
		workflow.add_conditional_edges(
			"process_message",
			self.route_after_process,
			{
				"try_fallback": "try_fallback",
				"handle_error": "handle_error",
				"end": END,
			},
		)
		workflow.add_conditional_edges(
			"try_fallback",
			self.route_after_fallback,
			{"handle_error": "handle_error", "end": END},
		)
		workflow.add_edge("handle_error", END)
		return workflow.compile()

	def process_message(self, state: AgentState) -> AgentState:
		if state.get("input_checked"):
			is_allowed = state.get("input_allowed", False)
			cleaned_message = state.get("cleaned_message", "")
			security_notes = state.get("security_notes", [])
		else:
			is_allowed, cleaned_message, security_notes = security_pipeline.check_input(
				state["message"]
			)
		if not is_allowed:
			return {
				"input_allowed": False,
				"blocked": True,
				"warnings": security_notes,
				"error_stage": "input",
			}

		try:
			answer, blocked, output_warnings = self._generate(
				self.primary_model,
				cleaned_message,
				self.settings.primary_model,
			)
			return {
				"cleaned_message": cleaned_message,
				"input_allowed": True,
				"answer": answer,
				"blocked": blocked,
				"warnings": security_notes + output_warnings,
				"error_stage": None,
			}
		except Exception as error:
			self._record_model_failure("primary", error)
			return {
				"cleaned_message": cleaned_message,
				"input_allowed": True,
				"warnings": security_notes,
				"error_stage": "primary",
			}

	def try_fallback(self, state: AgentState) -> AgentState:
		try:
			answer, blocked, output_warnings = self._generate(
				self.fallback_model,
				state["cleaned_message"],
				self.settings.fallback_model,
			)
			return {
				"answer": answer,
				"blocked": blocked,
				"warnings": state.get("warnings", []) + output_warnings,
				"error_stage": None,
			}
		except Exception as error:
			self._record_model_failure("fallback", error)
			return {"error_stage": "fallback"}

	def handle_error(self, state: AgentState) -> AgentState:
		if state.get("error_stage") == "input":
			return {"answer": self.INPUT_BLOCK_RESPONSE, "blocked": True}
		self.metrics.record_error("model_unavailable")
		return {
			"answer": self.ERROR_RESPONSE,
			"blocked": False,
			"warnings": [],
		}

	@staticmethod
	def route_after_process(state: AgentState) -> str:
		if state.get("error_stage") == "input":
			return "handle_error"
		if state.get("error_stage") == "primary":
			return "try_fallback"
		return "end"

	@staticmethod
	def route_after_fallback(state: AgentState) -> str:
		return "handle_error" if state.get("error_stage") == "fallback" else "end"

	def _generate(
		self,
		model: Runnable[Any, Any],
		prompt: str,
		model_name: str,
	) -> tuple[str, bool, list[str]]:
		with tracing_context(enabled=False):
			result = model.invoke(prompt)
		usage = getattr(result, "usage_metadata", None)
		if isinstance(usage, dict):
			self.metrics.record_llm_tokens(
				model_name,
				input_tokens=int(usage.get("input_tokens", 0)),
				output_tokens=int(usage.get("output_tokens", 0)),
			)
		content = result if isinstance(result, str) else getattr(result, "content", None)
		if isinstance(content, list):
			content = "\n".join(
				block["text"]
				for block in content
				if isinstance(block, dict) and isinstance(block.get("text"), str)
			)
		if not isinstance(content, str) or not content.strip():
			raise ValueError("The model returned no text content")
		is_allowed, safe_answer, warnings = security_pipeline.check_output(content)
		return safe_answer, not is_allowed, warnings

	def _record_model_failure(self, stage: str, error: Exception) -> None:
		self.metrics.record_error(type(error).__name__)
		logger.error(
			"agent_model_attempt_failed",
			stage=stage,
			error_type=type(error).__name__,
		)

	def _invoke_graph(
		self,
		message: str,
		conversation_id: str | None,
		input_allowed: bool,
		security_notes: list[str],
	) -> AgentState:
		return self.graph.invoke(
			{
				"message": message,
				"conversation_id": conversation_id,
				"input_checked": True,
				"input_allowed": input_allowed,
				"cleaned_message": message,
				"security_notes": security_notes,
			}
		)

	def invoke(self, message: str, conversation_id: str | None = None) -> ChatResponse:
		started_at = perf_counter()
		input_allowed, cleaned_message, security_notes = security_pipeline.check_input(message)
		with tracing_context(
			enabled=self.settings.langsmith_tracing,
			project_name=self.settings.langsmith_project,
			client=self._langsmith_client,
		):
			state = self._traced_graph_invoke(
				cleaned_message if input_allowed else "",
				conversation_id,
				input_allowed,
				security_notes,
			)

		return ChatResponse(
			answer=state.get("answer", self.ERROR_RESPONSE),
			model="production-agent",
			conversation_id=conversation_id,
			processing_time_ms=(perf_counter() - started_at) * 1_000,
			blocked=state.get("blocked", False),
			warnings=state.get("warnings", []),
		)
