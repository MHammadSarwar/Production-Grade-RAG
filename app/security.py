import re
import unicodedata
from typing import Any

from langsmith import traceable


class InputSanitizer:
	INJECTION_PATTERNS = (
		(
			r"\b(?:ignore|disregard|forget|override|bypass|erase)\b.{0,100}"
			r"\b(?:all|any|previous|prior|above|earlier|system|developer|safety)\b.{0,50}"
			r"\b(?:instructions?|prompts?|rules?|messages?|policies)\b",
			"Attempts to override or discard existing instructions.",
		),
		(
			r"\b(?:ignore|disregard|forget|override|bypass|disable)\b.{0,60}"
			r"\b(?:instructions?|prompts?|rules?|policies|safeguards|filters?)\b",
			"Attempts to bypass instructions or safety rules.",
		),
		(
			r"\b(?:do not|dont|never)\b.{0,30}\b(?:follow|obey|apply)\b.{0,60}"
			r"\b(?:system|developer|previous|prior|safety)?\s*"
			r"(?:instructions?|rules?|policies)\b",
			"Tells the assistant not to follow its instructions.",
		),
		(
			r"\b(?:reveal|show|print|repeat|output|leak|expose|dump|share)\b.{0,40}"
			r"\b(?:system|developer|hidden|internal|initial|original)\b.{0,30}"
			r"\b(?:prompt|instructions?|message|rules?)\b",
			"Requests hidden system or developer instructions.",
		),
		(
			r"\b(?:what|tell|show|give|repeat|print)\b.{0,40}"
			r"\b(?:are|is|me|us|your|the)?\s*"
			r"(?:system|developer|hidden|internal)\s+"
			r"(?:prompt|instructions?|message|rules?)\b",
			"Requests disclosure of hidden prompts or instructions.",
		),
		(
			r"\b(?:act as|pretend to be|roleplay as|you are now|become)\b.{0,50}"
			r"\b(?:unrestricted|dan|jailbreak|developer mode|no restrictions|unfiltered)\b",
			"Attempts to assign an unrestricted or jailbreak persona.",
		),
		(
			r"\b(?:jailbreak|developer mode|do anything now|dan mode|"
			r"unrestricted mode|unfiltered mode|no safety rules)\b",
			"Contains common jailbreak or unrestricted-mode language.",
		),
		(
			r"\b(?:from now on|starting now)\b.{0,60}"
			r"\b(?:you must|you will|ignore|follow only|obey only)\b",
			"Attempts to replace the assistant's behavior with new instructions.",
		),
		(
			r"\b(?:reveal|print|show|send|return|expose)\b.{0,40}"
			r"\b(?:api keys?|credentials?|passwords?|secrets?|access tokens?)\b",
			"Requests disclosure of credentials or secrets.",
		),
		(
			r"\b(?:pretend|act)\b.{0,40}\b(?:rules|policies|safety)\b.{0,30}"
			r"\b(?:do not apply|dont apply|are disabled|are irrelevant)\b",
			"Attempts to disable or disregard safety rules.",
		),
	)
	DELIMITER_PATTERNS = (
		r"<\|(?:system|developer|assistant|im_start|im_end|fim_prefix|fim_middle)\|>",
		r"<\s*/?\s*(?:system|developer|assistant)\s*>",
		r"\[\s*/?\s*(?:inst|system)\s*\]",
		r"<<\s*/?\s*sys\s*>>",
		r"(?:^|\n)\s*###\s*(?:system|developer|assistant)\b",
		r"\b(?:system|developer|assistant)\s+to\s*=\s*[a-z0-9_-]+",
	)

	def __init__(self, max_length: int = 10_000) -> None:
		if max_length < 1:
			raise ValueError("max_length must be greater than zero")
		self.max_length = max_length
		self._compiled_injection_patterns = tuple(
			(re.compile(pattern, re.IGNORECASE), reason)
			for pattern, reason in self.INJECTION_PATTERNS
		)
		self._compiled_delimiter_patterns = tuple(
			re.compile(pattern, re.IGNORECASE | re.MULTILINE)
			for pattern in self.DELIMITER_PATTERNS
		)

	def check(self, text: str) -> tuple[bool, str | None]:
		if not isinstance(text, str):
			return False, "Input must be text."
		if len(text) > self.max_length:
			return False, f"Input exceeds the {self.max_length}-character limit."

		normalized = self._normalize(text)
		if not normalized:
			return False, "Input must not be empty."
		if len(normalized) > self.max_length:
			return False, f"Input exceeds the {self.max_length}-character limit."

		canonical = re.sub(r"[^a-z0-9]+", " ", normalized.casefold())
		for pattern, reason in self._compiled_injection_patterns:
			if pattern.search(canonical):
				return False, reason
		if any(pattern.search(normalized) for pattern in self._compiled_delimiter_patterns):
			return False, "Input contains model-role or instruction delimiters."
		return True, None

	def clean(self, text: str) -> str:
		if not isinstance(text, str):
			raise TypeError("Input must be text")
		if len(text) > self.max_length:
			raise ValueError(f"Input exceeds the {self.max_length}-character limit")
		cleaned = self._normalize(text)
		for pattern in self._compiled_delimiter_patterns:
			cleaned = pattern.sub(" ", cleaned)
		cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
		return cleaned

	@staticmethod
	def _normalize(text: str) -> str:
		normalized = unicodedata.normalize("NFKC", text)
		return "".join(
			character
			for character in normalized
			if character in "\n\r\t"
			or unicodedata.category(character) not in {"Cc", "Cf"}
		).strip()


class PIIDetector:
	REDACTION_MARKERS = {
		"email": "[EMAIL_REDACTED]",
		"phone": "[PHONE_REDACTED]",
		"ssn": "[SSN_REDACTED]",
		"card": "[CARD_REDACTED]",
	}
	PATTERNS = {
		"email": (
			r"(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
			r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
			r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+"
		),
		"ssn": r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}(?!\d)",
		"card": r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
		"phone": r"(?<!\w)\+?(?:\d[ .()\-]*){6,14}\d(?!\w)",
	}

	def __init__(self) -> None:
		self._compiled_patterns = {
			category: re.compile(
				pattern,
				re.IGNORECASE if category == "email" else 0,
			)
			for category, pattern in self.PATTERNS.items()
		}

	def detect(self, text: str) -> list[str]:
		if not isinstance(text, str):
			raise TypeError("Text to scan must be a string")
		normalized = unicodedata.normalize("NFKC", text)
		matches = {
			category: list(pattern.finditer(normalized))
			for category, pattern in self._compiled_patterns.items()
		}
		protected_spans = [
			match.span()
			for category in ("ssn", "card")
			for match in matches[category]
		]
		matches["phone"] = [
			match
			for match in matches["phone"]
			if not any(
				match.start() < end and start < match.end()
				for start, end in protected_spans
			)
		]
		return [category for category, found in matches.items() if found]

	def redact(self, text: str) -> str:
		if not isinstance(text, str):
			raise TypeError("Text to redact must be a string")
		redacted = unicodedata.normalize("NFKC", text)
		for category, pattern in self._compiled_patterns.items():
			redacted = pattern.sub(self.REDACTION_MARKERS[category], redacted)
		return redacted


class OutputValidator:
	BLOCKED_RESPONSE = "This response was blocked because it may contain harmful content."
	HARMFUL_PATTERNS = (
		(
			r"\b(?:how to|steps? to|instructions? (?:to|for)|guide to|recipe for)\b"
			r".{0,100}\b(?:make|build|assemble|construct|create|manufacture)\b"
			r".{0,60}\b(?:bomb|explosive|weapon|poison)\b",
			"actionable weapon or poison instructions",
		),
		(
			r"\b(?:how to|steps? to|instructions? (?:to|for)|ways to)\b"
			r".{0,80}\b(?:kill myself|suicide|self[ -]harm|overdose|cut myself)\b",
			"actionable self-harm instructions",
		),
		(
			r"\b(?:you should|go ahead and|the best way is to|best way to|i recommend)\b"
			r".{0,100}\b(?:kill yourself|hurt yourself|commit suicide|overdos(?:e|ing)|"
			r"overdoze|get yourself (?:killed|dead)|take too many (?:pills|medications?))\b",
			"encouragement of self-harm",
		),
		(
			r"\b(?:go\s+(?:and\s+)?)?(?:commit suicide|kill yourself|hurt yourself|"
			r"take your own life|end your life|overdose|overdoze)\b|"
			r"\bget yourself (?:killed|dead)\b",
			"direct self-harm encouragement",
		),
		(
			r"\b(?:how to|steps? to|instructions? (?:to|for))\b.{0,80}"
			r"\b(?:kill|poison|attack|shoot|hurt|harm)\b.{0,60}"
			r"\b(?:someone|a person|people|neighbor|coworker|family member)\b",
			"actionable instructions to harm another person",
		),
		(
			r"\b(?:steal|harvest|exfiltrate)\b.{0,40}"
			r"\b(?:credentials?|passwords?|access tokens?|private data)\b|"
			r"\b(?:deploy|create|write|install)\b.{0,40}"
			r"\b(?:ransomware|keylogger|credential stealer|phishing kit)\b",
			"actionable cyber abuse instructions",
		),
	)

	def __init__(self, pii_detector: PIIDetector) -> None:
		self.pii_detector = pii_detector
		self._compiled_harmful_patterns = tuple(
			(re.compile(pattern, re.IGNORECASE), reason)
			for pattern, reason in self.HARMFUL_PATTERNS
		)

	def validate(self, text: str) -> tuple[bool, str, list[str]]:
		if not isinstance(text, str):
			raise TypeError("Response to validate must be a string")
		for pattern, reason in self._compiled_harmful_patterns:
			if pattern.search(text):
				return (
					False,
					self.BLOCKED_RESPONSE,
					[
						"Response blocked: potentially harmful instructions were detected. "
						f"Category: {reason}."
					],
				)

		categories = self.pii_detector.detect(text)
		cleaned = self.pii_detector.redact(text)
		warnings = (
			["Personal information was detected and redacted from the response. "
			f"Categories: {', '.join(categories)}."]
			if categories
			else []
		)
		return True, cleaned, warnings


def _trace_input(inputs: dict[str, Any]) -> dict[str, Any]:
	text = inputs.get("text")
	return {"input_length": len(text) if isinstance(text, str) else 0}


def _trace_result(result: Any) -> dict[str, Any]:
	if isinstance(result, tuple) and len(result) == 3:
		return {"allowed": result[0], "security_notes": result[2]}
	return {"result": "unavailable"}


class SecurityPipeline:
	"""Coordinates input sanitization, PII redaction, and output validation."""

	def __init__(self, max_input_length: int = 10_000) -> None:
		self.sanitizer = InputSanitizer(max_input_length)
		self.pii_detector = PIIDetector()
		self.output_validator = OutputValidator(self.pii_detector)

	@traceable(
		name="security_check_input",
		process_inputs=_trace_input,
		process_outputs=_trace_result,
	)
	def check_input(self, text: str) -> tuple[bool, str, list[str]]:
		is_allowed, reason = self.sanitizer.check(text)
		if not is_allowed:
			return False, "", [reason or "Input was rejected by security checks."]

		cleaned = self.sanitizer.clean(text)
		categories = self.pii_detector.detect(cleaned)
		cleaned = self.pii_detector.redact(cleaned)
		notes = (
			[f"PII detected and redacted: {', '.join(categories)}."]
			if categories
			else []
		)
		return True, cleaned, notes

	@traceable(
		name="security_check_output",
		process_inputs=_trace_input,
		process_outputs=_trace_result,
	)
	def check_output(self, text: str) -> tuple[bool, str, list[str]]:
		return self.output_validator.validate(text)


security_pipeline = SecurityPipeline()
