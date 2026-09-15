"""LLM response-generation package (Phase 6) grounded on the Phase 5 retriever."""

from .llm_client import GroqLLMClient, LLMClient, LLMError, MockLLMClient, MOCK_RESPONSE
from .prompt_builder import SYSTEM_PROMPT, HISTORICAL_WARNING, build_prompt, build_user_prompt
from .response_generator import (
    FALLBACK_TEXT, GenerationResult, InputValidationError, ResponseGenerator,
    build_generator, validate_message,
)

__all__ = [
    "GroqLLMClient", "LLMClient", "LLMError", "MockLLMClient", "MOCK_RESPONSE",
    "SYSTEM_PROMPT", "HISTORICAL_WARNING", "build_prompt", "build_user_prompt",
    "FALLBACK_TEXT", "GenerationResult", "InputValidationError", "ResponseGenerator",
    "build_generator", "validate_message",
]
