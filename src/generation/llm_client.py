"""LLM clients for Phase 6 generation.

This module is responsible ONLY for talking to the model. It performs no retrieval, no business
logic, and no CLI formatting. A small `LLMClient` protocol lets tests inject a fake client and
lets the app swap the real Groq client for a mock without touching business code.

Security: the API key is never logged, never printed, and never included in error messages.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

try:
    from src.config import GroqSettings
except ImportError:  # pragma: no cover
    from config import GroqSettings  # type: ignore

MOCK_RESPONSE = "[MOCK RESPONSE] A customer-support response would be generated here."

# Conservative, deterministic-ish generation defaults.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 350
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 1


class LLMError(RuntimeError):
    """Raised when the model cannot produce a usable response. Contains no secrets."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every client (real or fake) must satisfy."""

    model_name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text response, or raise LLMError."""
        ...


class MockLLMClient:
    """Offline client for tests and local runs. Never contacts Groq."""

    def __init__(self, model_name: str = "mock") -> None:
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return MOCK_RESPONSE


class GroqLLMClient:
    """Thin wrapper over the official Groq SDK. Bounded, low-temperature generation."""

    def __init__(self, settings: GroqSettings,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        # require_api_key() raises a clear, secret-free ConfigError when the key is missing.
        api_key = settings.require_api_key()
        self.model_name = settings.model
        self.temperature = temperature
        self.max_tokens = max_tokens
        try:
            import groq  # lazy import so mock mode never needs the SDK
        except ImportError as exc:  # pragma: no cover
            raise LLMError("groq SDK is not installed (pip install groq)") from exc
        # timeout + limited retries live on the client; no key material is stored elsewhere.
        self._client = groq.Groq(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = (completion.choices[0].message.content or "").strip()
        except Exception as exc:  # network/API/timeout errors
            # Only the exception *type* is surfaced - never the message (which could echo input).
            raise LLMError(f"Groq request failed ({type(exc).__name__})") from None
        if not text:
            raise LLMError("Groq returned an empty response")
        return text
