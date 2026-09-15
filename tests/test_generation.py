"""
Phase 6 unit tests (no real Groq API calls).

Uses a FakeRetriever and fake LLM clients injected into ResponseGenerator, plus direct calls to the
prompt builder / validation / config. The CLI mock test runs the script as a subprocess.

Run:
  python -m unittest discover -s tests -v
  (or: python -m pytest tests -q)
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, GroqSettings
from src.generation import (
    FALLBACK_TEXT, GenerationResult, GroqLLMClient, InputValidationError, MockLLMClient,
    MOCK_RESPONSE, ResponseGenerator, build_user_prompt, validate_message,
)
from src.generation.prompt_builder import HISTORICAL_WARNING

SECRET_DUMMY_KEY = "sk_dummy_SECRET_DO_NOT_LEAK_123456"


def make_examples(n: int = 3) -> list[dict]:
    out = []
    for i in range(1, n + 1):
        out.append({
            "rank": i,
            "interaction_id": f"tesco-SECRETID{i}",
            "conversation_id": f"tesco-conv-{i}",
            "brand_tweet_id": 1000 + i,
            "customer_text": f"@98765{i} original customer text {i}",
            "customer_redacted_text": f"@98765{i} my delivery was late number {i}",
            "brand_text": f"@111{i} original reply {i}",
            "brand_redacted_text": f"@Tesco we are sorry to hear this, please DM us detail {i}",
            "intent": "delivery_issue",
            "retrieval_score": round(0.5 - i * 0.05, 4),
        })
    return out


class FakeRetriever:
    """Minimal stand-in for TescoRetriever; records calls."""

    def __init__(self, examples: list[dict] | None = None) -> None:
        self._examples = examples if examples is not None else make_examples()
        self.called = False
        self.last_query = None
        self.last_top_k = None

    def retrieve(self, query: str, top_k: int = 5, exclude_interaction_id=None) -> list[dict]:
        self.called = True
        self.last_query = query
        self.last_top_k = top_k
        return self._examples[:top_k]


class EchoClient:
    model_name = "fake-echo"

    def __init__(self) -> None:
        self.last_system = None
        self.last_user = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system = system_prompt
        self.last_user = user_prompt
        return "Thanks for reaching out - here is a helpful reply."


class FailingClient:
    model_name = "fake-fail"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        from src.generation import LLMError
        raise LLMError("Groq request failed (RuntimeError)")


class EmptyClient:
    model_name = "fake-empty"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "   "


class TestValidation(unittest.TestCase):
    def test_01_none_rejected(self):
        with self.assertRaises(InputValidationError):
            validate_message(None)

    def test_02_non_string_rejected(self):
        with self.assertRaises(InputValidationError):
            validate_message(12345)

    def test_03_empty_rejected(self):
        with self.assertRaises(InputValidationError):
            validate_message("")

    def test_04_whitespace_rejected(self):
        with self.assertRaises(InputValidationError):
            validate_message("    \n\t ")

    def test_04b_generator_raises_on_bad_input(self):
        gen = ResponseGenerator(FakeRetriever(), EchoClient())
        with self.assertRaises(InputValidationError):
            gen.generate(None)


class TestPrompt(unittest.TestCase):
    def test_05_contains_customer_message(self):
        prompt = build_user_prompt("my delivery never arrived", make_examples())
        self.assertIn("my delivery never arrived", prompt)

    def test_06_contains_historical_reply(self):
        prompt = build_user_prompt("hi", make_examples())
        self.assertIn("please DM us detail", prompt)  # from brand_redacted_text

    def test_07_contains_warning(self):
        prompt = build_user_prompt("hi", make_examples())
        self.assertIn(HISTORICAL_WARNING, prompt)

    def test_08_limits_examples(self):
        prompt = build_user_prompt("hi", make_examples(10), max_examples=5)
        self.assertIn("Example 5:", prompt)
        self.assertNotIn("Example 6:", prompt)

    def test_09_no_sensitive_metadata(self):
        prompt = build_user_prompt("hi", make_examples(3))
        # interaction ids / conversation ids / handles must not leak into the prompt
        self.assertNotIn("tesco-SECRETID", prompt)
        self.assertNotIn("tesco-conv-", prompt)
        self.assertNotIn("@98765", prompt)   # customer handle scrubbed
        self.assertNotIn("@Tesco", prompt)   # brand handle scrubbed


class TestClients(unittest.TestCase):
    def test_10_mock_client_works(self):
        self.assertEqual(MockLLMClient().generate("s", "u"), MOCK_RESPONSE)


class TestGenerator(unittest.TestCase):
    def test_11_calls_retriever(self):
        retr = FakeRetriever()
        gen = ResponseGenerator(retr, EchoClient(), top_k=3)
        gen.generate("where is my order")
        self.assertTrue(retr.called)
        self.assertEqual(retr.last_query, "where is my order")
        self.assertEqual(retr.last_top_k, 3)

    def test_12_returns_structured_result(self):
        gen = ResponseGenerator(FakeRetriever(), EchoClient())
        res = gen.generate("hello")
        self.assertIsInstance(res, GenerationResult)
        self.assertTrue(res.success)
        self.assertFalse(res.used_fallback)
        self.assertEqual(res.model_name, "fake-echo")
        self.assertEqual(res.retrieved_count, 3)
        self.assertIsNotNone(res.top_similarity)

    def test_13_llm_failure_fallback(self):
        gen = ResponseGenerator(FakeRetriever(), FailingClient())
        res = gen.generate("hello")
        self.assertTrue(res.used_fallback)
        self.assertFalse(res.success)
        self.assertEqual(res.error_type, "llm_error")
        self.assertEqual(res.response_text, FALLBACK_TEXT)

    def test_15_empty_response_fallback(self):
        gen = ResponseGenerator(FakeRetriever(), EmptyClient())
        res = gen.generate("hello")
        self.assertTrue(res.used_fallback)
        self.assertEqual(res.error_type, "empty_response")
        self.assertEqual(res.response_text, FALLBACK_TEXT)

    def test_retrieval_failure_degrades(self):
        class BadRetriever:
            def retrieve(self, *a, **k):
                raise RuntimeError("index boom")
        gen = ResponseGenerator(BadRetriever(), EchoClient())
        res = gen.generate("hello")  # should not crash; no-context path
        self.assertTrue(res.success)
        self.assertEqual(res.retrieved_count, 0)


class TestConfigAndSecrets(unittest.TestCase):
    def test_14_missing_api_key_raises(self):
        settings = GroqSettings(api_key=None, model="llama-3.3-70b-versatile")
        self.assertFalse(settings.has_api_key)
        with self.assertRaises(ConfigError):
            settings.require_api_key()

    def test_17_api_key_not_in_error(self):
        # Build a live client with a dummy key, then force the underlying client to raise.
        client = GroqLLMClient(GroqSettings(api_key=SECRET_DUMMY_KEY, model="m"))

        class _Completions:
            def create(self, **kwargs):
                raise RuntimeError(f"network blew up with {SECRET_DUMMY_KEY}")

        class _Chat:
            def __init__(self):
                self.completions = _Completions()

        class _FakeGroq:
            def __init__(self):
                self.chat = _Chat()

        client._client = _FakeGroq()
        from src.generation import LLMError
        with self.assertRaises(LLMError) as ctx:
            client.generate("s", "u")
        self.assertNotIn(SECRET_DUMMY_KEY, str(ctx.exception))


class TestCLIMock(unittest.TestCase):
    def test_16_cli_mock_mode(self):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "generate_response.py"),
             "--message", "my delivery has not arrived", "--mock"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn(MOCK_RESPONSE, proc.stdout)
        self.assertNotIn(SECRET_DUMMY_KEY, proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
