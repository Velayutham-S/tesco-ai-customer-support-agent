"""
Phase 7 unit tests (no real Groq API calls).

Deterministic classification and escalation are tested directly; the workflow is tested with a
real ResponseGenerator wired to a FakeRetriever + fake LLM client. The CLI is tested via subprocess
in mock mode. No test needs a network, API key, or the original dataset.

Run:
  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.intent_taxonomy import PRIORITY_ORDER
from src.generation import FALLBACK_TEXT, InputValidationError, ResponseGenerator
from src.generation.llm_client import LLMError
from src.agent import IntentClassifier, TescoSupportAgent, decide_escalation, build_agent
from src.agent.models import AgentResult

SECRET_DUMMY_KEY = "gsk_dummy_SECRET_DO_NOT_LEAK_98765"


# --------------------------------------------------------------------- fakes
def make_examples(n: int = 3) -> list[dict]:
    return [{
        "rank": i, "interaction_id": f"tesco-X{i}", "conversation_id": f"c{i}",
        "brand_tweet_id": 100 + i,
        "customer_text": f"@11{i} my delivery is late", "customer_redacted_text": f"my delivery is late {i}",
        "brand_text": "@t sorry", "brand_redacted_text": "we are sorry, please DM us",
        "intent": "delivery_issue", "retrieval_score": round(0.6 - 0.05 * i, 4),
    } for i in range(1, n + 1)]


class FakeRetriever:
    def __init__(self, examples=None):
        self._examples = examples if examples is not None else make_examples()

    def retrieve(self, query, top_k=5, exclude_interaction_id=None):
        return self._examples[:top_k]


class BadRetriever:
    def retrieve(self, *a, **k):
        raise RuntimeError("index unavailable")


class EchoClient:
    model_name = "fake-echo"

    def generate(self, system_prompt, user_prompt):
        return "Thanks for getting in touch - here is a helpful reply."


class FailingClient:
    model_name = "fake-fail"

    def generate(self, system_prompt, user_prompt):
        raise LLMError("Groq request failed (RuntimeError)")


class FakeGroqIntent:
    model_name = "fake-groq"

    def __init__(self, payload: str):
        self.payload = payload

    def generate(self, system_prompt, user_prompt):
        return self.payload


def build_test_agent(retriever=None, client=None, mode="mock"):
    rg = ResponseGenerator(retriever or FakeRetriever(), client or EchoClient())
    return TescoSupportAgent(rg, IntentClassifier(), mode=mode, model_name="fake-echo")


# ----------------------------------------------------------------- validation
class TestInputValidation(unittest.TestCase):
    def test_01_empty_rejected(self):
        with self.assertRaises(InputValidationError):
            build_test_agent().handle("")

    def test_02_non_string_rejected(self):
        with self.assertRaises(InputValidationError):
            build_test_agent().handle(123)


# ----------------------------------------------------------- intent mapping
class TestIntentMapping(unittest.TestCase):
    def setUp(self):
        self.c = IntentClassifier()

    def test_03_delivery(self):
        self.assertEqual(self.c.classify("My delivery has not arrived").intent, "delivery_issue")

    def test_04_missing_item(self):
        self.assertEqual(self.c.classify("half my items were missing from my order").intent,
                         "missing_or_incorrect_item")

    def test_05_duplicate_charge(self):
        self.assertEqual(self.c.classify("I was charged twice for my order").intent, "refund_or_payment")

    def test_06_clubcard(self):
        self.assertEqual(self.c.classify("my clubcard points are missing").intent, "clubcard_or_loyalty")

    def test_07_product_stock(self):
        self.assertEqual(self.c.classify("is the gluten free bread back in stock").intent,
                         "product_availability")

    def test_08_store_queue(self):
        self.assertEqual(self.c.classify("huge queue at the till in your store").intent, "store_experience")

    def test_09_ambiguous_low_conf(self):
        pred = self.c.classify("asdfghjkl qwerty zzz")
        self.assertTrue(pred.intent == "other_or_unclear" or pred.confidence < 0.55)

    def test_10_labels_in_taxonomy(self):
        msgs = ["My delivery is late", "refund please", "clubcard points", "is milk in stock",
                "queue at store", "random gibberish", "I can't log into my account"]
        for m in msgs:
            self.assertIn(self.c.classify(m).intent, PRIORITY_ORDER)

    def test_11_confidence_range(self):
        for m in ["My delivery is late", "asdfghjkl", "refund for double charge"]:
            conf = self.c.classify(m).confidence
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)


# ------------------------------------------------------- groq fallback safety
class TestGroqFallback(unittest.TestCase):
    def test_12_invalid_json_falls_back(self):
        c = IntentClassifier(FakeGroqIntent("this is not json"), use_groq=True)
        self.assertEqual(c.classify("asdfghjkl qwerty").intent, "other_or_unclear")

    def test_13_unknown_label_falls_back(self):
        c = IntentClassifier(FakeGroqIntent('{"intent":"not_a_label","confidence":0.9,"evidence":"x"}'),
                             use_groq=True)
        self.assertEqual(c.classify("asdfghjkl qwerty").intent, "other_or_unclear")

    def test_12b_valid_groq_used(self):
        c = IntentClassifier(FakeGroqIntent('{"intent":"delivery_issue","confidence":0.9,"evidence":"x"}'),
                             use_groq=True)
        pred = c.classify("asdfghjkl qwerty")  # low deterministic -> groq consulted
        self.assertEqual(pred.intent, "delivery_issue")
        self.assertEqual(pred.method, "groq")

    def test_14_deterministic_only_needs_no_key(self):
        # No client, no key: deterministic classification must still work.
        self.assertIn(IntentClassifier().classify("My delivery is late").intent, PRIORITY_ORDER)


# ----------------------------------------------------------------- escalation
class TestEscalation(unittest.TestCase):
    def setUp(self):
        self.c = IntentClassifier()

    def _esc(self, msg):
        return decide_escalation(msg, self.c.classify(msg))

    def test_15_human_request(self):
        self.assertTrue(self._esc("I want to speak to a real person").needs_human_review)

    def test_16_payment_issue(self):
        self.assertTrue(self._esc("please refund the duplicate charge").needs_human_review)

    def test_17_live_order_request(self):
        self.assertTrue(self._esc("My delivery has not arrived").needs_human_review)

    def test_18_info_request_no_review(self):
        # A normal product-availability question should not trigger human review.
        self.assertFalse(self._esc("Is bread available?").needs_human_review)


# ------------------------------------------------------------------ workflow
class TestWorkflow(unittest.TestCase):
    def test_19_retrieval_failure_safe(self):
        agent = build_test_agent(retriever=BadRetriever())
        res = agent.handle("My delivery has not arrived")
        self.assertEqual(res.retrieved_count, 0)
        self.assertTrue(res.generation_success)  # no-context generation still works

    def test_20_generation_fallback_preserved(self):
        agent = build_test_agent(client=FailingClient())
        res = agent.handle("My delivery has not arrived")
        self.assertTrue(res.used_fallback)
        self.assertEqual(res.generated_response, FALLBACK_TEXT)
        self.assertEqual(res.error_type, "llm_error")

    def test_21_result_json_serializable(self):
        res = build_test_agent().handle("My delivery has not arrived")
        self.assertIsInstance(res, AgentResult)
        parsed = json.loads(res.to_json())
        self.assertEqual(parsed["intent"], "delivery_issue")
        self.assertIn("workflow_steps", parsed)

    def test_24_no_secret_in_output(self):
        res = build_test_agent().handle("My delivery has not arrived")
        blob = res.to_json().lower()
        self.assertNotIn("gsk_", blob)
        self.assertNotIn("api_key", blob)


# ----------------------------------------------------------------------- CLI
class TestCLI(unittest.TestCase):
    def _run(self, *extra):
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "run_agent.py"),
             "--message", "My delivery has not arrived", "--mock", *extra],
            cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8")

    def test_22_cli_mock(self):
        proc = self._run()
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("delivery_issue", proc.stdout)
        self.assertNotIn(SECRET_DUMMY_KEY, proc.stdout)

    def test_23_cli_json_valid(self):
        proc = self._run("--json")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["intent"], "delivery_issue")
        self.assertTrue(data["needs_human_review"])


class TestProductAvailabilityFix(unittest.TestCase):
    """Regression tests for the product_availability keyword fix (available/availability/unavailable)."""

    def setUp(self):
        self.c = IntentClassifier()

    def _res(self, msg):
        return decide_escalation(msg, self.c.classify(msg))

    def test_available_maps_to_product_availability(self):
        self.assertEqual(self.c.classify("Is bread available?").intent, "product_availability")

    def test_back_in_stock_maps_to_product_availability(self):
        self.assertEqual(self.c.classify("When will this product be back in stock?").intent,
                         "product_availability")

    def test_unavailable_maps_to_product_availability(self):
        self.assertEqual(self.c.classify("This item is unavailable").intent, "product_availability")

    def test_availability_questions_no_human_review(self):
        for msg in ["Is bread available?", "When will this product be back in stock?",
                    "This item is unavailable"]:
            self.assertFalse(self._res(msg).needs_human_review, msg=msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
