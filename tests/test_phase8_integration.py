"""
Phase 8 end-to-end integration tests (mock generation only; no real Groq API call).

The full agent is built in mock mode (which loads the real retrieval index but uses the offline
mock LLM). A few tests check config/credential behaviour without contacting Groq.

Run:
  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, GroqSettings
from src.intent_taxonomy import PRIORITY_ORDER
from src.generation import InputValidationError, MOCK_RESPONSE
from src.generation.llm_client import GroqLLMClient
from src.agent import build_agent
from src.agent.models import AgentResult

EXPECTED_LABELS = [
    "clubcard_or_loyalty", "missing_or_incorrect_item", "delivery_issue", "online_order_issue",
    "product_availability", "refund_or_payment", "account_or_technical_issue", "store_experience",
    "complaint", "general_information", "other_or_unclear",
]


class TestPhase8Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = build_agent(mock=True)  # real retriever index + offline mock LLM

    # 1
    def test_01_end_to_end_mock(self):
        res = self.agent.handle("My delivery has not arrived")
        self.assertIsInstance(res, AgentResult)
        self.assertTrue(res.generated_response.strip())

    # 2
    def test_02_delivery_intent(self):
        self.assertEqual(self.agent.handle("My delivery has not arrived").intent, "delivery_issue")

    # 3
    def test_03_product_availability_intent(self):
        self.assertEqual(self.agent.handle("Is bread available?").intent, "product_availability")

    # 4
    def test_04_refund_intent(self):
        self.assertEqual(self.agent.handle("I was charged twice for my order").intent,
                         "refund_or_payment")

    # 5
    def test_05_account_intent(self):
        self.assertEqual(self.agent.handle("I cannot log into my account").intent,
                         "account_or_technical_issue")

    # 6
    def test_06_human_escalation(self):
        self.assertTrue(self.agent.handle("I want to speak to a human").needs_human_review)

    # 7
    def test_07_legal_escalation(self):
        self.assertTrue(self.agent.handle("I will take legal action against Tesco").needs_human_review)

    # 8
    def test_08_low_confidence_unclear(self):
        res = self.agent.handle("asdfghjkl qwerty zzz")
        self.assertTrue(res.intent == "other_or_unclear" or res.intent_confidence < 0.55)
        self.assertTrue(res.needs_human_review)

    # 9
    def test_09_retrieval_included(self):
        self.assertGreaterEqual(self.agent.handle("My delivery has not arrived").retrieved_count, 1)

    # 10
    def test_10_response_non_empty_mock(self):
        self.assertEqual(self.agent.handle("My delivery has not arrived").generated_response,
                         MOCK_RESPONSE)

    # 11
    def test_11_json_serializable(self):
        parsed = json.loads(self.agent.handle("My delivery has not arrived").to_json())
        self.assertEqual(parsed["intent"], "delivery_issue")
        self.assertIn("needs_human_review", parsed)

    # 12
    def test_12_empty_message_rejected(self):
        with self.assertRaises(InputValidationError):
            self.agent.handle("   ")

    # 13
    def test_13_missing_index_handled(self):
        with self.assertRaises(FileNotFoundError):
            build_agent(mock=True, index_dir=str(REPO_ROOT / "data" / "_no_such_index_"))

    # 14
    def test_14_mock_needs_no_api_key(self):
        saved = os.environ.pop("GROQ_API_KEY", None)
        try:
            agent = build_agent(mock=True)
            self.assertTrue(agent.handle("My delivery has not arrived").generated_response.strip())
        finally:
            if saved is not None:
                os.environ["GROQ_API_KEY"] = saved

    # 15
    def test_15_real_mode_missing_credentials(self):
        with self.assertRaises(ConfigError):
            GroqLLMClient(GroqSettings(api_key=None, model="openai/gpt-oss-20b"))

    # 16
    def test_16_taxonomy_preserved(self):
        self.assertEqual(list(PRIORITY_ORDER), EXPECTED_LABELS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
