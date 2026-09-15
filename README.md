# Tesco AI Customer-Support Agent

A focused, explainable prototype that turns a real customer-support tweet dataset into an AI agent
for one brand (**Tesco**): it classifies the customer's intent, retrieves how Tesco historically
handled similar messages, drafts a grounded reply with an LLM, and recommends whether a human should
review the case. Built as a simple, reproducible Python CLI - no web UI, database, or cloud services.

## 1. Problem statement
Given noisy real-world customer-support conversations, build an agent that (a) classifies an incoming
customer message into a small intent set, (b) drafts a reply grounded in the brand's historical
responses, and (c) decides whether to auto-handle or recommend human review - and, crucially, shows
evidence that it works.

## 2. Dataset
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
(`dataset/twcs/twcs.csv`, ~2.81M tweets). It is **historical** and is never modified by this project.

## 3. Why Tesco was selected
Chosen by a transparent, weighted score over 25 candidate brands (not by raw volume). Tesco offered a
strong balance of substantive (non-"DM us") replies, high multi-turn rate, good issue diversity,
sufficient usable pairs for splits + a golden set, and a corpus small enough to reproduce quickly.
See `reports/brand_selection_report.md`.

## 4. System architecture
```
customer message
   -> validate
   -> intent classification (deterministic rules + optional Groq fallback)
   -> escalation recommendation (transparent rules)
   -> retrieval of similar historical Tesco interactions (TF-IDF)
   -> grounded response generation (Groq, or mock)
   -> structured AgentResult (human-readable + JSON)
```

## 5. Main components
- `src/config.py` - central config + Groq settings (`.env`).
- `src/text_utils.py` - normalization, redaction, generic-reply heuristic.
- `src/intent_taxonomy.py` - the 11 intents + keyword rules (single source of truth).
- `src/retrieval/` - `TescoRetriever` (TF-IDF index build + query).
- `src/generation/` - Groq client, prompt builder, response generator (mock + live, safe fallback).
- `src/agent/` - intent classifier, escalation policy, workflow (`TescoSupportAgent`), `AgentResult`.
- `scripts/` - data prep, index build, evaluation, demo, and the agent CLI.

## 6. Intent taxonomy
`clubcard_or_loyalty`, `missing_or_incorrect_item`, `delivery_issue`, `online_order_issue`,
`product_availability`, `refund_or_payment`, `account_or_technical_issue`, `store_experience`,
`complaint`, `general_information`, `other_or_unclear`. (See `reports/tesco_intent_taxonomy.md`.)

## 7. Retrieval approach
**TF-IDF lexical retrieval** (scikit-learn) with cosine similarity over ~24.9k unique historical Tesco
customer messages. Deterministic ranking with a stable tie-break. It is lexical, not semantic
(no embeddings). See `reports/phase5_retrieval_report.md`.

## 8. Response generation approach
**Groq-based generation** (default `openai/gpt-oss-20b`) with a grounded system prompt that uses the
retrieved historical replies as guidance only, forbids inventing policies/actions, and asks for
clarification when needed. A **mock mode** returns a deterministic placeholder for offline runs, and a
safe fallback message is returned on any failure. See `reports/phase6_generation_report.md`.

## 9. Escalation policy
A transparent, **recommendation-only** policy (it never creates tickets or contacts staff). It flags
human review for: explicit human requests, legal/safety concerns, payment/refund issues, account-access
issues, unresolved/repeated problems, order-data requests, and low-confidence/unclear messages. Each
decision carries a short, observable reason.

## 10. Deterministic intent classification
Rule-based first (keyword rules from the taxonomy, re-ranked by an agent-specific priority for
overlaps), with a heuristic confidence score. An optional Groq JSON fallback is consulted only for
low-confidence messages in live mode, and any failure safely falls back to `other_or_unclear`.

## 11. Project structure
```
dataset/twcs/twcs.csv              # raw dataset (unmodified)
data/processed/                    # tesco_interactions.*, tesco_golden_set.*
data/retrieval/                    # TF-IDF index artifacts
src/{config,text_utils,intent_taxonomy}.py
src/{retrieval,generation,agent}/
scripts/                           # build/eval/demo/CLI scripts
tests/                             # unittest suites
reports/                           # phase reports + evaluations
```

## 12. Setup instructions
- Python 3.11+ (developed on 3.11).
- `pip install -r requirements.txt`
- The processed corpus, golden set, and retrieval index are already in `data/`. To rebuild:
  ```bash
  python scripts/build_tesco_corpus.py --dataset dataset/twcs/twcs.csv --output-dir data/processed
  python scripts/build_retrieval_index.py --corpus data/processed/tesco_interactions.parquet --output-dir data/retrieval
  ```

## 13. Environment variables
Copy `.env.example` to `.env` (git-ignored; never commit it):
```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```
`GROQ_API_KEY` is only needed for live mode. `llama-3.3-70b-versatile` is decommissioned on Groq; use an
available model such as `openai/gpt-oss-20b`.

## 14. Mock mode usage (no API key)
```bash
python scripts/run_agent.py --message "My delivery has not arrived" --mock
python scripts/run_agent.py --message "My delivery has not arrived" --mock --json
python scripts/run_agent.py --interactive --mock
```

## 15. Real Groq mode usage (needs GROQ_API_KEY)
```bash
python scripts/run_agent.py --message "My delivery has not arrived" --top-k 5
```
Without a key, real mode fails with a clear message (use `--mock`).

## 16. Evaluation command
```bash
python scripts/evaluate_agent.py --mock      # -> reports/phase8_agent_evaluation.{json,md}
```

## 17. Demo command
```bash
python scripts/demo_agent.py --mock
```

## 18. Test command
```bash
python -m unittest discover -s tests -v
python scripts/validate_retrieval.py
```

## 19. Example output (mock)
```
=== Customer-facing response ===
[MOCK RESPONSE] A customer-support response would be generated here.
=== Agent diagnostics ===
intent             : delivery_issue
intent_confidence  : 0.88
retrieved_count    : 5
needs_human_review : True
escalation_reason  : Request requires live order information.
mode               : mock
```
(In live mode the response is a real grounded reply from Groq.)

## 20. Limitations
- The dataset is **historical**; retrieved replies are **not** guaranteed to reflect current Tesco policy.
- Golden-set intent labels are **provisional** (rule-based, not human-verified).
- Retrieval is **TF-IDF lexical** (no semantic embeddings); it can miss paraphrases.
- Intent confidence is a **heuristic**, not a calibrated probability.
- The system has **no** access to live Tesco orders, accounts, payments, or delivery systems.
- It **recommends** escalation but does **not** perform escalation, refunds, cancellations, or any action.
- Generated replies depend on the retrieved examples and the selected LLM; factual correctness is **not** guaranteed and was not human-reviewed.
- This is a **prototype**, not a production deployment.

## 21. Future improvements
- Human-review a sample of retrievals/replies to calibrate the intent-proxy and add an LLM-as-judge with human agreement.
- Compare the TF-IDF baseline against an embedding retriever on the same golden set.
- Curate the provisional golden labels into human-verified ground truth.
- Broaden/curate intent keyword rules and confidence calibration.

---
Reports for each phase are in `reports/` (dataset profile, brand selection, corpus, intents, golden set,
retrieval, generation, agent workflow, and this phase's evaluation + final report).
