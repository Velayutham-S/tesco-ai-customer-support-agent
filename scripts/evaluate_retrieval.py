"""
Phase 5 (Parts 5-6, 8) - Evaluate the TF-IDF retriever on the golden set + write the report.

For every golden example we query the retriever with the customer message (excluding the query's
own indexed message), retrieve top-k, and measure an INTENT-PROXY: does a retrieved historical
interaction share the query's (provisional) intent? We report Hit Rate@1/3/5/10 and MRR, plus
counts of skipped/empty/no-result queries, and >=10 qualitative examples.

IMPORTANT: intent labels are PROVISIONAL (rule-based) and intent-match is a PROXY for relevance,
not human-judged semantic relevance. See the limitations section of the report.

Usage:
  python scripts/evaluate_retrieval.py --golden data/processed/tesco_golden_set.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.text_utils import normalize_text
from src.retrieval.retriever import TescoRetriever, DEFAULT_INDEX_DIR

K_VALUES = [1, 3, 5, 10]
TOP_K = 10


def _stdout_utf8_safe() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def run_eval(retriever: TescoRetriever, golden: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Query the retriever for each golden row and compute intent-proxy metrics."""
    # map normalized customer text -> indexed interaction_id, to exclude the query's own message
    norm_to_id = dict(zip(retriever.metadata["customer_normalized_text"].astype(str),
                          retriever.metadata["interaction_id"].astype(str)))

    per_example: list[dict] = []
    skipped_empty = 0
    no_result = 0

    for _, row in golden.iterrows():
        query = str(row["customer_text"])
        q_intent = str(row["intent"])
        norm = normalize_text(query)
        if not norm.strip():
            skipped_empty += 1
            continue
        exclude_id = norm_to_id.get(norm)
        results = retriever.retrieve(query, top_k=TOP_K, exclude_interaction_id=exclude_id)
        if not results:
            no_result += 1

        first_match_rank = None
        for r in results:
            if r["intent"] == q_intent:
                first_match_rank = r["rank"]
                break

        per_example.append({
            "golden_id": str(row["golden_id"]),
            "query_intent": q_intent,
            "query_redacted": str(row["customer_redacted_text"]),
            "is_short": bool(row.get("is_short", False)),
            "is_ambiguous": bool(row.get("is_ambiguous", False)),
            "is_multi_turn": bool(row.get("is_multi_turn", False)),
            "has_url": bool(row.get("has_url", False)),
            "n_results": len(results),
            "top1_score": results[0]["retrieval_score"] if results else 0.0,
            "first_match_rank": first_match_rank,
            "results": results[:5],  # keep top-5 detail for qualitative report
        })

    evaluated = len(per_example)

    def hit_rate(k: int) -> float:
        if evaluated == 0:
            return 0.0
        hits = sum(1 for e in per_example
                   if e["first_match_rank"] is not None and e["first_match_rank"] <= k)
        return round(hits / evaluated, 4)

    mrr = round(sum(1.0 / e["first_match_rank"] for e in per_example
                    if e["first_match_rank"] is not None) / evaluated, 4) if evaluated else 0.0
    mean_top1 = round(sum(e["top1_score"] for e in per_example) / evaluated, 4) if evaluated else 0.0

    agg = {
        "golden_set_size": int(len(golden)),
        "evaluated_examples": evaluated,
        "skipped_empty_query": skipped_empty,
        "empty_query_examples": skipped_empty,
        "no_nonzero_result_queries": no_result,
        "top_k": TOP_K,
        "hit_rate": {f"@{k}": hit_rate(k) for k in K_VALUES},
        "mrr": mrr,
        "mean_top1_score": mean_top1,
        "metric_type": "intent-proxy (provisional labels); NOT human-judged relevance",
    }
    return agg, per_example


def pick_qualitative(per_example: list[dict], n_min: int = 10) -> list[dict]:
    """Deterministically select >=10 examples covering good/weak/ambiguous/short/url/multi/no-result."""
    by_id = sorted(per_example, key=lambda e: e["golden_id"])
    chosen: dict[str, str] = {}  # golden_id -> category label

    def add(cands: list[dict], label: str, k: int = 2) -> None:
        cnt = 0
        for e in cands:
            if cnt >= k:
                break
            if e["golden_id"] not in chosen:
                chosen[e["golden_id"]] = label
                cnt += 1

    with_results = [e for e in by_id if e["n_results"] > 0]
    add(sorted(with_results, key=lambda e: (-e["top1_score"], e["golden_id"])), "good_high_similarity", 2)
    add(sorted(with_results, key=lambda e: (e["top1_score"], e["golden_id"])), "weak_low_similarity", 2)
    add([e for e in by_id if e["first_match_rank"] is None and e["n_results"] > 0], "weak_no_intent_match", 2)
    add([e for e in by_id if e["is_ambiguous"]], "ambiguous_query", 2)
    add([e for e in by_id if e["is_short"]], "short_query", 2)
    add([e for e in by_id if e["has_url"]], "url_bearing_query", 1)
    add([e for e in by_id if e["is_multi_turn"]], "multi_turn_query", 1)
    add([e for e in by_id if e["n_results"] == 0], "no_result_query", 1)
    # fill to n_min with a spread
    if len(chosen) < n_min:
        add(with_results, "additional", n_min - len(chosen))

    out = []
    for e in by_id:
        if e["golden_id"] in chosen:
            e2 = dict(e)
            e2["category"] = chosen[e["golden_id"]]
            out.append(e2)
    return out


def _explain(e: dict) -> str:
    if e["n_results"] == 0:
        return "No lexical overlap with any indexed message (TF-IDF found nothing) - a retrieval miss."
    top = e["results"][0]
    matched = e["first_match_rank"] is not None
    if matched and e["first_match_rank"] == 1 and top["retrieval_score"] >= 0.3:
        return "Strong lexical match; top result shares the provisional intent."
    if matched and e["first_match_rank"] == 1:
        return "Top result shares the provisional intent, but similarity is modest (short/varied wording)."
    if matched:
        return f"An intent-sharing result appears at rank {e['first_match_rank']} (not the top hit)."
    return "No retrieved result shares the provisional intent - likely lexical-but-not-topical overlap."


def write_report(agg: dict, per_example: list[dict], qualitative: list[dict],
                 index_meta: dict, reports_dir: Path, golden_path: str, index_dir: str) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    # JSON (aggregate + compact per-example + selected qualitative)
    compact = [{k: e[k] for k in ("golden_id", "query_intent", "n_results", "top1_score",
                                  "first_match_rank", "is_short", "is_ambiguous",
                                  "is_multi_turn", "has_url")} for e in per_example]
    eval_json = {
        "brand": config.SELECTED_BRAND,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "index_metadata": index_meta,
        "aggregate": agg,
        "per_example": compact,
        "qualitative_example_ids": [e["golden_id"] for e in qualitative],
    }
    json_path = reports_dir / "phase5_retrieval_eval.json"
    json_path.write_text(json.dumps(eval_json, indent=2), encoding="utf-8")

    hr = agg["hit_rate"]
    L: list[str] = []
    a = L.append
    a("# Phase 5 - Local Retrieval Baseline and Evaluation")
    a("")
    a(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `scripts/evaluate_retrieval.py`._")
    a("")
    a("## 1. Objective")
    a("Provide a simple, local, deterministic **TF-IDF retriever** that returns historical Tesco "
      "customer-support interactions similar to a new customer message, and evaluate it on the "
      "141-example golden set with an intent-proxy metric. No response generation, no LLM, no API.")
    a("## 2. Input corpus")
    a(f"- Source: `{index_meta.get('source_corpus_path')}`")
    a(f"- Indexed rows (unique customer messages): **{index_meta.get('num_indexed_rows'):,}** "
      f"(deduplicated from usable pairs; {index_meta.get('duplicate_text_rows_removed'):,} duplicate-text rows removed)")
    a(f"- Retrieval text column: `{index_meta.get('retrieval_text_column')}` (the historical customer message)")
    a("## 3. Retrieval method")
    a("- `TfidfVectorizer` (scikit-learn) fit on normalized customer messages -> L2-normalized sparse "
      "matrix. Query is vectorized the same way; similarity = cosine (sparse dot product). "
      "Results ranked by score desc with a deterministic tie-break on `brand_tweet_id` asc.")
    a("## 4. Why TF-IDF was selected")
    a("- Local, deterministic, fast, and fully explainable (scores trace to shared terms).")
    a("- No external API/model downloads; trivial to reproduce under the 15-minute budget.")
    a("- A strong, honest lexical **baseline** to beat before investing in embeddings (a later phase).")
    a("## 5. Vectorizer configuration")
    a("```json")
    a(json.dumps(index_meta.get("vectorizer_config", {}), indent=2))
    a("```")
    a("## 6. Index artifacts (`data/retrieval/`)")
    for k, v in index_meta.get("artifacts", {}).items():
        a(f"- {k}: `{v}`")
    a("## 7. Retrieval contract")
    a("- Input: `query` (customer message), `top_k` (default 5), `exclude_interaction_id` (optional).")
    a("- Output per result: `rank`, `interaction_id`, `conversation_id`, `customer_text`, "
      "`customer_redacted_text`, `brand_text`, `brand_redacted_text`, `intent` (provisional), "
      "`retrieval_score`.")
    a("## 8. Evaluation methodology")
    a("- Query = each golden customer message; the query's own indexed message is excluded "
      "(via normalized-text -> interaction_id) so it cannot match itself.")
    a("- Metric = **intent-proxy hit rate**: a retrieved result 'hits' if its provisional intent "
      "equals the query's provisional intent. This is a proxy, not human relevance.")
    a(f"- Deterministic; top_k={TOP_K}.")
    a("## 9. Metrics (intent-proxy; provisional labels)")
    a("")
    a("| Metric | Value |")
    a("| --- | ---: |")
    a(f"| Examples evaluated | {agg['evaluated_examples']} / {agg['golden_set_size']} |")
    a(f"| Empty-query (skipped) | {agg['empty_query_examples']} |")
    a(f"| Queries with no non-zero result | {agg['no_nonzero_result_queries']} |")
    a(f"| Hit Rate@1 | {hr['@1']} |")
    a(f"| Hit Rate@3 | {hr['@3']} |")
    a(f"| Hit Rate@5 | {hr['@5']} |")
    a(f"| Hit Rate@10 | {hr['@10']} |")
    a(f"| MRR (intent) | {agg['mrr']} |")
    a(f"| Mean top-1 cosine | {agg['mean_top1_score']} |")
    a("")
    a("## 10. Qualitative examples")
    a(f"Generated from real retriever output ({len(qualitative)} examples; redacted text).")
    for e in qualitative:
        a("")
        a(f"### {e['golden_id']} - _{e['category']}_  (query intent: `{e['query_intent']}`)")
        flags = [f for f in ("is_short", "is_ambiguous", "is_multi_turn", "has_url") if e.get(f)]
        a(f"- Query: {e['query_redacted'][:200]}")
        if flags:
            a(f"- Flags: {', '.join(flags)}")
        if not e["results"]:
            a("- Retrieved: (none)")
        for r in e["results"][:3]:
            mark = "MATCH" if r["intent"] == e["query_intent"] else "diff"
            a(f"  - #{r['rank']} score={r['retrieval_score']:.4f} intent=`{r['intent']}` [{mark}]")
            a(f"    - hist. customer: {r['customer_redacted_text'][:160]}")
            a(f"    - Tesco reply: {r['brand_redacted_text'][:160]}")
        a(f"- Assessment: {_explain(e)}")
    a("")
    a("## 11. Known limitations")
    a("- TF-IDF captures **lexical** similarity only; synonyms/paraphrases may not retrieve well.")
    a("- Short queries have few tokens and often retrieve weakly or not at all.")
    a("- Intent labels are **provisional** (rule-based, not human-verified).")
    a("- Intent-match is a **proxy** for relevance; it is also mildly **optimistic** here because the "
      "provisional intent rules and TF-IDF both key off surface terms, so lexical matches tend to "
      "share the rule-assigned intent.")
    a("- Retrieval relevance has **not** yet been judged by a human evaluator.")
    a("- This phase does **not** generate replies and uses **no external API**.")
    a("- A later phase may compare this baseline with embeddings; that comparison is not implemented here.")
    a("## 12. Reproducibility commands")
    a("```bash")
    a(f"python scripts/build_retrieval_index.py --corpus {index_meta.get('source_corpus_path')} "
      f"--output-dir {index_dir}")
    a(f"python scripts/evaluate_retrieval.py --golden {golden_path} --index-dir {index_dir}")
    a("python scripts/validate_retrieval.py")
    a('python -m src.retrieval.retriever --query "my delivery never arrived" --top-k 5')
    a("```")
    a("## 13. Next-step recommendation")
    a("- Add a small human relevance check on ~30-50 retrievals to calibrate the intent-proxy metric.")
    a("- Only then consider an embedding retriever (e.g. sentence-transformers) and compare against "
      "this TF-IDF baseline on the same golden set.")
    a("")
    md_path = reports_dir / "phase5_retrieval_report.md"
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return md_path, json_path


def main() -> None:
    _stdout_utf8_safe()
    parser = argparse.ArgumentParser(description="Evaluate the Tesco TF-IDF retriever on the golden set.")
    parser.add_argument("--golden", default=str(Path(config.PROCESSED_DIR) / "tesco_golden_set.parquet"))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        alt = golden_path.with_suffix(".csv")
        golden_path = alt if alt.exists() else golden_path
    if not golden_path.exists():
        print(f"ERROR: golden set not found at {args.golden}", file=sys.stderr)
        sys.exit(1)

    golden = pd.read_parquet(golden_path) if golden_path.suffix == ".parquet" else pd.read_csv(golden_path)
    retriever = TescoRetriever.load(args.index_dir)
    print(f"Loaded index ({len(retriever.metadata):,} rows); evaluating {len(golden)} golden examples ...",
          flush=True)

    agg, per_example = run_eval(retriever, golden)
    qualitative = pick_qualitative(per_example)
    md_path, json_path = write_report(agg, per_example, qualitative, retriever.index_meta,
                                      Path(args.reports_dir), str(golden_path).replace("\\", "/"),
                                      str(args.index_dir).replace("\\", "/"))

    print("\n===== RETRIEVAL EVAL (intent-proxy; provisional labels) =====", flush=True)
    print(f"evaluated        : {agg['evaluated_examples']}/{agg['golden_set_size']} "
          f"(empty {agg['empty_query_examples']}, no-result {agg['no_nonzero_result_queries']})")
    for k in K_VALUES:
        print(f"HitRate@{k:<2}       : {agg['hit_rate'][f'@{k}']}")
    print(f"MRR              : {agg['mrr']}")
    print(f"mean top-1 cosine: {agg['mean_top1_score']}")
    print(f"\nreports:\n  {md_path}\n  {json_path}")


if __name__ == "__main__":
    main()
