"""
Phase 4 (Steps 4-5) - Build a compact Tesco golden evaluation set.

Samples ~100-150 real customer -> Tesco pairs from the Phase 3 corpus, assigns PROVISIONAL
intent labels with the deterministic rule engine (src/intent_taxonomy.py), and marks everything
as pending human review. Sampling is deterministic (fixed seed), stratified across all intents,
and tops up difficulty buckets (short / multi-turn / ambiguous / URL-bearing). No synthetic text
is created and nothing is copied from outside the dataset.

Outputs:
  data/processed/tesco_golden_set.csv
  data/processed/tesco_golden_set.parquet
  reports/tesco_golden_set_profile.md
  reports/tesco_golden_set_summary.json

Usage:
  python scripts/build_tesco_golden_set.py --corpus data/processed/tesco_interactions.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import config
from src import intent_taxonomy as tax
from src.text_utils import word_count, has_real_content

# Sampling configuration (deterministic).
PER_INTENT_TARGET = 12          # aim per intent (11 intents -> ~132 base)
MAX_TOTAL = 150                 # hard cap
SHORT_MAX_WORDS = 4             # "short" customer message (content words)
BUCKET_MINIMUMS = {             # ensure difficult examples are represented
    "short": 10,
    "multi_turn": 10,
    "ambiguous": 10,
    "has_url": 5,
}
# map difficulty-bucket name -> pool/column flag name
BUCKET_COLS = {
    "short": "is_short",
    "multi_turn": "is_multi_turn",
    "ambiguous": "is_ambiguous",
    "has_url": "has_url",
}


def load_corpus(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        alt = path.replace(".parquet", ".csv")
        path = alt if os.path.exists(alt) else path
    if not os.path.exists(path):
        print(f"ERROR: corpus not found at {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def build_pool(df: pd.DataFrame) -> list[dict]:
    """Usable rows, deduped by normalized customer text and limited to <=1 per conversation."""
    usable = df[df["is_usable"] == True]  # noqa: E712
    rows = []
    seen_cust_text: set = set()
    seen_conversations: set = set()
    # deterministic scan order by brand_tweet_id
    usable = usable.sort_values("brand_tweet_id", kind="stable")
    for _, r in usable.iterrows():
        cust_norm = str(r["customer_normalized_text"]).strip().lower()
        conv = r["conversation_id"]
        if not cust_norm or cust_norm in seen_cust_text or conv in seen_conversations:
            continue
        seen_cust_text.add(cust_norm)
        seen_conversations.add(conv)

        cust_text = str(r["customer_text"])
        res = tax.assign_intent(cust_text)
        rows.append({
            "conversation_id": conv,
            "customer_message_id": int(r["customer_tweet_id"]),
            "brand_reply_id": int(r["brand_tweet_id"]),
            "customer_text": cust_text,
            "customer_redacted_text": str(r["customer_redacted_text"]),
            "brand_reply_text": str(r["brand_text"]),
            "brand_reply_redacted_text": str(r["brand_redacted_text"]),
            "brand_norm": str(r["brand_normalized_text"]).strip().lower(),
            "intent": res["intent"],
            "matched_intents": ";".join(res["matched"]),
            "label_rule": res["rule"],
            "is_short": word_count(cust_text) <= SHORT_MAX_WORDS,
            "is_multi_turn": bool(r["is_multi_turn_conversation"]) or int(r["turn_in_conversation"]) > 1,
            "is_ambiguous": bool(res["is_ambiguous"]),
            "is_generic_reply": bool(r["is_generic_reply"]),
            "has_url": bool(str(r["customer_urls"]).strip()) and has_real_content(cust_text),
            "turn_in_conversation": int(r["turn_in_conversation"]),
        })
    return rows


def select(pool: list[dict], seed: int) -> list[dict]:
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)

    selected: dict[int, set] = {}
    used_reply: set = set()

    def try_pick(i: int, reason: str) -> bool:
        if i in selected:
            selected[i].add(reason)
            return False
        if pool[i]["brand_norm"] in used_reply:  # avoid duplicate reply text
            return False
        selected[i] = {reason}
        used_reply.add(pool[i]["brand_norm"])
        return True

    # 1) intent-stratified coverage
    for intent in tax.PRIORITY_ORDER:
        n = 0
        for i in order:
            if len(selected) >= MAX_TOTAL:
                break
            if pool[i]["intent"] == intent and try_pick(i, f"intent:{intent}"):
                n += 1
                if n >= PER_INTENT_TARGET:
                    break

    # 2) difficulty top-ups
    for bucket, minimum in BUCKET_MINIMUMS.items():
        col = BUCKET_COLS[bucket]
        have = sum(1 for i in selected if pool[i][col])
        if have >= minimum:
            continue
        for i in order:
            if have >= minimum or len(selected) >= MAX_TOTAL:
                break
            if pool[i][col] and try_pick(i, f"difficulty:{bucket}"):
                have += 1

    out = []
    for i, reasons in selected.items():
        row = dict(pool[i])
        row["sampling_reason"] = ";".join(sorted(reasons))
        out.append(row)
    return out


def finalize(rows: list[dict]) -> pd.DataFrame:
    # deterministic ordering, then stable golden ids
    rows.sort(key=lambda r: (r["intent"], r["conversation_id"], r["brand_reply_id"]))
    records = []
    for idx, r in enumerate(rows, start=1):
        notes = ""
        if r["is_ambiguous"]:
            others = [m for m in r["matched_intents"].split(";") if m and m != r["intent"]]
            if others:
                notes = "ambiguous; also matched: " + ", ".join(others)
        records.append({
            "golden_id": f"gold-{idx:04d}",
            "conversation_id": r["conversation_id"],
            "customer_message_id": r["customer_message_id"],
            "brand_reply_id": r["brand_reply_id"],
            "customer_text": r["customer_text"],
            "customer_redacted_text": r["customer_redacted_text"],
            "brand_reply_text": r["brand_reply_text"],
            "brand_reply_redacted_text": r["brand_reply_redacted_text"],
            "intent": r["intent"],
            "intent_is_provisional": True,
            "label_method": "provisional_keyword_rule",
            "label_rule": r["label_rule"],
            "matched_intents": r["matched_intents"],
            "source_split": "tesco_usable_corpus",
            "sampling_reason": r["sampling_reason"],
            "is_short": r["is_short"],
            "is_multi_turn": r["is_multi_turn"],
            "is_ambiguous": r["is_ambiguous"],
            "is_generic_reply": r["is_generic_reply"],
            "has_url": r["has_url"],
            "turn_in_conversation": r["turn_in_conversation"],
            "annotation_status": "pending",
            "notes": notes,
        })
    return pd.DataFrame(records)


def write_outputs(gdf: pd.DataFrame, out_dir: str, reports_dir: str, pool_size: int, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "tesco_golden_set.csv")
    gdf.to_csv(csv_path, index=False, encoding="utf-8")
    parquet_path = os.path.join(out_dir, "tesco_golden_set.parquet")
    parquet_ok, parquet_note = True, ""
    try:
        gdf.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover
        parquet_ok, parquet_note, parquet_path = False, f"{exc.__class__.__name__}: {exc}", None

    intent_counts = gdf["intent"].value_counts().to_dict()
    per_intent = {name: int(intent_counts.get(name, 0)) for name in tax.PRIORITY_ORDER}
    bucket_counts = {
        "short": int(gdf["is_short"].sum()),
        "multi_turn": int(gdf["is_multi_turn"].sum()),
        "ambiguous": int(gdf["is_ambiguous"].sum()),
        "has_url": int(gdf["has_url"].sum()),
        "generic_reply": int(gdf["is_generic_reply"].sum()),
    }
    summary = {
        "brand": config.SELECTED_BRAND,
        "taxonomy_version": tax.TAXONOMY_VERSION,
        "golden_set_size": int(len(gdf)),
        "target_size_range": "100-150",
        "per_intent_target": PER_INTENT_TARGET,
        "max_total": MAX_TOTAL,
        "random_seed": seed,
        "dedup_pool_size": pool_size,
        "intent_distribution": per_intent,
        "difficulty_bucket_counts": bucket_counts,
        "labeling_method": "provisional_keyword_rule (deterministic); all rows annotation_status=pending",
        "intents_are_provisional": True,
        "parquet_written": parquet_ok,
        "parquet_note": parquet_note,
        "reproduction_command": (
            "python scripts/build_tesco_golden_set.py "
            "--corpus data/processed/tesco_interactions.parquet --output-dir data/processed"
        ),
    }
    json_path = os.path.join(reports_dir, "tesco_golden_set_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    md_path = write_profile(summary, gdf, reports_dir)
    return csv_path, parquet_path, json_path, md_path, summary


def write_profile(summary: dict, gdf: pd.DataFrame, reports_dir: str) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Tesco Golden Evaluation Set - Profile")
    add("")
    add(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `scripts/build_tesco_golden_set.py`._")
    add("")
    add(f"- Brand: **{summary['brand']}**  |  Taxonomy version: {summary['taxonomy_version']}")
    add(f"- Golden-set size: **{summary['golden_set_size']}** (target {summary['target_size_range']})")
    add(f"- Deterministic seed: {summary['random_seed']}  |  Dedup pool size: {summary['dedup_pool_size']:,}")
    add("")
    add("## Labeling method (IMPORTANT)")
    add("- Intent labels are **PROVISIONAL**, assigned by the deterministic keyword/priority rule "
        "engine in `src/intent_taxonomy.py`. They are **not** human-verified.")
    add("- Every row has `annotation_status = pending` and `intent_is_provisional = True`; the "
        "matched rule is stored in `label_rule` and secondary matches in `matched_intents`.")
    add("- Intended use: a human reviewer confirms/curates these labels before they are treated as "
        "ground truth for evaluation.")
    add("")
    add("## Intent distribution (stratified by design, not corpus-proportional)")
    add("")
    add("| Intent | Count |")
    add("| --- | ---: |")
    for name in tax.PRIORITY_ORDER:
        add(f"| `{name}` | {summary['intent_distribution'].get(name, 0)} |")
    add("")
    add("## Difficulty coverage")
    add("")
    add("| Bucket | Count |")
    add("| --- | ---: |")
    for k, v in summary["difficulty_bucket_counts"].items():
        add(f"| {k} | {v} |")
    add("")
    add("## Sampling strategy")
    add("- Source: usable customer→Tesco pairs from `data/processed/tesco_interactions.parquet`.")
    add("- Dedup: exact normalized customer text removed; at most **one pair per conversation**; "
        "no duplicate brand-reply text within the set.")
    add(f"- Stratified: up to {summary['per_intent_target']} per intent across all "
        f"{len(tax.PRIORITY_ORDER)} intents, then difficulty top-ups "
        "(short / multi-turn / ambiguous / URL-bearing) to guarantee hard cases.")
    add(f"- Deterministic: fixed seed {summary['random_seed']} controls a stable shuffle; re-running "
        "reproduces the identical set.")
    add("- No synthetic messages; nothing copied from outside the dataset; original dataset unchanged.")
    add("")
    add("## Row schema")
    add("`" + ", ".join(gdf.columns.tolist()) + "`")
    add("")
    add("## Example rows (redacted)")
    for name in ["delivery_issue", "refund_or_payment", "other_or_unclear"]:
        sub = gdf[gdf["intent"] == name].head(1)
        for _, r in sub.iterrows():
            add(f"- **{name}** ({r['golden_id']}): {str(r['customer_redacted_text'])[:140]}")
    add("")
    add("## Limitations")
    add("- Provisional labels reflect keyword rules, so some will be corrected on human review "
        "(especially `store_experience` breadth and `other_or_unclear`).")
    add("- Balanced-by-intent sampling does not reflect the real corpus prior (e.g. delivery/store "
        "dominate); this is intentional for evaluation coverage of rare intents.")
    add("- Small set (~130): metrics will have wide confidence intervals; treat as a smoke/quality "
        "gauge, not a precise leaderboard.")
    add("")
    add("## Reproduction command")
    add("```")
    add(summary["reproduction_command"])
    add("```")
    add("")
    path = os.path.join(reports_dir, "tesco_golden_set_profile.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Tesco golden evaluation set (deterministic).")
    p.add_argument("--corpus", default=os.path.join(config.PROCESSED_DIR, "tesco_interactions.parquet"))
    p.add_argument("--output-dir", default=config.PROCESSED_DIR)
    p.add_argument("--reports-dir", default=config.REPORTS_DIR)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if (config.SELECTED_BRAND or "").lower() != "tesco":
        print(f"WARNING: expected brand Tesco, got {config.SELECTED_BRAND}", file=sys.stderr)
    print(f"Loading corpus: {args.corpus}", flush=True)
    df = load_corpus(args.corpus)
    print(f"  rows={len(df):,}; usable={int((df['is_usable']==True).sum()):,}", flush=True)  # noqa: E712

    pool = build_pool(df)
    print(f"  dedup pool (<=1/conversation, unique customer text) = {len(pool):,}", flush=True)
    selected = select(pool, args.seed)
    gdf = finalize(selected)

    csv_path, parquet_path, json_path, md_path, summary = write_outputs(
        gdf, args.output_dir, args.reports_dir, len(pool), args.seed)

    print("\n===== GOLDEN SET SUMMARY =====", flush=True)
    print(f"size            : {summary['golden_set_size']}")
    print(f"intent distribution:")
    for name in tax.PRIORITY_ORDER:
        print(f"  {name:28s} {summary['intent_distribution'].get(name,0):3d}")
    print(f"difficulty      : {summary['difficulty_bucket_counts']}")
    print(f"parquet written : {summary['parquet_written']}")
    print(f"\nfiles:\n  {csv_path}\n  {parquet_path}\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
