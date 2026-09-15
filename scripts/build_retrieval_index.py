"""
Phase 5 (Part 3) - Build the local TF-IDF retrieval index for Tesco.

Reads the Phase 3 corpus, keeps usable interactions, deduplicates identical retrieval text,
attaches a PROVISIONAL intent per row (same rule engine as Phase 4), fits a TfidfVectorizer on
the normalized customer messages, and saves reusable index artifacts to data/retrieval/.

Usage:
  python scripts/build_retrieval_index.py --corpus data/processed/tesco_interactions.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src import intent_taxonomy as tax
from src.retrieval.retriever import (
    VECTORIZER_FILE, MATRIX_FILE, METADATA_FILE, INDEX_META_FILE, DEFAULT_INDEX_DIR,
)

RETRIEVAL_TEXT_COLUMN = "customer_normalized_text"
REQUIRED_COLUMNS = [
    "pair_id", "conversation_id", "customer_tweet_id", "brand_tweet_id",
    "customer_text", "customer_normalized_text", "customer_redacted_text",
    "brand_text", "brand_redacted_text", "is_usable",
]
METADATA_COLUMNS = [
    "interaction_id", "conversation_id", "customer_tweet_id", "brand_tweet_id",
    "customer_text", "customer_redacted_text", "customer_normalized_text",
    "brand_text", "brand_redacted_text", "intent",
    "is_generic_reply", "is_multi_turn_conversation",
]


def load_and_filter(corpus_path: Path) -> tuple[pd.DataFrame, dict]:
    if not corpus_path.exists():
        print(f"ERROR: corpus not found at {corpus_path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(corpus_path) if corpus_path.suffix == ".parquet" else pd.read_csv(corpus_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"ERROR: corpus missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    n_total = len(df)
    df = df[df["is_usable"] == True].copy()  # noqa: E712
    n_usable = len(df)
    # drop empty retrieval text
    df[RETRIEVAL_TEXT_COLUMN] = df[RETRIEVAL_TEXT_COLUMN].fillna("").astype(str)
    df = df[df[RETRIEVAL_TEXT_COLUMN].str.strip().str.len() > 0]
    n_nonempty = len(df)
    # deterministic dedupe of identical retrieval text (keep first by brand_tweet_id)
    df = df.sort_values("brand_tweet_id", kind="stable")
    df = df.drop_duplicates(subset=[RETRIEVAL_TEXT_COLUMN], keep="first")
    n_dedup = len(df)

    stats = {
        "rows_total": n_total,
        "rows_usable": n_usable,
        "rows_nonempty_text": n_nonempty,
        "rows_after_dedupe": n_dedup,
        "duplicate_text_rows_removed": n_nonempty - n_dedup,
    }
    return df.reset_index(drop=True), stats


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    intents = [tax.assign_intent(t)["intent"] for t in df["customer_text"].astype(str)]
    meta = pd.DataFrame({
        "interaction_id": df["pair_id"].astype(str),
        "conversation_id": df["conversation_id"].astype(str),
        "customer_tweet_id": df["customer_tweet_id"].astype("int64"),
        "brand_tweet_id": df["brand_tweet_id"].astype("int64"),
        "customer_text": df["customer_text"].astype(str),
        "customer_redacted_text": df["customer_redacted_text"].astype(str),
        "customer_normalized_text": df[RETRIEVAL_TEXT_COLUMN].astype(str),
        "brand_text": df["brand_text"].astype(str),
        "brand_redacted_text": df["brand_redacted_text"].astype(str),
        "intent": intents,
        "is_generic_reply": df["is_generic_reply"].astype(bool) if "is_generic_reply" in df else False,
        "is_multi_turn_conversation": df["is_multi_turn_conversation"].astype(bool)
        if "is_multi_turn_conversation" in df else False,
    })
    # stable final order by brand_tweet_id (matrix rows will align to this order)
    return meta.sort_values("brand_tweet_id", kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Tesco TF-IDF retrieval index.")
    parser.add_argument("--corpus", default=str(Path(config.PROCESSED_DIR) / "tesco_interactions.parquet"))
    parser.add_argument("--output-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--ngram-max", type=int, default=2)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus: {corpus_path}", flush=True)
    df, stats = load_and_filter(corpus_path)
    print(f"  usable={stats['rows_usable']:,} nonempty={stats['rows_nonempty_text']:,} "
          f"after_dedupe={stats['rows_after_dedupe']:,} "
          f"(removed {stats['duplicate_text_rows_removed']:,} duplicate-text rows)", flush=True)

    meta = build_metadata(df)
    print(f"  building TF-IDF over {len(meta):,} rows using column '{RETRIEVAL_TEXT_COLUMN}'", flush=True)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(meta["customer_normalized_text"].tolist())
    print(f"  TF-IDF matrix shape={matrix.shape} nnz={matrix.nnz:,} "
          f"vocab={len(vectorizer.vocabulary_):,}", flush=True)

    # save artifacts
    joblib.dump(vectorizer, out_dir / VECTORIZER_FILE)
    sparse.save_npz(out_dir / MATRIX_FILE, matrix)
    meta.to_parquet(out_dir / METADATA_FILE, index=False)

    index_meta = {
        "brand": config.SELECTED_BRAND,
        "phase": "phase5_retrieval",
        "source_corpus_path": str(corpus_path).replace("\\", "/"),
        "num_indexed_rows": int(matrix.shape[0]),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "retrieval_text_column": RETRIEVAL_TEXT_COLUMN,
        "vectorizer_config": {
            "lowercase": True, "ngram_range": [1, args.ngram_max],
            "min_df": args.min_df, "max_df": args.max_df, "sublinear_tf": True, "norm": "l2",
        },
        "filtering_rules": [
            "is_usable == True",
            f"non-empty {RETRIEVAL_TEXT_COLUMN}",
            "deduplicated identical retrieval text (keep first by brand_tweet_id)",
        ],
        "duplicate_text_rows_removed": stats["duplicate_text_rows_removed"],
        "row_counts": stats,
        "interaction_id_field": "pair_id",
        "deterministic": True,
        "intent_labels_provisional": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {
            "vectorizer": VECTORIZER_FILE, "matrix": MATRIX_FILE,
            "metadata": METADATA_FILE, "index_metadata": INDEX_META_FILE,
        },
    }
    (out_dir / INDEX_META_FILE).write_text(json.dumps(index_meta, indent=2), encoding="utf-8")

    print("\n===== INDEX BUILD SUMMARY =====", flush=True)
    print(f"indexed rows : {index_meta['num_indexed_rows']:,}")
    print(f"vocab size   : {index_meta['vocabulary_size']:,}")
    print(f"artifacts in : {out_dir}")
    for f in (VECTORIZER_FILE, MATRIX_FILE, METADATA_FILE, INDEX_META_FILE):
        print(f"  {out_dir / f}")


if __name__ == "__main__":
    main()
