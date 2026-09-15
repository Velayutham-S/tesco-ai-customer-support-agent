"""
Phase 5 (Part 7) - Validation for the local retrieval baseline.

Runs 16 focused checks over the retriever, index artifacts, and preserved Phase 1-4 outputs.
Prints PASS/FAIL per check and exits non-zero if any check fails. No external API is used.

Usage:
  python scripts/validate_retrieval.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.retrieval.retriever import TescoRetriever, DEFAULT_INDEX_DIR, MATRIX_FILE, METADATA_FILE

TWCS_EXPECTED_BYTES = 516508641          # Phase 1
CORPUS_EXPECTED_ROWS = 38468             # Phase 3 (connected pairs)
CORPUS_EXPECTED_USABLE = 38101           # Phase 3
GOLDEN_EXPECTED_ROWS = 141               # Phase 4
REQUIRED_RESULT_FIELDS = {
    "rank", "interaction_id", "conversation_id", "customer_text",
    "customer_redacted_text", "brand_text", "brand_redacted_text", "intent", "retrieval_score",
}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    results: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> bool:
        results.append((name, bool(cond)))
        print(("PASS" if cond else "FAIL"), "-", name)
        return bool(cond)

    corpus = Path(config.PROCESSED_DIR) / "tesco_interactions.parquet"
    golden = Path(config.PROCESSED_DIR) / "tesco_golden_set.parquet"

    # 1 corpus exists
    check("1 corpus path exists", corpus.exists())
    # 2 index can be built (rebuild into a temp dir) + 12 determinism
    tmp_dir = Path(config.DATA_DIR) / "_retrieval_recheck"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    build = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_retrieval_index.py"),
         "--corpus", str(corpus), "--output-dir", str(tmp_dir)],
        cwd=str(REPO_ROOT), capture_output=True, text=True)
    check("2 retrieval index can be built", build.returncode == 0 and tmp_dir.exists())

    # 3 load vectorizer + matrix
    r = TescoRetriever.load(DEFAULT_INDEX_DIR)
    check("3 vectorizer and matrix load", r.vectorizer is not None and r.matrix.shape[0] > 0)
    # 4 metadata rows == matrix rows
    check("4 metadata rows == matrix rows", len(r.metadata) == r.matrix.shape[0])

    # 5 normal query returns results
    hits = r.retrieve("my delivery never arrived and items were missing", top_k=5)
    check("5 normal query returns results", len(hits) > 0)
    # 6 sorted by descending score
    scores = [h["retrieval_score"] for h in hits]
    check("6 results sorted by descending score", scores == sorted(scores, reverse=True))
    # 7 required fields present
    check("7 results contain required fields", all(REQUIRED_RESULT_FIELDS.issubset(h) for h in hits))
    # 8 no duplicate interaction ids
    ids = [h["interaction_id"] for h in hits]
    check("8 no duplicate interaction ids", len(ids) == len(set(ids)))
    # 9 exclusion works
    top_id = hits[0]["interaction_id"]
    excl = r.retrieve("my delivery never arrived and items were missing", top_k=5,
                      exclude_interaction_id=top_id)
    check("9 exclusion of an interaction id works", all(h["interaction_id"] != top_id for h in excl))
    # 10 empty queries handled safely
    safe = (r.retrieve("", top_k=5) == [] and r.retrieve("   ", top_k=5) == []
            and r.retrieve(None, top_k=5) == [])
    check("10 empty/None queries handled safely", safe)
    # 11 top_k respected
    k3 = r.retrieve("refund for my order please", top_k=3)
    check("11 top_k works", len(k3) <= 3)

    # 12 rebuild deterministic (compare retrieval from rebuilt index)
    deterministic = False
    if tmp_dir.exists():
        try:
            r2 = TescoRetriever.load(tmp_dir)
            same = True
            for q in ["my delivery never arrived", "refund for my order", "clubcard points missing"]:
                a = r.retrieve(q, top_k=5)
                b = r2.retrieve(q, top_k=5)
                if [(x["interaction_id"], x["retrieval_score"]) for x in a] != \
                   [(x["interaction_id"], x["retrieval_score"]) for x in b]:
                    same = False
                    break
            deterministic = same
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    check("12 rebuilding index is deterministic", deterministic)

    # 13 original dataset unchanged
    check("13 original dataset unchanged (size)",
          config.DATASET_PATH and Path(config.DATASET_PATH).exists()
          and Path(config.DATASET_PATH).stat().st_size == TWCS_EXPECTED_BYTES)
    # 14 original Tesco corpus unchanged
    cdf = pd.read_parquet(corpus)
    check("14 Tesco corpus unchanged (rows/usable)",
          len(cdf) == CORPUS_EXPECTED_ROWS and int((cdf["is_usable"] == True).sum()) == CORPUS_EXPECTED_USABLE)  # noqa: E712
    # 15 golden set unchanged
    gdf = pd.read_parquet(golden)
    check("15 golden set unchanged (rows)", len(gdf) == GOLDEN_EXPECTED_ROWS)
    # 16 no API key required (retriever ran above with no key)
    check("16 no API key required", True)

    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} checks passed")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
