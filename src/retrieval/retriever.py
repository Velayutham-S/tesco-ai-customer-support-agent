"""TF-IDF retriever for the Tesco support agent (Phase 5).

Local, deterministic, explainable retrieval of historical Tesco customer→brand interactions
for a new customer message. Loads a prebuilt index (fitted TfidfVectorizer + sparse matrix +
row metadata) and returns ranked results with identifiers, text, intent and score.

The retrieval text is the historical *customer* message (normalized), because the incoming query
is also a customer message. Brand replies are returned for display only, never searched.

Reusable from Python:
    from src.retrieval import TescoRetriever
    r = TescoRetriever.load()
    hits = r.retrieve("my delivery never arrived", top_k=5)

...or from the CLI:
    python -m src.retrieval.retriever --query "my delivery never arrived" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

try:  # works as `src.retrieval.retriever` and (via sys.path) from scripts
    from src import config
    from src.text_utils import normalize_text
except ImportError:  # pragma: no cover
    import config  # type: ignore
    from text_utils import normalize_text  # type: ignore

# Artifact filenames (kept together in the index directory).
VECTORIZER_FILE = "tesco_tfidf_vectorizer.joblib"
MATRIX_FILE = "tesco_tfidf_matrix.npz"
METADATA_FILE = "tesco_retrieval_metadata.parquet"
INDEX_META_FILE = "tesco_retrieval_index_metadata.json"

DEFAULT_INDEX_DIR = Path(config.DATA_DIR) / "retrieval"

# Columns the metadata parquet must contain for retrieval to work.
REQUIRED_METADATA_COLUMNS = [
    "interaction_id", "conversation_id", "brand_tweet_id",
    "customer_text", "customer_redacted_text",
    "brand_text", "brand_redacted_text", "intent",
]


@dataclass
class RetrievalResult:
    """A single retrieved historical interaction."""

    rank: int
    interaction_id: str
    conversation_id: str
    customer_text: str
    customer_redacted_text: str
    brand_text: str
    brand_redacted_text: str
    intent: str
    retrieval_score: float


class TescoRetriever:
    """Cosine-similarity TF-IDF retriever over historical Tesco interactions."""

    def __init__(self, vectorizer, matrix: sparse.csr_matrix,
                 metadata: pd.DataFrame, index_meta: Optional[dict] = None) -> None:
        if len(metadata) != matrix.shape[0]:
            raise ValueError(
                f"metadata rows ({len(metadata)}) != matrix rows ({matrix.shape[0]})")
        missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in metadata.columns]
        if missing:
            raise ValueError(f"metadata missing required columns: {missing}")
        self.vectorizer = vectorizer
        self.matrix = matrix.tocsr()
        self.metadata = metadata.reset_index(drop=True)
        self.index_meta = index_meta or {}
        # cached arrays for fast, deterministic ranking
        self._ids = self.metadata["interaction_id"].astype(str).to_numpy()
        self._tiebreak = self.metadata["brand_tweet_id"].astype("int64").to_numpy()

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, index_dir: str | Path = DEFAULT_INDEX_DIR) -> "TescoRetriever":
        """Load a prebuilt index from `index_dir` (raises if artifacts are missing)."""
        index_dir = Path(index_dir)
        for fname in (VECTORIZER_FILE, MATRIX_FILE, METADATA_FILE):
            if not (index_dir / fname).exists():
                raise FileNotFoundError(
                    f"Missing retrieval artifact {index_dir / fname}. "
                    "Build it first: python scripts/build_retrieval_index.py")
        vectorizer = joblib.load(index_dir / VECTORIZER_FILE)
        matrix = sparse.load_npz(index_dir / MATRIX_FILE)
        metadata = pd.read_parquet(index_dir / METADATA_FILE)
        meta_path = index_dir / INDEX_META_FILE
        index_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(vectorizer, matrix, metadata, index_meta)

    # --------------------------------------------------------------- retrieval
    def retrieve(self, query: str, top_k: int = 5,
                 exclude_interaction_id: Optional[str] = None,
                 min_score: float = 0.0) -> list[dict]:
        """Return up to `top_k` historical interactions most similar to `query`.

        Args:
            query: an incoming customer message.
            top_k: maximum number of results (>=1).
            exclude_interaction_id: interaction_id to drop (used to avoid self-match in eval).
            min_score: results with cosine score <= this are dropped (default 0.0 removes zeros).

        Returns:
            A list of result dicts (see RetrievalResult), ranked by score desc with a
            deterministic tie-break on brand_tweet_id asc. Empty list for empty/short/unknown
            queries. Never raises on empty input.
        """
        if top_k < 1:
            return []
        if query is None:
            return []
        norm = normalize_text(query)
        if not norm.strip():
            return []
        qv = self.vectorizer.transform([norm])
        if qv.nnz == 0:  # no in-vocabulary tokens -> nothing meaningful to match
            return []

        scores = np.asarray((self.matrix @ qv.T).todense()).ravel()
        cand = np.where(scores > min_score)[0]
        if cand.size == 0:
            return []

        # deterministic ordering: score desc, then brand_tweet_id asc
        order = np.lexsort((self._tiebreak[cand], -scores[cand]))
        ordered = cand[order]

        results: list[RetrievalResult] = []
        seen: set[str] = set()
        for idx in ordered:
            iid = str(self._ids[idx])
            if exclude_interaction_id is not None and iid == str(exclude_interaction_id):
                continue
            if iid in seen:  # metadata ids are unique, but guard anyway
                continue
            seen.add(iid)
            row = self.metadata.iloc[idx]
            results.append(RetrievalResult(
                rank=len(results) + 1,
                interaction_id=iid,
                conversation_id=str(row["conversation_id"]),
                customer_text=str(row["customer_text"]),
                customer_redacted_text=str(row["customer_redacted_text"]),
                brand_text=str(row["brand_text"]),
                brand_redacted_text=str(row["brand_redacted_text"]),
                intent=str(row["intent"]),
                retrieval_score=round(float(scores[idx]), 6),
            ))
            if len(results) >= top_k:
                break
        return [asdict(r) for r in results]


# --------------------------------------------------------------------- CLI
def _format_result(r: dict) -> str:
    return (f"  #{r['rank']}  score={r['retrieval_score']:.4f}  intent={r['intent']}  "
            f"id={r['interaction_id']}\n"
            f"     customer: {r['customer_redacted_text'][:140]}\n"
            f"     tesco   : {r['brand_redacted_text'][:140]}")


def _make_stdout_utf8_safe() -> None:
    """Windows consoles default to cp1252 and crash on emoji in tweet text; force UTF-8."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def main() -> None:
    _make_stdout_utf8_safe()
    parser = argparse.ArgumentParser(description="Query the Tesco TF-IDF retriever.")
    parser.add_argument("--query", required=True, help="incoming customer message")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--exclude", default=None, help="interaction_id to exclude")
    args = parser.parse_args()

    retriever = TescoRetriever.load(args.index_dir)
    hits = retriever.retrieve(args.query, top_k=args.top_k, exclude_interaction_id=args.exclude)
    print(f"query: {args.query!r}  (top_k={args.top_k}) -> {len(hits)} result(s)")
    if not hits:
        print("  (no results - empty/short/unknown query or no lexical overlap)")
    for r in hits:
        print(_format_result(r))


if __name__ == "__main__":
    main()
