"""
Phase 1 dataset inspection for the Hiver SDE assignment.

Profiles the Customer Support on Twitter dataset (thoughtvector/customer-support-on-twitter)
without loading the whole ~500 MB / ~3M-row file into memory. Reads in chunks, aggregates
statistics, verifies the schema, and writes a Markdown profile to reports/dataset_profile.md.

Usage:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --path dataset/twcs/twcs.csv --chunksize 200000

The script is deterministic: given the same input file it always produces the same report.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime

import pandas as pd

# Columns we expect for the Customer Support on Twitter dataset.
EXPECTED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

# created_at looks like: "Tue Oct 31 22:10:47 +0000 2017"
CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the twcs dataset (chunked).")
    parser.add_argument("--path", default="dataset/twcs/twcs.csv",
                        help="Path to twcs.csv (default: dataset/twcs/twcs.csv)")
    parser.add_argument("--chunksize", type=int, default=200_000,
                        help="Rows per chunk (default: 200000)")
    parser.add_argument("--out", default="reports/dataset_profile.md",
                        help="Output Markdown path (default: reports/dataset_profile.md)")
    parser.add_argument("--top-brands", type=int, default=25,
                        help="How many top brands to list (default: 25)")
    return parser.parse_args()


def human_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 2)


def truncate(text: object, limit: int = 160) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "\u2026"


def inspect(path: str, chunksize: int, top_brands: int) -> dict:
    if not os.path.exists(path):
        print(f"ERROR: dataset not found at {path}", file=sys.stderr)
        sys.exit(1)

    size_bytes = os.path.getsize(path)

    # --- dtypes as pandas would naturally infer them (from a small sample) ---
    sample_df = pd.read_csv(path, nrows=5000, encoding="utf-8")
    inferred_dtypes = {c: str(t) for c, t in sample_df.dtypes.items()}
    header_columns = list(sample_df.columns)

    # --- accumulators for the full chunked pass ---
    total_rows = 0
    null_counts: Counter = Counter()
    inbound_true = 0
    inbound_false = 0
    inbound_other = 0
    brand_reply_counter: Counter = Counter()   # outbound author_id -> reply count
    author_set: set = set()                      # all unique author_ids (brands + customers)
    has_response = 0                             # rows with a response_tweet_id
    has_in_response_to = 0                       # rows that are themselves a reply
    roots = 0                                    # rows with no in_response_to (thread starts)
    mid_thread = 0                               # rows that both reply and get replied to
    branching = 0                                # response_tweet_id with multiple ids (comma)
    ts_min: pd.Timestamp | None = None
    ts_max: pd.Timestamp | None = None
    ts_parse_fail = 0
    empty_text = 0
    examples: list[dict] = []

    reader = pd.read_csv(
        path,
        chunksize=chunksize,
        dtype=str,                # read everything as text; missing -> NaN
        encoding="utf-8",
        on_bad_lines="warn",      # surface malformed rows instead of hiding them
    )

    for i, chunk in enumerate(reader):
        total_rows += len(chunk)

        # missing values per column
        for col in chunk.columns:
            null_counts[col] += int(chunk[col].isna().sum())

        # inbound distribution (strings 'True' / 'False')
        inbound = chunk["inbound"]
        it = int((inbound == "True").sum())
        ifa = int((inbound == "False").sum())
        inbound_true += it
        inbound_false += ifa
        inbound_other += len(chunk) - it - ifa

        # brand reply volume = author_id of outbound (brand) tweets
        outbound_authors = chunk.loc[inbound == "False", "author_id"].dropna()
        brand_reply_counter.update(outbound_authors.tolist())

        # unique authors overall
        author_set.update(chunk["author_id"].dropna().unique().tolist())

        # threading signals
        resp = chunk["response_tweet_id"]
        inresp = chunk["in_response_to_tweet_id"]
        has_response += int(resp.notna().sum())
        has_in_response_to += int(inresp.notna().sum())
        roots += int(inresp.isna().sum())
        mid_thread += int((resp.notna() & inresp.notna()).sum())
        branching += int(resp.dropna().str.contains(",").sum())

        # empty / whitespace-only text
        empty_text += int(chunk["text"].fillna("").str.strip().eq("").sum())

        # timestamps
        ts = pd.to_datetime(chunk["created_at"], format=CREATED_AT_FORMAT,
                            errors="coerce", utc=True)
        ts_parse_fail += int(ts.isna().sum())
        valid = ts.dropna()
        if len(valid):
            cmin, cmax = valid.min(), valid.max()
            ts_min = cmin if ts_min is None else min(ts_min, cmin)
            ts_max = cmax if ts_max is None else max(ts_max, cmax)

        # keep first few example rows
        if i == 0:
            for _, row in chunk.head(5).iterrows():
                examples.append({
                    "tweet_id": row.get("tweet_id"),
                    "author_id": row.get("author_id"),
                    "inbound": row.get("inbound"),
                    "created_at": row.get("created_at"),
                    "text": truncate(row.get("text")),
                    "response_tweet_id": row.get("response_tweet_id"),
                    "in_response_to_tweet_id": row.get("in_response_to_tweet_id"),
                })

        print(f"  processed chunk {i + 1} | cumulative rows = {total_rows:,}")

    return {
        "path": path,
        "size_bytes": size_bytes,
        "size_mb": human_mb(size_bytes),
        "header_columns": header_columns,
        "inferred_dtypes": inferred_dtypes,
        "total_rows": total_rows,
        "null_counts": dict(null_counts),
        "inbound_true": inbound_true,
        "inbound_false": inbound_false,
        "inbound_other": inbound_other,
        "unique_authors": len(author_set),
        "unique_brands": len(brand_reply_counter),
        "top_brands": brand_reply_counter.most_common(top_brands),
        "has_response": has_response,
        "has_in_response_to": has_in_response_to,
        "roots": roots,
        "mid_thread": mid_thread,
        "branching": branching,
        "ts_min": None if ts_min is None else ts_min.isoformat(),
        "ts_max": None if ts_max is None else ts_max.isoformat(),
        "ts_parse_fail": ts_parse_fail,
        "empty_text": empty_text,
        "examples": examples,
    }


def pct(part: int, whole: int) -> str:
    if not whole:
        return "0.00%"
    return f"{100.0 * part / whole:.2f}%"


def build_markdown(stats: dict, chunksize: int, top_brands: int) -> str:
    s = stats
    schema_ok = s["header_columns"] == EXPECTED_COLUMNS
    reconstructable = s["has_response"] > 0 and s["has_in_response_to"] > 0

    lines: list[str] = []
    add = lines.append

    add("# Dataset Profile - Customer Support on Twitter (twcs)")
    add("")
    add(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by "
        f"`scripts/inspect_dataset.py` (chunksize={chunksize:,})._")
    add("")
    add("This profile is produced by a reproducible chunked pass over the full file. "
        "Re-running the script on the same file yields the same report.")
    add("")

    add("## 1. Dataset Path & Size")
    add("")
    add(f"- Path: `{s['path']}`")
    add(f"- Size: **{s['size_mb']} MB** ({s['size_bytes']:,} bytes)")
    add(f"- Read strategy: `pandas.read_csv(chunksize={chunksize:,}, dtype=str)` - "
        "the full file is never held in memory at once.")
    add("")

    add("## 2. Row Count")
    add("")
    add(f"- Total data rows (excluding header): **{s['total_rows']:,}**")
    add("")

    add("## 3. Columns & Data Types")
    add("")
    add("Native CSV values are text; the types below are what pandas infers from a "
        "5,000-row sample (IDs with missing values become floats under naive inference, "
        "which is why the pipeline reads them as strings).")
    add("")
    add("| Column | Inferred dtype |")
    add("| --- | --- |")
    for c in s["header_columns"]:
        add(f"| `{c}` | {s['inferred_dtypes'].get(c, 'n/a')} |")
    add("")

    add("## 4. Missing Values")
    add("")
    add("| Column | Missing | % of rows |")
    add("| --- | --- | --- |")
    for c in s["header_columns"]:
        n = s["null_counts"].get(c, 0)
        add(f"| `{c}` | {n:,} | {pct(n, s['total_rows'])} |")
    add("")
    add(f"- Empty / whitespace-only `text` values: {s['empty_text']:,} "
        f"({pct(s['empty_text'], s['total_rows'])})")
    add("")

    add("## 5. Example Records")
    add("")
    add("First 5 rows (text truncated for readability):")
    add("")
    add("| tweet_id | author_id | inbound | created_at | text | response_tweet_id | in_response_to_tweet_id |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for ex in s["examples"]:
        add("| {tweet_id} | {author_id} | {inbound} | {created_at} | {text} | "
            "{response_tweet_id} | {in_response_to_tweet_id} |".format(
                tweet_id=ex["tweet_id"], author_id=ex["author_id"], inbound=ex["inbound"],
                created_at=ex["created_at"],
                text=str(ex["text"]).replace("|", "\\|"),
                response_tweet_id=ex["response_tweet_id"],
                in_response_to_tweet_id=ex["in_response_to_tweet_id"]))
    add("")

    add("## 6. Customer vs Brand Messages")
    add("")
    add(f"- Customer messages (`inbound=True`): **{s['inbound_true']:,}** "
        f"({pct(s['inbound_true'], s['total_rows'])})")
    add(f"- Brand replies (`inbound=False`): **{s['inbound_false']:,}** "
        f"({pct(s['inbound_false'], s['total_rows'])})")
    if s["inbound_other"]:
        add(f"- Rows with unexpected inbound value: {s['inbound_other']:,}")
    add("")
    add("Interpretation: `inbound=True` is a message *from a customer to a brand*; "
        "`inbound=False` is a *brand reply*. Brand handles are preserved in text "
        "(e.g. `@sprintcare`) while customer handles are anonymized to numeric ids.")
    add("")

    add("## 7. Brand / Handle Distribution")
    add("")
    add(f"- Unique author ids overall (brands + anonymized customers): **{s['unique_authors']:,}**")
    add(f"- Unique brand / support handles (authors of outbound tweets): **{s['unique_brands']:,}**")
    add("")
    add(f"Top {top_brands} brands by reply volume (outbound tweet count):")
    add("")
    add("| Rank | Brand handle | Brand replies |")
    add("| --- | --- | --- |")
    for rank, (brand, cnt) in enumerate(s["top_brands"], start=1):
        add(f"| {rank} | `{brand}` | {cnt:,} |")
    add("")
    add("_Reply volume is the most reliable brand-activity signal in Phase 1 because "
        "brand handles are never anonymized. Per-brand customer/thread counts are computed "
        "during brand selection (Phase 2)._")
    add("")

    add("## 8. Threading Information")
    add("")
    add(f"- Rows with `response_tweet_id` (received at least one reply): {s['has_response']:,} "
        f"({pct(s['has_response'], s['total_rows'])})")
    add(f"- Rows with `in_response_to_tweet_id` (are themselves a reply): {s['has_in_response_to']:,} "
        f"({pct(s['has_in_response_to'], s['total_rows'])})")
    add(f"- Thread roots (no `in_response_to_tweet_id`) - approx. conversation count: "
        f"**{s['roots']:,}**")
    add(f"- Mid-thread rows (both reply to something and get a reply): {s['mid_thread']:,} "
        f"- evidence of multi-turn conversations")
    add(f"- Rows whose `response_tweet_id` lists multiple ids (branching replies): {s['branching']:,}")
    add("")
    add(f"- Conversations reconstructable from `tweet_id` <-> `response_tweet_id` / "
        f"`in_response_to_tweet_id` links: **{'YES' if reconstructable else 'NO'}**")
    add("")

    add("## 9. Timestamp Range")
    add("")
    add(f"- `created_at` format: `{CREATED_AT_FORMAT}` (e.g. `Tue Oct 31 22:10:47 +0000 2017`)")
    add(f"- Earliest: {s['ts_min']}")
    add(f"- Latest: {s['ts_max']}")
    add(f"- Unparseable timestamps: {s['ts_parse_fail']:,} ({pct(s['ts_parse_fail'], s['total_rows'])})")
    add("")

    add("## 10. Data Quality Issues")
    add("")
    add("- `response_tweet_id` and `in_response_to_tweet_id` are sparse by design: not every "
        "tweet starts or ends a thread, so high null rates in these columns are expected, not errors.")
    add("- `response_tweet_id` can contain multiple comma-separated ids (branching threads); the "
        "pipeline must split these when reconstructing conversations.")
    add("- Customer ids are anonymized integers; only brand handles are human-readable.")
    add("- Text contains mentions, URLs, emoji and non-English messages; light normalization will "
        "be needed for retrieval/classification while preserving original text for reply grounding.")
    if s["empty_text"] > 0:
        add(f"- {s['empty_text']:,} empty/whitespace-only `text` rows were found and should be dropped.")
    else:
        add("- No empty/whitespace-only `text` rows were found; the pipeline will still guard against them.")
    add("- Some brand replies are generic 'please DM us' redirects; the proportion of these per "
        "brand is a key groundability signal to measure during brand selection.")
    add("")

    add("## 11. Verification: Is this the Customer Support on Twitter dataset?")
    add("")
    add(f"- Expected columns present and in order: **{'YES' if schema_ok else 'NO'}**")
    add(f"  - Expected: `{', '.join(EXPECTED_COLUMNS)}`")
    add(f"  - Found:    `{', '.join(s['header_columns'])}`")
    add("- Brand handles observed (e.g. `sprintcare`, and the top-brands table above) and the "
        "`inbound` customer/brand flag match the known thoughtvector/customer-support-on-twitter "
        "schema.")
    add(f"- Conclusion: **{'CONFIRMED' if schema_ok else 'MISMATCH - investigate'}** this is the "
        "Customer Support on Twitter dataset.")
    add("")

    add("## 12. Recommended Brand-Selection Strategy (for Phase 2)")
    add("")
    add("Rank candidate brands using measurable, evaluation-relevant criteria rather than fame:")
    add("")
    add("1. **Volume**: enough reply volume and reconstructable customer->brand pairs to build a "
        "retrieval corpus, a 150-250 example golden set, and train/val/test splits without "
        "starving any split. Use the top-brands table as the starting shortlist.")
    add("2. **Groundability**: measure the share of brand replies that are generic redirects "
        "(e.g. 'please DM us', 'sorry to hear, DM us'). Prefer brands whose replies carry real "
        "resolution content so grounded reply generation is a meaningful task.")
    add("3. **Intent diversity**: sample customer messages and check they span several "
        "distinguishable problems (login, billing, outage, delivery, etc.) so a small 6-12 "
        "intent taxonomy is non-trivial and a confusion matrix is informative.")
    add("4. **Reconstructable multi-turn threads**: prefer brands with a healthy count of linked "
        "customer->brand pairs, since grounding and escalation depend on real resolutions.")
    add("5. **Language / noise**: prefer predominantly English, lower duplicate/near-duplicate "
        "rate, so hand-labelling and LLM judging stay reliable.")
    add("6. **Reproducibility budget**: the brand's working subsample must let the headline "
        "results reproduce in under 15 minutes.")
    add("")
    add("Avoid brands that are mostly DM-redirects (nothing to ground), heavily multilingual, "
        "dominated by a single intent (no room to stratify), or so large they blow the runtime "
        "budget. Phase 2 will compute per-brand customer messages, brand replies, reconstructable "
        "pairs, reply-length distribution, DM-redirect ratio, and a rough intent spread before "
        "committing to one brand.")
    add("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    print(f"Inspecting {args.path} in chunks of {args.chunksize:,} rows ...")
    stats = inspect(args.path, args.chunksize, args.top_brands)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    md = build_markdown(stats, args.chunksize, args.top_brands)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(md)

    # concise stdout summary
    print("\n===== SUMMARY =====")
    print(f"rows                 : {stats['total_rows']:,}")
    print(f"columns              : {stats['header_columns']}")
    print(f"customer messages    : {stats['inbound_true']:,}")
    print(f"brand replies        : {stats['inbound_false']:,}")
    print(f"unique brand handles : {stats['unique_brands']:,}")
    print(f"unique authors       : {stats['unique_authors']:,}")
    print(f"thread roots (approx): {stats['roots']:,}")
    print(f"has response_tweet_id: {stats['has_response']:,}")
    print(f"timestamp range      : {stats['ts_min']} -> {stats['ts_max']}")
    print(f"top 5 brands         : {stats['top_brands'][:5]}")
    print(f"\nReport written to    : {args.out}")


if __name__ == "__main__":
    main()
