"""
Phase 3 - Build a clean, reproducible Tesco-only conversation corpus.

From dataset/twcs/twcs.csv we extract customer -> Tesco reply pairs using the REAL response
relationships (tweet_id / in_response_to_tweet_id), not just the presence of "@Tesco" in the
text. Each pair keeps original + normalized + redacted text, conversation id (thread root),
turn order, parent-child link, timestamps, and quality flags.

Method (memory-bounded, deterministic, no full-file load):
  * Pass 0: stream `tweet_id` to size the forest arrays.
  * Pass A: stream all needed columns; fill inbound/exists/parent arrays; record every Tesco
    outbound reply (id, text, created_at, author, parent id); mark the parent ids we must fetch;
    mark Tesco thread nodes.
  * Pointer-jumping resolves each tweet to its conversation root (a complete forest: every tweet
    is a root or has a parent).
  * Pass B: stream again and fetch text/author/timestamp of the customer parents we need.
  * Assemble pairs, apply transparent data-quality + duplicate filters, redact PII, compute
    conversation-level metrics, and write parquet/csv + a profile + a summary.

Usage:
  python scripts/build_tesco_corpus.py --dataset dataset/twcs/twcs.csv --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

# make `src` importable when run as `python scripts/build_tesco_corpus.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import config
from src.text_utils import (
    normalize_text, extract_urls, has_real_content, reply_generic_signals,
    redact_text, word_count,
)

CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"
USECOLS_A = ["tweet_id", "author_id", "inbound", "created_at", "text", "in_response_to_tweet_id"]
USECOLS_B = ["tweet_id", "author_id", "inbound", "created_at", "text"]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def parse_created(raw) -> tuple[str, bool]:
    """Return (iso_utc_str, is_valid)."""
    ts = pd.to_datetime(raw, format=CREATED_AT_FORMAT, errors="coerce", utc=True)
    if pd.isna(ts):
        return "", False
    return ts.isoformat(), True


def dist_stats(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean": round(float(a.mean()), 3),
        "min": int(a.min()),
        "p10": int(np.percentile(a, 10)),
        "p25": int(np.percentile(a, 25)),
        "median": int(np.percentile(a, 50)),
        "p75": int(np.percentile(a, 75)),
        "p90": int(np.percentile(a, 90)),
        "p95": int(np.percentile(a, 95)),
        "max": int(a.max()),
    }


def length_histogram(sizes: np.ndarray) -> dict:
    buckets = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6-10": 0, "11+": 0}
    for s in sizes:
        s = int(s)
        if s <= 5:
            buckets[str(s)] += 1
        elif s <= 10:
            buckets["6-10"] += 1
        else:
            buckets["11+"] += 1
    return buckets


# --------------------------------------------------------------------------------------
# Passes
# --------------------------------------------------------------------------------------
def pass0_max_id(path: str, chunksize: int) -> int:
    max_id = 0
    for chunk in pd.read_csv(path, usecols=["tweet_id"], chunksize=chunksize, encoding="utf-8"):
        cmax = pd.to_numeric(chunk["tweet_id"], errors="coerce").max()
        if pd.notna(cmax):
            max_id = max(max_id, int(cmax))
    return max_id


def pass_a(path: str, chunksize: int, brand_lower: str, max_id: int):
    """Fill forest/lookup arrays and collect Tesco outbound reply records + needed parent ids."""
    n = max_id + 1
    parent = np.arange(n, dtype=np.int64)
    inbound_arr = np.zeros(n, dtype=np.int8)
    exists_arr = np.zeros(n, dtype=np.int8)
    tesco_node = np.zeros(n, dtype=np.int8)
    needed_arr = np.zeros(n, dtype=np.int8)

    r_id, r_text, r_created, r_author, r_parent = [], [], [], [], []

    total = 0
    for ci, chunk in enumerate(pd.read_csv(path, usecols=USECOLS_A, chunksize=chunksize,
                                           dtype=str, encoding="utf-8", on_bad_lines="warn")):
        total += len(chunk)
        ids = pd.to_numeric(chunk["tweet_id"], errors="coerce").fillna(0).astype(np.int64).values
        inbound_true = (chunk["inbound"] == "True").values
        text = chunk["text"].fillna("")
        good = ids > 0

        exists_arr[ids[good]] = 1
        inbound_arr[ids[good & inbound_true]] = 1

        par = pd.to_numeric(chunk["in_response_to_tweet_id"], errors="coerce").values
        p_int = np.zeros(len(chunk), dtype=np.int64)
        pmask = good & ~np.isnan(par)
        p_int[pmask] = par[pmask].astype(np.int64)
        inrange = pmask & (p_int >= 1) & (p_int <= max_id)
        parent[ids[inrange]] = p_int[inrange]

        author_lower = chunk["author_id"].fillna("").str.lower()
        out_mask = (~inbound_true) & (author_lower == brand_lower).values & good
        if out_mask.any():
            sub_ids = ids[out_mask]
            tesco_node[sub_ids] = 1
            r_id.extend(sub_ids.tolist())
            r_text.extend(text.values[out_mask].tolist())
            r_created.extend(chunk["created_at"].fillna("").values[out_mask].tolist())
            r_author.extend(chunk["author_id"].fillna("").values[out_mask].tolist())
            sub_par = p_int[out_mask]
            r_parent.extend(sub_par.tolist())
            pin = sub_par[(sub_par >= 1) & (sub_par <= max_id)]
            needed_arr[pin] = 1

        if (ci + 1) % 3 == 0:
            print(f"  [pass A] processed {total:,} rows; tesco replies so far = {len(r_id):,}", flush=True)

    records = {
        "id": np.array(r_id, dtype=np.int64),
        "text": r_text,
        "created_raw": r_created,
        "author": r_author,
        "parent": np.array(r_parent, dtype=np.int64),
    }
    arrays = {"parent": parent, "inbound": inbound_arr, "exists": exists_arr,
              "tesco_node": tesco_node, "needed": needed_arr}
    return records, arrays, total


def pass_b(path: str, chunksize: int, needed_arr: np.ndarray):
    """Fetch text/author/created_at/inbound for the needed parent (customer) tweets."""
    parent_data: dict[int, dict] = {}
    parent_created_invalid = 0
    total = 0
    for ci, chunk in enumerate(pd.read_csv(path, usecols=USECOLS_B, chunksize=chunksize,
                                           dtype=str, encoding="utf-8", on_bad_lines="warn")):
        total += len(chunk)
        ids = pd.to_numeric(chunk["tweet_id"], errors="coerce").fillna(0).astype(np.int64).values
        good = ids > 0
        need_mask = good & (needed_arr[ids] == 1)
        if need_mask.any():
            sub_ids = ids[need_mask]
            sub_text = chunk["text"].fillna("").values[need_mask]
            sub_auth = chunk["author_id"].fillna("").values[need_mask]
            sub_created = chunk["created_at"].fillna("").values[need_mask]
            sub_inb = (chunk["inbound"] == "True").values[need_mask]
            for k in range(len(sub_ids)):
                iso, ok = parse_created(sub_created[k])
                if not ok:
                    parent_created_invalid += 1
                parent_data[int(sub_ids[k])] = {
                    "text": sub_text[k], "author": sub_auth[k],
                    "created": iso, "inbound": bool(sub_inb[k]),
                }
        if (ci + 1) % 3 == 0:
            print(f"  [pass B] processed {total:,} rows; parents fetched = {len(parent_data):,}", flush=True)
    return {"parent_data": parent_data, "parent_created_invalid": parent_created_invalid}


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
def resolve_roots(parent: np.ndarray) -> np.ndarray:
    for _ in range(200):
        newp = parent[parent]
        if np.array_equal(newp, parent):
            break
        parent = newp
    return parent


def build_corpus(records, arrays, extra, max_id):
    parent = arrays["parent"]
    inbound_arr = arrays["inbound"]
    exists_arr = arrays["exists"]
    tesco_node = arrays["tesco_node"]

    roots = resolve_roots(parent)
    n = parent.shape[0]
    comp_size = np.bincount(roots, weights=exists_arr.astype(np.float64), minlength=n)
    comp_inbound = np.bincount(roots, weights=inbound_arr.astype(np.float64), minlength=n)

    tesco_roots = np.unique(roots[tesco_node == 1])
    conv_sizes = comp_size[tesco_roots]
    conv_inb = comp_inbound[tesco_roots]
    conv_out = conv_sizes - conv_inb
    thread_multi_turn = int(np.sum((conv_sizes >= 3) & (conv_inb >= 1) & (conv_out >= 1)))

    parent_data = extra["parent_data"]

    # deterministic order (by reply id) for duplicate resolution
    order = np.argsort(records["id"], kind="stable")
    rid = records["id"][order]
    rparent = records["parent"][order]
    rtext = [records["text"][i] for i in order]
    rauthor = [records["author"][i] for i in order]
    rcreated = [records["created_raw"][i] for i in order]

    counts = Counter()
    reasons = Counter()
    redaction_totals = Counter()
    pii_pairs = 0
    cust_lens, brand_lens = [], []
    generic_usable = 0
    tesco_created_valid = 0
    tesco_created_invalid = 0
    seen_pairs: set = set()
    rows = []

    for k in range(len(rid)):
        bid = int(rid[k])
        p = int(rparent[k])
        counts["tesco_replies"] += 1

        if p < 1 or p > max_id:
            counts["missing_no_parent_id"] += 1
            continue
        if exists_arr[p] == 0:
            counts["missing_parent_not_in_data"] += 1
            continue
        if inbound_arr[p] == 0:
            counts["reply_to_non_customer"] += 1
            continue
        pd_ = parent_data.get(p)
        if pd_ is None:
            counts["missing_parent_not_in_data"] += 1
            continue

        counts["candidate_pairs"] += 1

        cust_raw = pd_["text"]
        brand_raw = rtext[k]
        cust_norm = normalize_text(cust_raw)
        brand_norm = normalize_text(brand_raw)

        usable, reason = True, ""
        if not brand_norm:
            usable, reason = False, "empty_brand_text"
        elif not cust_norm:
            usable, reason = False, "empty_customer_text"
        elif not has_real_content(cust_raw):
            usable, reason = False, "customer_only_mention_or_url"
        elif not has_real_content(brand_raw):
            usable, reason = False, "brand_only_mention_or_url"

        dup_key = (cust_norm.lower(), brand_norm.lower())
        is_duplicate = dup_key in seen_pairs
        if usable and not is_duplicate:
            seen_pairs.add(dup_key)
        if usable and is_duplicate:
            usable, reason = False, "duplicate"

        sig = reply_generic_signals(brand_raw)
        is_generic = sig["is_generic"]

        cust_red, cred = redact_text(cust_raw)
        brand_red, bred = redact_text(brand_raw)
        if sum(cred.values()) + sum(bred.values()) > 0:
            pii_pairs += 1
            for kk in ("url", "email", "phone", "number"):
                redaction_totals[kk] += cred[kk] + bred[kk]

        brand_iso, brand_ok = parse_created(rcreated[k])
        if brand_ok:
            tesco_created_valid += 1
        else:
            tesco_created_invalid += 1

        root = int(roots[bid])
        conv_size = int(comp_size[root])
        conv_inbound_n = int(comp_inbound[root])
        is_multi = conv_size >= 3 and conv_inbound_n >= 1 and (conv_size - conv_inbound_n) >= 1

        if usable:
            counts["usable_pairs"] += 1
            cust_lens.append(word_count(cust_raw))
            brand_lens.append(word_count(brand_raw))
            if is_generic:
                generic_usable += 1
        else:
            reasons[reason] += 1
        if is_duplicate:
            counts["duplicate_pairs"] += 1

        rows.append({
            "pair_id": f"tesco-{bid}",
            "conversation_id": f"tesco-conv-{root}",
            "conversation_message_count": conv_size,
            "is_multi_turn_conversation": is_multi,
            "customer_tweet_id": p,
            "customer_author_id": pd_["author"],
            "customer_inbound": True,
            "customer_created_at": pd_["created"],
            "customer_text": cust_raw,
            "customer_normalized_text": cust_norm,
            "customer_redacted_text": cust_red,
            "customer_urls": " | ".join(extract_urls(cust_raw)),
            "brand_tweet_id": bid,
            "brand_author_id": rauthor[k],
            "brand_inbound": False,
            "brand_created_at": brand_iso,
            "brand_text": brand_raw,
            "brand_normalized_text": brand_norm,
            "brand_redacted_text": brand_red,
            "brand_urls": " | ".join(extract_urls(brand_raw)),
            "in_response_to_tweet_id": p,
            "is_generic_reply": is_generic,
            "is_duplicate": is_duplicate,
            "is_usable": usable,
            "unusable_reason": reason,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["conversation_id", "customer_created_at", "brand_tweet_id"],
                            kind="stable").reset_index(drop=True)
        df["turn_in_conversation"] = df.groupby("conversation_id").cumcount() + 1

    stats = {
        "counts": dict(counts),
        "unusable_reasons": dict(reasons),
        "thread_conversation_count": int(tesco_roots.size),
        "thread_multi_turn_count": thread_multi_turn,
        "conversation_length_stats": dist_stats([int(x) for x in conv_sizes]),
        "conversation_length_histogram": length_histogram(conv_sizes),
        "customer_message_length_stats": dist_stats(cust_lens),
        "brand_reply_length_stats": dist_stats(brand_lens),
        "generic_usable_count": generic_usable,
        "pii_pairs_with_redaction": pii_pairs,
        "pii_redaction_totals": dict(redaction_totals),
        "tesco_created_valid": tesco_created_valid,
        "tesco_created_invalid": tesco_created_invalid,
        "parent_created_invalid": extra["parent_created_invalid"],
    }
    return df, stats


# --------------------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------------------
def write_outputs(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "tesco_interactions.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    parquet_path = os.path.join(out_dir, "tesco_interactions.parquet")
    parquet_ok, parquet_note = True, ""
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover
        parquet_ok, parquet_note, parquet_path = False, (
            f"Parquet not written ({exc.__class__.__name__}: {exc}); CSV is the source of truth."), None
    return csv_path, parquet_path, parquet_ok, parquet_note


def write_summary(df, stats, brand, dataset, out_dir, reports_dir, parquet_ok, parquet_note):
    usable = int(df["is_usable"].sum()) if not df.empty else 0
    conv_usable = int(df.loc[df["is_usable"], "conversation_id"].nunique()) if usable else 0
    c = stats["counts"]
    summary = {
        "selected_brand": brand,
        "dataset_path": dataset,
        "original_tesco_brand_replies": c.get("tesco_replies", 0),
        "connected_pairs": c.get("candidate_pairs", 0),
        "candidate_pairs": c.get("candidate_pairs", 0),
        "usable_pairs": usable,
        "duplicate_pairs": c.get("duplicate_pairs", 0),
        "conversations_with_usable_pairs": conv_usable,
        "thread_conversation_count": stats["thread_conversation_count"],
        "thread_multi_turn_count": stats["thread_multi_turn_count"],
        "unusable_reasons": stats["unusable_reasons"],
        "missing_relationship": {
            "no_parent_id": c.get("missing_no_parent_id", 0),
            "parent_not_in_data": c.get("missing_parent_not_in_data", 0),
            "reply_to_non_customer": c.get("reply_to_non_customer", 0),
        },
        "generic_usable_count": stats["generic_usable_count"],
        "generic_usable_percentage": round(100.0 * stats["generic_usable_count"] / usable, 2) if usable else 0.0,
        "conversation_length_stats": stats["conversation_length_stats"],
        "conversation_length_histogram": stats["conversation_length_histogram"],
        "customer_message_length_stats": stats["customer_message_length_stats"],
        "brand_reply_length_stats": stats["brand_reply_length_stats"],
        "pii_pairs_with_redaction": stats["pii_pairs_with_redaction"],
        "pii_redaction_totals": stats["pii_redaction_totals"],
        "timestamp_validity": {
            "tesco_reply_valid": stats["tesco_created_valid"],
            "tesco_reply_invalid": stats["tesco_created_invalid"],
            "customer_parent_invalid": stats["parent_created_invalid"],
        },
        "parquet_written": parquet_ok,
        "parquet_note": parquet_note,
        "reproduction_command": (
            f"python scripts/build_tesco_corpus.py --dataset {dataset} --output-dir {out_dir}"
        ),
    }
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "tesco_corpus_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return path, summary


def write_profile(summary, brand, dataset, out_dir, reports_dir, chunksize):
    s = summary
    lines: list[str] = []
    add = lines.append
    add("# Tesco Corpus Profile")
    add("")
    add(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `scripts/build_tesco_corpus.py`._")
    add("")
    add("## 1. Source dataset")
    add(f"- `{dataset}` (unmodified; the raw CSV is never written to).")
    add("## 2. Selected brand")
    add(f"- `{brand}` (from `reports/selected_brand_summary.json` via `src/config.py`).")
    add("## 3. Processing method")
    add(f"- Chunked streaming (`chunksize={chunksize:,}`, `dtype=str`); the full file is never loaded at once.")
    add("- Pass 0 sizes forest arrays; Pass A fills inbound/exists/parent arrays, records Tesco replies "
        "and marks needed parents; pointer-jumping resolves conversation roots; Pass B fetches parent "
        "customer messages.")
    add("- Pairs are built from real `in_response_to` links (a Tesco reply linked to its parent customer "
        "message), NOT from '@Tesco' mentions.")
    add("## 4. Original Tesco message count")
    add(f"- Tesco outbound messages (author = {brand}, inbound=False): **{s['original_tesco_brand_replies']:,}**")
    add("## 5. Original Tesco brand-reply count")
    add(f"- {s['original_tesco_brand_replies']:,} (this dataset identifies brand messages by author, so "
        "the Tesco message count equals the Tesco brand-reply count).")
    add("## 6. Connected pair count")
    add(f"- Tesco replies whose parent is an existing customer message: **{s['candidate_pairs']:,}**")
    add("## 7. Final usable pair count")
    add(f"- **{s['usable_pairs']:,}** usable customer -> Tesco pairs after quality + duplicate filtering.")
    add("## 8. Conversation count")
    add(f"- Tesco threads (forest components containing a Tesco reply): **{s['thread_conversation_count']:,}**")
    add(f"- Conversations represented among usable pairs: {s['conversations_with_usable_pairs']:,}")
    add("## 9. Multi-turn conversation count")
    add(f"- Threads with >=3 messages and both a customer and a brand turn: **{s['thread_multi_turn_count']:,}**")
    add("## 10. Conversation-length distribution (messages per Tesco thread)")
    cl = s["conversation_length_stats"]
    add(f"- mean {cl.get('mean')}, median {cl.get('median')}, p90 {cl.get('p90')}, max {cl.get('max')}")
    add(f"- histogram: `{s['conversation_length_histogram']}`")
    add("## 11. Customer-message-length distribution (words, usable pairs)")
    ml = s["customer_message_length_stats"]
    add(f"- mean {ml.get('mean')}, median {ml.get('median')}, p90 {ml.get('p90')}, max {ml.get('max')}")
    add("## 12. Brand-reply-length distribution (words, usable pairs)")
    bl = s["brand_reply_length_stats"]
    add(f"- mean {bl.get('mean')}, median {bl.get('median')}, p90 {bl.get('p90')}, max {bl.get('max')}")
    add("## 13. Duplicate statistics")
    add(f"- Duplicate pairs detected (exact normalized customer+brand text): **{s['duplicate_pairs']:,}** "
        "(excluded from usable).")
    add("## 14. Generic-reply statistics (within usable pairs)")
    add(f"- Generic replies: **{s['generic_usable_count']:,}** ({s['generic_usable_percentage']}%).")
    add("## 15. Missing relationship statistics")
    mr = s["missing_relationship"]
    add(f"- Tesco replies with no parent id: {mr['no_parent_id']:,}")
    add(f"- Parent referenced but not present in data: {mr['parent_not_in_data']:,}")
    add(f"- Tesco reply whose parent is not a customer (brand->brand etc.): {mr['reply_to_non_customer']:,}")
    add("## 16. Language / noise observations")
    add("- Text is predominantly English (Phase 2 heuristic ~93%); mentions, casing and punctuation are "
        "preserved in `*_text` and `*_normalized_text`.")
    add("- Unusable-reason breakdown: `" + json.dumps(s["unusable_reasons"]) + "`")
    add("## 17. PII-redaction statistics")
    add(f"- Pairs with >=1 redaction: **{s['pii_pairs_with_redaction']:,}**; totals: `{s['pii_redaction_totals']}`")
    add("- Rules: URLs -> [URL], emails -> [EMAIL], phone-like (>=~10 digits) -> [PHONE], long numbers "
        "(>=5 digits; order/reference) -> [NUM]. @mentions (incl. anonymized numeric handles) are preserved. "
        "Original text stays in `*_text`; redaction lives only in `*_redacted_text`.")
    add("## 18. Filtering rules")
    add("- USABLE iff: Tesco outbound reply with real content; parent is an existing customer "
        "(inbound=True) message with real content (not only a mention/URL/empty); and not an exact "
        "duplicate. Nothing is deleted silently - every reply is accounted for in the counts above.")
    add("## 19. Known limitations")
    add("- Threads reconstructed via `in_response_to` only (complete here: every tweet is a root or child); "
        "rare cross-thread quoting is not modelled.")
    add("- Duplicate detection is exact-match on normalized text; paraphrased near-duplicates are not merged.")
    add("- The generic-reply label and PII redaction are transparent heuristics (may over/under-fire).")
    add("- `customer_message_count` here is relationship-based (parents of Tesco replies), which differs "
        "from the mention-based count used during brand selection.")
    add("## 20. Reproduction command")
    add("```")
    add(s["reproduction_command"])
    add("```")
    add("")
    os.makedirs(reports_dir, exist_ok=True)
    path = os.path.join(reports_dir, "tesco_corpus_profile.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the Tesco-only conversation corpus (chunked, deterministic).")
    p.add_argument("--dataset", default=config.DATASET_PATH)
    p.add_argument("--output-dir", default=config.PROCESSED_DIR)
    p.add_argument("--reports-dir", default=config.REPORTS_DIR)
    p.add_argument("--brand", default=config.SELECTED_BRAND or "Tesco")
    p.add_argument("--chunksize", type=int, default=200_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.dataset):
        print(f"ERROR: dataset not found at {args.dataset}", file=sys.stderr)
        sys.exit(1)
    brand = args.brand
    if brand is None or brand.lower() != "tesco":
        print(f"WARNING: expected brand 'Tesco' (Phase 2) but got '{brand}'. Proceeding with '{brand}'.",
              file=sys.stderr)
    brand_lower = brand.lower()
    print(f"Building corpus for brand='{brand}' from {args.dataset}", flush=True)

    print("Pass 0: scanning tweet_id for max id ...", flush=True)
    max_id = pass0_max_id(args.dataset, args.chunksize)
    print(f"  max tweet_id = {max_id:,}", flush=True)

    print("Pass A: streaming full dataset ...", flush=True)
    records, arrays, total = pass_a(args.dataset, args.chunksize, brand_lower, max_id)
    print(f"  Tesco replies found = {len(records['id']):,}", flush=True)
    if len(records["id"]) == 0:
        print(f"ERROR: no '{brand}' replies found - brand validation failed.", file=sys.stderr)
        sys.exit(1)

    print("Pass B: fetching customer parents ...", flush=True)
    extra = pass_b(args.dataset, args.chunksize, arrays["needed"])

    print("Assembling corpus ...", flush=True)
    df, stats = build_corpus(records, arrays, extra, max_id)

    csv_path, parquet_path, parquet_ok, parquet_note = write_outputs(df, args.output_dir)
    json_path, summary = write_summary(df, stats, brand, args.dataset, args.output_dir,
                                       args.reports_dir, parquet_ok, parquet_note)
    md_path = write_profile(summary, brand, args.dataset, args.output_dir, args.reports_dir, args.chunksize)

    print("\n===== PHASE 3 SUMMARY =====", flush=True)
    print(f"brand                 : {brand}")
    print(f"tesco replies (raw)   : {summary['original_tesco_brand_replies']:,}")
    print(f"connected pairs       : {summary['candidate_pairs']:,}")
    print(f"usable pairs          : {summary['usable_pairs']:,}")
    print(f"duplicate pairs       : {summary['duplicate_pairs']:,}")
    print(f"conversations (thread): {summary['thread_conversation_count']:,} "
          f"(multi-turn {summary['thread_multi_turn_count']:,})")
    print(f"generic in usable     : {summary['generic_usable_count']:,} ({summary['generic_usable_percentage']}%)")
    print(f"pii pairs redacted    : {summary['pii_pairs_with_redaction']:,}")
    print(f"parquet written       : {parquet_ok} {('- '+parquet_note) if not parquet_ok else ''}")
    print(f"\nfiles:\n  {csv_path}\n  {parquet_path}\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
