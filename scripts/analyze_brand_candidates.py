"""
Phase 2 - Brand candidate analysis and selection for the Hiver SDE assignment.

Goal: from the Customer Support on Twitter dataset (dataset/twcs/twcs.csv), analyze a
shortlist of candidate brands with measurable, evaluation-relevant criteria and select ONE
brand for the AI support agent. The selection score is transparent and is deliberately NOT
a function of raw reply volume alone, so the biggest brand does not win automatically.

Design (memory-bounded, deterministic, no full-file load):
  * Pass 0: stream the `tweet_id` column only to learn the max id -> size the forest arrays.
  * Pass 1: stream all needed columns in chunks. For every row we (vectorized):
      - record inbound flag, text-validity and (for candidate brands) the outbound author code,
      - set a parent pointer child -> in_response_to_tweet_id (this reconstructs threads exactly:
        every tweet is a root or has a parent, so the reply graph is a complete forest),
      - attribute inbound (customer) messages to a candidate brand via the first candidate @handle
        mentioned, and accumulate customer-side metrics (unique authors, duplicates, url/mention,
        english heuristic, intent-category hits),
      - for outbound candidate-brand replies, compute reply-level flags (valid / generic / meaningful
        / short / url) and store compact records for pair reconstruction.
  * Forest roots are found by vectorized pointer-jumping (no Python per-node loop).
  * Per-brand conversation metrics (count, multi-turn, avg/median length) and pair metrics
    (connected / usable) are computed with numpy masks.
  * A weighted score combines groundability, usable volume, intent diversity, multi-turn rate,
    data quality and reproducibility. Weights and formula are documented in the report.

Outputs (in --output-dir, default reports/):
  brand_candidates.csv, brand_selection_report.md, selected_brand_summary.json

Usage:
  python scripts/analyze_brand_candidates.py --dataset dataset/twcs/twcs.csv --output-dir reports
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Configuration: candidate shortlist (verified against the dataset at runtime)
# --------------------------------------------------------------------------------------
CANDIDATE_BRANDS = [
    "AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares", "Delta", "Tesco",
    "AmericanAir", "TMobileHelp", "comcastcares", "British_Airways", "SouthwestAir",
    "VirginTrains", "Ask_Spectrum", "XboxSupport", "sprintcare", "hulu_support",
    "sainsburys", "GWRHelp", "AskPlayStation", "ChipotleTweets", "VerizonSupport",
    "UPSHelp", "ATVIAssist", "O2", "Safaricom_Care",
]

# --------------------------------------------------------------------------------------
# Text heuristics (all lower-cased matching, transparent and rule-based)
# --------------------------------------------------------------------------------------
URL_PAT = r"https?://\S+|\bt\.co/\S+|www\.\S+"
MENTION_PAT = r"@\w+"
DIGIT_PAT = r"\d"
WS_RE = re.compile(r"\s+")

# Redirection / low-information phrases (a single hit is NOT enough to call a reply generic).
REDIRECT_PHRASES = [
    "dm us", "dm you", "dm me", "send us a dm", "send a dm", "shoot us a dm", "pls dm",
    "please dm", "drop us a dm", "send us a private message", "private message", "send us a pm",
    "send us a message", "message us", "direct message", "click 'message'", "click message",
    "follow us", "follow and dm", "reach out via", "in a dm", "via dm", "send us your details",
    "sent you a dm", "check your dm", "we've dm'd", "we have dm",
]
# Terms that indicate a reply carries actionable / resolution content.
ACTIONABLE_TERMS = [
    "try", "settings", "update", "reset", "restart", "reboot", "reinstall", "password",
    "refund", "refunded", "track", "tracking", "order", "email", "link", "step", "steps",
    "check", "confirm", "account", "code", "status", "upgrade", "cancel", "replace",
    "replacement", "credit", "voucher", "dispatch", "delivery", "delivered", "payment",
    "charge", "charged", "invoice", "verify", "app", "version", "browser", "clear cache",
    "sim", "network", "coverage", "book", "booking", "flight", "seat", "gate",
    "sorted", "resolved", "fixed", "here's", "here is", "you can", "please go to", "visit",
]
# English-token heuristic (crude presence check; NOT a real language detector).
# Non-capturing groups so pandas str.contains does not warn about match groups.
ENGLISH_STOPWORDS_PAT = (
    r"\b(?:the|you|your|and|for|please|not|with|have|this|that|are|is|to|my|me|we|it|of|on|in|can|will|"
    r"i|a|do|no|why|how|when|what|get|got|need|help|been|was|but|so|if|at)\b"
)
REDIRECT_PAT = "(?:" + "|".join(re.escape(p) for p in REDIRECT_PHRASES) + ")"
ACTIONABLE_PAT = r"\b(?:" + "|".join(re.escape(t) for t in ACTIONABLE_TERMS) + r")\b"

# Intent / issue categories (only used to estimate diversity, not the final taxonomy).
# Non-capturing groups so pandas str.contains does not warn about match groups.
INTENT_CATEGORIES = {
    "account_login": r"\b(?:account|log ?in|login|sign ?in|signin|password|locked|verify|verification|otp|username|credential)\b",
    "payment_billing": r"\b(?:payment|pay|bill|billing|charge|charged|invoice|card|overcharge|debit)\b",
    "refund": r"\b(?:refund|money back|reimburse|chargeback)\b",
    "cancellation": r"\b(?:cancel|cancelled|canceled|cancellation|unsubscribe|terminate)\b",
    "delivery_order": r"\b(?:order|delivery|deliver|delivered|package|parcel|shipping|shipment|dispatch|courier|track|tracking)\b",
    "technical_issue": r"\b(?:not working|doesn'?t work|won'?t|error|bug|crash|crashing|broken|glitch|freeze|frozen|stuck|fault)\b",
    "outage": r"\b(?:down|outage|offline|no service|no signal|can'?t connect|not connecting|server)\b",
    "subscription": r"\b(?:subscription|subscribe|subscribed|plan|renew|renewal|premium|membership|trial)\b",
    "travel_booking": r"\b(?:flight|booking|book|delayed|delay|gate|boarding|seat|baggage|luggage|train|ticket|reschedule|check ?in)\b",
    "complaint": r"\b(?:worst|terrible|awful|disappointed|disappointing|ridiculous|unacceptable|complaint|rubbish|furious|angry)\b",
    "info_request": r"\b(?:how do i|how to|how can i|can i|when will|what is|where is|do you|is there)\b",
    "product_issue": r"\b(?:screen|battery|volume|sound|charging|charger|device|phone|headphone|controller|game|hardware|button)\b",
}

# --------------------------------------------------------------------------------------
# Scoring configuration (documented in the report)
# --------------------------------------------------------------------------------------
VOL_TARGET_USABLE = 5000        # usable pairs considered "comfortably enough" for corpus+splits+golden
GOLDEN_MIN = 250                # upper end of the required 150-250 golden set
TEST_FRACTION = 0.15            # held-out fraction used to size evaluation_capacity
REPRO_SOFT = 60_000             # brand replies below this -> no reproducibility penalty
REPRO_HARD = 200_000            # brand replies at/above this -> maximum penalty (score 0.5)
SHORT_REPLY_MAX_WORDS = 6       # content-word count threshold for "short"

WEIGHTS = {
    "groundability": 0.30,      # meaningful (non-generic) reply share - core to grounded generation
    "usable_volume": 0.20,      # enough usable customer->brand pairs
    "intent_diversity": 0.15,   # spread of issue types (normalized entropy)
    "multi_turn": 0.10,         # share of conversations that are genuine multi-turn
    "data_quality": 0.10,       # english-token share x (1 - reply duplicate rate)
    "golden_capacity": 0.10,    # can we draw a 150-250 golden set
    "reproducibility": 0.05,    # smaller corpus is faster to reproduce under the 15-min budget
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def clean_series(s: pd.Series) -> pd.Series:
    """Lower-case, strip @mentions and URLs, collapse whitespace. Vectorized."""
    s = s.fillna("")
    s = s.str.replace(URL_PAT, " ", regex=True, flags=re.IGNORECASE)
    s = s.str.replace(MENTION_PAT, " ", regex=True)
    s = s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def normalized_entropy(counts: dict) -> float:
    """Shannon entropy of category hit-counts, normalized to [0,1] over the number of categories."""
    total = sum(counts.values())
    k = len(INTENT_CATEGORIES)
    if total <= 0 or k <= 1:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    return round(ent / math.log(k), 4)


# --------------------------------------------------------------------------------------
# Core analysis
# --------------------------------------------------------------------------------------
def pass0_max_id(path: str, chunksize: int) -> int:
    max_id = 0
    for chunk in pd.read_csv(path, usecols=["tweet_id"], chunksize=chunksize, encoding="utf-8"):
        cmax = pd.to_numeric(chunk["tweet_id"], errors="coerce").max()
        if pd.notna(cmax):
            max_id = max(max_id, int(cmax))
    return max_id


def analyze(path: str, chunksize: int) -> dict:
    if not os.path.exists(path):
        print(f"ERROR: dataset not found at {path}", file=sys.stderr)
        sys.exit(1)

    handles_lower = {b.lower(): i for i, b in enumerate(CANDIDATE_BRANDS)}
    nbrands = len(CANDIDATE_BRANDS)
    cand_alt = "|".join(re.escape(b) for b in CANDIDATE_BRANDS)
    cand_capture_pat = r"@(" + cand_alt + r")\b"

    print("Pass 0: scanning tweet_id column for max id ...", flush=True)
    max_id = pass0_max_id(path, chunksize)
    n = max_id + 1
    print(f"  max tweet_id = {max_id:,}; allocating forest arrays of size {n:,}", flush=True)

    # forest / lookup arrays indexed by tweet_id
    parent = np.arange(n, dtype=np.int64)
    inbound_arr = np.zeros(n, dtype=np.int8)
    has_text_arr = np.zeros(n, dtype=np.int8)
    exists_arr = np.zeros(n, dtype=np.int8)
    brand_code_arr = np.full(n, -1, dtype=np.int16)

    # customer-side accumulators (per brand)
    cust_count = np.zeros(nbrands, dtype=np.int64)
    cust_url = np.zeros(nbrands, dtype=np.int64)
    cust_mention = np.zeros(nbrands, dtype=np.int64)
    cust_english = np.zeros(nbrands, dtype=np.int64)
    cust_authors: list[set] = [set() for _ in range(nbrands)]
    cust_seen: list[set] = [set() for _ in range(nbrands)]
    intent_counts = {name: np.zeros(nbrands, dtype=np.int64) for name in INTENT_CATEGORIES}

    # reply-side records (compact arrays appended per chunk)
    r_brand_parts, r_parent_parts = [], []
    r_valid_parts, r_generic_parts, r_meaningful_parts = [], [], []
    r_short_parts, r_url_parts, r_len_parts = [], [], []
    reply_seen: list[set] = [set() for _ in range(nbrands)]  # unique reply text hashes per brand

    usecols = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]
    total = 0
    print("Pass 1: streaming full dataset in chunks ...", flush=True)
    for ci, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                                           dtype=str, encoding="utf-8", on_bad_lines="warn")):
        total += len(chunk)
        ids = pd.to_numeric(chunk["tweet_id"], errors="coerce").fillna(0).astype(np.int64).values
        inbound_true = (chunk["inbound"] == "True").values
        text = chunk["text"].fillna("")
        cleaned = clean_series(text)
        valid_text = (cleaned.str.len() > 0).values
        good = ids > 0

        # lookup arrays
        exists_arr[ids[good]] = 1
        inbound_arr[ids[good & inbound_true]] = 1
        has_text_arr[ids[good & valid_text]] = 1

        # parent pointers (child -> parent)
        par = pd.to_numeric(chunk["in_response_to_tweet_id"], errors="coerce").values
        pmask = good & ~np.isnan(par)
        p_int = np.zeros(len(chunk), dtype=np.int64)
        p_int[pmask] = par[pmask].astype(np.int64)
        inrange = pmask & (p_int >= 1) & (p_int <= max_id)
        parent[ids[inrange]] = p_int[inrange]

        author_lower = chunk["author_id"].fillna("").str.lower()

        # ---- outbound candidate-brand replies (vectorized) ----
        out_mask = (~inbound_true) & author_lower.isin(handles_lower.keys()).values
        if out_mask.any():
            b_idx = author_lower[out_mask].map(handles_lower).astype(np.int64).values
            sub_ids = ids[out_mask]
            brand_code_arr[sub_ids] = b_idx.astype(np.int16)
            raw_out = text[out_mask]
            clean_out = cleaned[out_mask]
            valid = (clean_out.str.len() > 0).values
            nwords = clean_out.str.split().str.len().fillna(0).astype(np.int64).values
            has_url = raw_out.str.contains(URL_PAT, regex=True, flags=re.IGNORECASE).values
            has_digit = clean_out.str.contains(DIGIT_PAT, regex=True).values
            has_redirect = clean_out.str.contains(REDIRECT_PAT, regex=True).values
            has_action = has_url | has_digit | clean_out.str.contains(ACTIONABLE_PAT, regex=True).values
            is_short = nwords <= SHORT_REPLY_MAX_WORDS
            signals = has_redirect.astype(int) + is_short.astype(int) + (~has_action).astype(int)
            is_generic = valid & (signals >= 2)
            is_meaningful = valid & ~is_generic

            r_brand_parts.append(b_idx)
            r_parent_parts.append(p_int[out_mask])
            r_valid_parts.append(valid)
            r_generic_parts.append(is_generic)
            r_meaningful_parts.append(is_meaningful)
            r_short_parts.append(is_short)
            r_url_parts.append(has_url)
            r_len_parts.append(nwords)

            # unique reply text per brand (for duplicate counting)
            ho = pd.util.hash_pandas_object(clean_out, index=False).values
            dfr = pd.DataFrame({"b": b_idx, "h": ho, "valid": valid})
            for name, g in dfr[dfr["valid"]].groupby("b"):
                reply_seen[int(name)].update(g["h"].tolist())

        # ---- inbound customer messages attributed to a candidate brand (vectorized) ----
        if inbound_true.any():
            in_raw = text[inbound_true]
            in_clean = cleaned[inbound_true]
            in_auth = chunk["author_id"].fillna("").values[inbound_true]
            first_handle = in_raw.str.extract(cand_capture_pat, flags=re.IGNORECASE, expand=False)
            b_series = first_handle.str.lower().map(handles_lower)
            attr = b_series.notna().values
            if attr.any():
                b_in = b_series[attr].astype(np.int64).values
                clean_attr = in_clean[attr]
                raw_attr = in_raw[attr]
                auth_attr = in_auth[attr]

                cust_count += np.bincount(b_in, minlength=nbrands)
                cust_url += np.bincount(b_in, weights=raw_attr.str.contains(URL_PAT, regex=True, flags=re.IGNORECASE).values,
                                        minlength=nbrands).astype(np.int64)
                cust_mention += np.bincount(b_in, weights=raw_attr.str.contains(MENTION_PAT, regex=True).values,
                                            minlength=nbrands).astype(np.int64)
                cust_english += np.bincount(b_in, weights=clean_attr.str.contains(ENGLISH_STOPWORDS_PAT, regex=True, flags=re.IGNORECASE).values,
                                            minlength=nbrands).astype(np.int64)
                for name, pat in INTENT_CATEGORIES.items():
                    hits = clean_attr.str.contains(pat, regex=True, flags=re.IGNORECASE).values
                    intent_counts[name] += np.bincount(b_in, weights=hits, minlength=nbrands).astype(np.int64)

                # unique authors + duplicate messages (per brand) via groupby
                h_in = pd.util.hash_pandas_object(clean_attr, index=False).values
                dfi = pd.DataFrame({"b": b_in, "author": auth_attr, "h": h_in})
                for name, g in dfi.groupby("b"):
                    cust_authors[int(name)].update(g["author"].tolist())
                    cust_seen[int(name)].update(g["h"].tolist())

        if (ci + 1) % 3 == 0:
            print(f"  processed {total:,} rows", flush=True)

    print(f"Pass 1 done. total rows = {total:,}", flush=True)

    # ---- resolve forest roots via vectorized pointer jumping ----
    print("Resolving conversation roots (pointer jumping) ...", flush=True)
    for _ in range(200):
        newp = parent[parent]
        if np.array_equal(newp, parent):
            break
        parent = newp
    roots = parent

    comp_size = np.bincount(roots, weights=exists_arr.astype(np.float64), minlength=n)
    comp_inbound = np.bincount(roots, weights=inbound_arr.astype(np.float64), minlength=n)
    comp_outbound = comp_size - comp_inbound

    # ---- assemble reply records ----
    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.array([], dtype=dtype)

    rb = cat(r_brand_parts, np.int64)
    rp = cat(r_parent_parts, np.int64)
    rv = cat(r_valid_parts, bool)
    rg = cat(r_generic_parts, bool)
    rm = cat(r_meaningful_parts, bool)
    rs = cat(r_short_parts, bool)
    ru = cat(r_url_parts, bool)

    rp_valid = (rp >= 1) & (rp <= max_id)
    rp_safe = np.where(rp_valid, rp, 0)
    parent_exists = (exists_arr[rp_safe] == 1) & rp_valid
    parent_inbound = (inbound_arr[rp_safe] == 1) & parent_exists
    parent_hastext = (has_text_arr[rp_safe] == 1) & parent_exists

    results = []
    for b, brand in enumerate(CANDIDATE_BRANDS):
        mask_b = rb == b
        brand_reply_count = int(mask_b.sum())
        if brand_reply_count == 0:
            results.append({"brand": brand, "present": False})
            continue

        valid_replies = int((mask_b & rv).sum())
        generic_replies = int((mask_b & rg).sum())
        meaningful_replies = int((mask_b & rm).sum())
        short_replies = int((mask_b & rs).sum())
        url_replies = int((mask_b & ru).sum())
        duplicate_replies = max(valid_replies - len(reply_seen[b]), 0)

        connected = int((mask_b & parent_exists).sum())
        usable_mask = mask_b & rv & parent_inbound & parent_hastext
        usable = int(usable_mask.sum())
        cust_with_resp = int(np.unique(rp[usable_mask]).size)

        brand_nodes = np.where(brand_code_arr == b)[0]
        comp_roots_b = np.unique(roots[brand_nodes])
        conversation_count = int(comp_roots_b.size)
        sizes_b = comp_size[comp_roots_b]
        inb_b = comp_inbound[comp_roots_b]
        outb_b = comp_outbound[comp_roots_b]
        multi_turn = int(np.sum((sizes_b >= 3) & (inb_b >= 1) & (outb_b >= 1)))
        avg_len = float(np.mean(sizes_b)) if sizes_b.size else 0.0
        med_len = float(np.median(sizes_b)) if sizes_b.size else 0.0

        cmsg = int(cust_count[b])
        uniq_cust = len(cust_authors[b])
        english_pct = round(100.0 * cust_english[b] / cmsg, 2) if cmsg else None
        cust_dup = cmsg - len(cust_seen[b]) if cmsg else 0
        cust_dup_rate = round(cust_dup / cmsg, 4) if cmsg else None
        counts_dict = {name: int(intent_counts[name][b]) for name in INTENT_CATEGORIES}
        diversity = normalized_entropy(counts_dict)
        active_categories = int(sum(1 for c in counts_dict.values() if cmsg and c / cmsg >= 0.02))

        meaningful_pct = round(100.0 * meaningful_replies / valid_replies, 2) if valid_replies else 0.0
        generic_pct = round(100.0 * generic_replies / valid_replies, 2) if valid_replies else 0.0
        reply_dup_rate = round(duplicate_replies / valid_replies, 4) if valid_replies else 0.0

        results.append({
            "brand": brand, "present": True,
            "brand_reply_count": brand_reply_count,
            "customer_message_count": cmsg,
            "unique_customer_count": uniq_cust,
            "connected_pair_count": connected,
            "usable_pair_count": usable,
            "conversation_count": conversation_count,
            "multi_turn_conversation_count": multi_turn,
            "average_conversation_length": round(avg_len, 3),
            "median_conversation_length": round(med_len, 3),
            "customer_messages_with_direct_response": cust_with_resp,
            "brand_replies_with_text": valid_replies,
            "generic_reply_count": generic_replies,
            "generic_reply_percentage": generic_pct,
            "meaningful_reply_count": meaningful_replies,
            "meaningful_reply_percentage": meaningful_pct,
            "duplicate_count": int(duplicate_replies),
            "short_reply_count": short_replies,
            "url_message_count": int(cust_url[b]),
            "mention_message_count": int(cust_mention[b]),
            "customer_duplicate_rate": cust_dup_rate,
            "approximate_english_percentage": english_pct,
            "intent_active_categories": active_categories,
            "intent_diversity_score": diversity,
            "golden_set_capacity": usable,
            "evaluation_capacity": int(usable * TEST_FRACTION),
            "_reply_dup_rate": reply_dup_rate,
            "_url_replies": url_replies,
            "_intent_counts": counts_dict,
        })

    return {"total_rows": total, "max_id": max_id, "results": results}


def score_brands(results: list[dict]) -> list[dict]:
    """Attach transparent sub-scores and the final weighted selection_score."""
    scored = []
    for r in results:
        if not r.get("present"):
            continue
        usable = r["usable_pair_count"]
        s_ground = (r["meaningful_reply_percentage"] or 0) / 100.0
        s_volume = min(usable / VOL_TARGET_USABLE, 1.0)
        s_div = r["intent_diversity_score"] or 0.0
        s_multi = (r["multi_turn_conversation_count"] / r["conversation_count"]) if r["conversation_count"] else 0.0
        eng = (r["approximate_english_percentage"] or 0) / 100.0
        s_quality = max(0.0, eng * (1.0 - (r["_reply_dup_rate"] or 0.0)))
        s_golden = 1.0 if usable >= GOLDEN_MIN else usable / GOLDEN_MIN
        rc = r["brand_reply_count"]
        if rc <= REPRO_SOFT:
            s_repro = 1.0
        else:
            over = min((rc - REPRO_SOFT) / (REPRO_HARD - REPRO_SOFT), 1.0)
            s_repro = 1.0 - 0.5 * over
        subs = {
            "groundability": round(s_ground, 4),
            "usable_volume": round(s_volume, 4),
            "intent_diversity": round(s_div, 4),
            "multi_turn": round(s_multi, 4),
            "data_quality": round(s_quality, 4),
            "golden_capacity": round(s_golden, 4),
            "reproducibility": round(s_repro, 4),
        }
        r["sub_scores"] = subs
        r["selection_score"] = round(sum(WEIGHTS[k] * subs[k] for k in WEIGHTS), 4)
        scored.append(r)
    scored.sort(key=lambda x: x["selection_score"], reverse=True)
    return scored


def weakness_reason(r: dict, best: dict) -> str:
    reasons = []
    if r["meaningful_reply_percentage"] < best["meaningful_reply_percentage"] - 3:
        reasons.append(f"lower meaningful-reply share ({r['meaningful_reply_percentage']}% vs "
                       f"{best['meaningful_reply_percentage']}%, i.e. more generic/DM-redirect replies)")
    if r["sub_scores"]["reproducibility"] < 0.9:
        reasons.append(f"reproducibility penalty (large corpus: {r['brand_reply_count']:,} replies)")
    if r["intent_diversity_score"] < best["intent_diversity_score"] - 0.05:
        reasons.append(f"narrower issue mix (diversity {r['intent_diversity_score']} vs {best['intent_diversity_score']})")
    mt_r = r["multi_turn_conversation_count"] / max(r["conversation_count"], 1)
    mt_b = best["multi_turn_conversation_count"] / max(best["conversation_count"], 1)
    if mt_r < mt_b - 0.1:
        reasons.append(f"fewer multi-turn conversations ({round(100*mt_r)}% vs {round(100*mt_b)}% of threads)")
    if r["usable_pair_count"] < best["usable_pair_count"] * 0.5:
        reasons.append(f"fewer usable pairs ({r['usable_pair_count']:,})")
    if not reasons:
        reasons.append(f"lower overall score ({r['selection_score']} vs {best['selection_score']})")
    return "; ".join(reasons[:2])


def build_reasons(best: dict) -> list[str]:
    conv = max(best["conversation_count"], 1)
    return [
        f"Meaningful (non-generic) reply share = {best['meaningful_reply_percentage']}% of "
        f"{best['brand_replies_with_text']:,} text replies -> strong grounding material.",
        f"{best['usable_pair_count']:,} usable customer->brand pairs -> enough for a retrieval corpus, "
        f"train/val/test splits, and a 150-250 golden set without starving any split.",
        f"Intent diversity (normalized entropy) = {best['intent_diversity_score']} across "
        f"{best['intent_active_categories']} active issue categories -> non-trivial 6-12 intent taxonomy.",
        f"Multi-turn conversations = {best['multi_turn_conversation_count']:,} of {best['conversation_count']:,} "
        f"({round(100.0*best['multi_turn_conversation_count']/conv,1)}%) -> real resolution context.",
        f"Approx. English share = {best['approximate_english_percentage']}% (heuristic) -> reliable "
        f"hand-labelling and LLM judging.",
        f"Reproducibility sub-score = {best['sub_scores']['reproducibility']} ({best['brand_reply_count']:,} "
        f"replies) -> fits the under-15-minute reproduction budget.",
    ]


def build_limitations() -> list[str]:
    return [
        "Customer messages are attributed to a brand via the first candidate @handle mentioned; a message "
        "naming multiple candidate brands is counted once (first match).",
        "English share is a crude stopword-presence heuristic, not a real language detector.",
        "The generic-reply label is a transparent multi-signal heuristic (redirect phrase + short + no "
        "actionable content); it approximates, not perfectly measures, low-information replies.",
        "Threads are reconstructed via `in_response_to_tweet_id` only (complete here because every tweet is "
        "a root or has a parent); rare cross-thread quoting is not modelled.",
        "usable_pair_count is measured before removing duplicate replies; duplicate_count is reported "
        "separately so the effect is visible.",
        "All metrics are brand-specific; the selected brand's numbers are not expected to generalize.",
    ]


# --------------------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------------------
CSV_COLUMNS = [
    "brand", "brand_reply_count", "customer_message_count", "unique_customer_count",
    "connected_pair_count", "usable_pair_count", "conversation_count",
    "multi_turn_conversation_count", "average_conversation_length", "median_conversation_length",
    "customer_messages_with_direct_response", "brand_replies_with_text",
    "generic_reply_count", "generic_reply_percentage", "meaningful_reply_count",
    "meaningful_reply_percentage", "duplicate_count", "short_reply_count",
    "url_message_count", "mention_message_count", "approximate_english_percentage",
    "intent_active_categories", "intent_diversity_score", "golden_set_capacity",
    "evaluation_capacity", "selection_score",
]


def write_csv(scored: list[dict], out_dir: str) -> str:
    df = pd.DataFrame(scored)
    for c in CSV_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[CSV_COLUMNS].sort_values("selection_score", ascending=False)
    path = os.path.join(out_dir, "brand_candidates.csv")
    df.to_csv(path, index=False)
    return path


def write_json(best: dict, scored: list[dict], missing: list[str], out_dir: str) -> str:
    rejected = [{
        "brand": r["brand"],
        "selection_score": r["selection_score"],
        "main_reason": weakness_reason(r, best),
    } for r in scored if r["brand"] != best["brand"]]
    summary = {
        "selected_brand": best["brand"],
        "dataset_path": "dataset/twcs/twcs.csv",
        "brand_reply_count": best["brand_reply_count"],
        "customer_message_count": best["customer_message_count"],
        "usable_pair_count": best["usable_pair_count"],
        "conversation_count": best["conversation_count"],
        "multi_turn_conversation_count": best["multi_turn_conversation_count"],
        "generic_reply_percentage": best["generic_reply_percentage"],
        "meaningful_reply_percentage": best["meaningful_reply_percentage"],
        "intent_diversity_score": best["intent_diversity_score"],
        "golden_set_capacity": best["golden_set_capacity"],
        "evaluation_capacity": best["evaluation_capacity"],
        "selection_score": best["selection_score"],
        "sub_scores": best["sub_scores"],
        "scoring_weights": WEIGHTS,
        "selection_reasons": build_reasons(best),
        "rejected_alternatives": rejected[:8],
        "brands_not_found": missing,
        "known_limitations": build_limitations(),
    }
    path = os.path.join(out_dir, "selected_brand_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return path


def write_report(best: dict, scored: list[dict], missing: list[str], meta: dict,
                 out_dir: str, dataset_path: str, chunksize: int) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Brand Selection Report - Customer Support on Twitter")
    add("")
    add(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `scripts/analyze_brand_candidates.py`._")
    add("")
    add("## 1. Objective")
    add("Select ONE brand best suited for an AI customer-support agent: intent classification, "
        "historical-reply retrieval, grounded reply generation, escalation, a 150-250 example golden set, "
        "train/val/test splitting, failure analysis, and reproducible evaluation under 15 minutes.")
    add("")
    add("## 2. Dataset")
    add(f"- Path: `{dataset_path}`")
    add(f"- Rows scanned: {meta['total_rows']:,}; max tweet_id: {meta['max_id']:,}")
    add("")
    add("## 3. Dataset Processing Method")
    add(f"- Chunked streaming with `pandas.read_csv(usecols=..., dtype=str, chunksize={chunksize:,})`; the "
        "full file is never loaded at once.")
    add("- Pass 0 finds the max `tweet_id`; Pass 1 fills numpy lookup arrays and per-brand accumulators.")
    add("- Threads are reconstructed as a forest via `in_response_to_tweet_id` (child -> parent), then roots "
        "are found by vectorized pointer-jumping. This is exact here because every tweet is a root or has a parent.")
    add("")
    add("## 4. Candidate Selection Method")
    add(f"- Started from the {len(CANDIDATE_BRANDS)} shortlisted handles (top brands from the dataset profile); "
        "each is verified to exist before scoring.")
    add(f"- Not found in dataset (skipped): {', '.join('`'+m+'`' for m in missing) if missing else 'none'}.")
    add("")
    add("## 5. Conversation Reconstruction")
    add("- Parent pointer: `child.in_response_to_tweet_id -> parent.tweet_id`.")
    add("- A conversation = one connected tree (component) after pointer-jumping to roots.")
    add("- Conversation length = number of real tweets in the component.")
    add("- Multi-turn = component with >=3 messages AND >=1 customer message AND >=1 brand reply.")
    add("")
    add("## 6. Definition of a Usable Pair")
    add("A usable customer->brand pair requires ALL of: (1) a candidate-brand reply (`inbound=False`) with "
        "valid text; (2) its `in_response_to_tweet_id` resolves to an existing tweet; (3) that parent is a "
        "customer message (`inbound=True`) with valid text. Duplicates are reported separately; "
        "`usable_pair_count` is measured before duplicate removal.")
    add("")
    add("## 7. Generic-Reply Heuristic (multi-signal)")
    add("On mention/URL-stripped, lower-cased reply text, three signals are computed: `has_redirect` (a "
        "DM/redirect phrase), `is_short` (<= 6 content words), and `lacks_actionable` (no URL, digit, or "
        "resolution term). A reply is **generic** when >= 2 signals fire; otherwise, with valid text, it is "
        "**meaningful**. No single phrase alone marks a reply generic.")
    add("")
    add("## 8. Intent-Diversity Analysis")
    add(f"- {len(INTENT_CATEGORIES)} keyword issue-categories are matched against each brand's customer "
        "messages; diversity = normalized Shannon entropy of the category distribution in [0,1].")
    add("- `intent_active_categories` = categories each covering >= 2% of the brand's messages.")
    add("")
    add("## 9. Language & Noise Analysis")
    add("- `approximate_english_percentage`: share of customer messages containing a common English stopword "
        "(crude heuristic, not a language detector).")
    add("- Duplicate replies / customer messages: exact match of cleaned text within a brand.")
    add("")
    add("## 10. Selection Scoring Formula")
    add("Each sub-score is an absolute normalization in [0,1] (not min-max across candidates, for stability). "
        "The weighted sum is the `selection_score`:")
    add("")
    add("| Sub-score | Definition | Weight |")
    add("| --- | --- | ---: |")
    add(f"| groundability | meaningful_reply_% / 100 | {WEIGHTS['groundability']} |")
    add(f"| usable_volume | min(usable_pairs / {VOL_TARGET_USABLE:,}, 1) | {WEIGHTS['usable_volume']} |")
    add(f"| intent_diversity | normalized entropy | {WEIGHTS['intent_diversity']} |")
    add(f"| multi_turn | multi_turn_convs / conversations | {WEIGHTS['multi_turn']} |")
    add(f"| data_quality | english_share x (1 - reply_dup_rate) | {WEIGHTS['data_quality']} |")
    add(f"| golden_capacity | 1 if usable >= {GOLDEN_MIN} else usable/{GOLDEN_MIN} | {WEIGHTS['golden_capacity']} |")
    add(f"| reproducibility | 1.0 up to {REPRO_SOFT:,} replies, down to 0.5 at {REPRO_HARD:,} | {WEIGHTS['reproducibility']} |")
    add("")
    add("This intentionally prevents the largest brand from winning automatically: groundability, data "
        "quality and reproducibility penalize DM-heavy, noisy, or very large corpora.")
    add("")
    add("## 11. Comparison Table")
    add("")
    add("| Brand | Brand Replies | Usable Pairs | Conversations | Multi-turn | Meaningful % | Generic % | "
        "Intent Diversity | Golden Cap. | Score |")
    add("|------|--------------:|-------------:|--------------:|-----------:|-------------:|---------:|"
        "-----------------:|------------:|------:|")
    for r in scored:
        name = f"**{r['brand']}**" if r["brand"] == best["brand"] else r["brand"]
        add(f"| {name} | {r['brand_reply_count']:,} | {r['usable_pair_count']:,} | {r['conversation_count']:,} "
            f"| {r['multi_turn_conversation_count']:,} | {r['meaningful_reply_percentage']} "
            f"| {r['generic_reply_percentage']} | {r['intent_diversity_score']} | {r['golden_set_capacity']:,} "
            f"| {r['selection_score']} |")
    add("")
    add("## 12. Selected Brand")
    add(f"### -> `{best['brand']}`  (selection_score = {best['selection_score']})")
    add("")
    add("Key measured values:")
    add(f"- Brand replies: {best['brand_reply_count']:,} (with text: {best['brand_replies_with_text']:,})")
    add(f"- Customer messages (mention-attributed): {best['customer_message_count']:,} "
        f"(unique customers: {best['unique_customer_count']:,})")
    add(f"- Connected pairs: {best['connected_pair_count']:,}; usable pairs: {best['usable_pair_count']:,}")
    add(f"- Conversations: {best['conversation_count']:,}; multi-turn: {best['multi_turn_conversation_count']:,} "
        f"(avg len {best['average_conversation_length']}, median {best['median_conversation_length']})")
    add(f"- Meaningful replies: {best['meaningful_reply_percentage']}%; generic: {best['generic_reply_percentage']}%")
    add(f"- Intent diversity: {best['intent_diversity_score']} across {best['intent_active_categories']} active categories")
    add(f"- Approx. English: {best['approximate_english_percentage']}%; duplicate replies: {best['duplicate_count']:,}")
    add(f"- Golden-set capacity: {best['golden_set_capacity']:,}; evaluation capacity (~{int(TEST_FRACTION*100)}% test): {best['evaluation_capacity']:,}")
    add("")
    add("## 13. Why This Brand Is Suitable")
    for reason in build_reasons(best):
        add(f"- {reason}")
    add("")
    add("## 14. Why Other Brands Were Not Selected")
    for r in scored:
        if r["brand"] == best["brand"]:
            continue
        add(f"- `{r['brand']}` (score {r['selection_score']}): {weakness_reason(r, best)}")
    add("")
    add("## 15. Limitations")
    for lim in build_limitations():
        add(f"- {lim}")
    add("")
    add("## 16. Reproduction Command")
    add("```")
    add(f"python scripts/analyze_brand_candidates.py --dataset {dataset_path} --output-dir {out_dir}")
    add("```")
    add("")
    path = os.path.join(out_dir, "brand_selection_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze and select one brand (chunked, deterministic).")
    p.add_argument("--dataset", default="dataset/twcs/twcs.csv")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--chunksize", type=int, default=200_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = analyze(args.dataset, args.chunksize)
    all_results = data["results"]
    missing = [r["brand"] for r in all_results if not r.get("present")]
    scored = score_brands(all_results)
    if not scored:
        print("ERROR: no candidate brands found in dataset.", file=sys.stderr)
        sys.exit(1)
    best = scored[0]

    csv_path = write_csv(scored, args.output_dir)
    json_path = write_json(best, scored, missing, args.output_dir)
    md_path = write_report(best, scored, missing, data, args.output_dir, args.dataset, args.chunksize)

    print("\n===== PHASE 2 SUMMARY =====", flush=True)
    print(f"selected brand   : {best['brand']}  (score {best['selection_score']})")
    print(f"brand replies    : {best['brand_reply_count']:,}")
    print(f"usable pairs     : {best['usable_pair_count']:,}")
    print(f"conversations    : {best['conversation_count']:,} (multi-turn {best['multi_turn_conversation_count']:,})")
    print(f"meaningful reply : {best['meaningful_reply_percentage']}%  generic {best['generic_reply_percentage']}%")
    print(f"intent diversity : {best['intent_diversity_score']}")
    print(f"brands not found : {missing}")
    print(f"top 5 by score   : {[(r['brand'], r['selection_score']) for r in scored[:5]]}")
    print(f"\nfiles written:\n  {csv_path}\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
