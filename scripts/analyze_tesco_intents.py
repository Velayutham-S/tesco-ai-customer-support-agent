"""
Phase 4 (Step 2) - Analyze Tesco customer messages to derive/validate the intent taxonomy.

Loads the Phase 3 corpus, keeps only USABLE customer -> Tesco pairs, and produces evidence for
the taxonomy: primary-intent distribution (via the deterministic rule labeller), raw keyword
coverage per intent, top unigrams/bigrams, ambiguity rate, and real (redacted) examples per
intent. Output: reports/tesco_intent_analysis.md (+ a small JSON of counts).

This is analysis only - it assigns PROVISIONAL labels to size the taxonomy; it does not create
the golden set.

Usage:
  python scripts/analyze_tesco_intents.py --corpus data/processed/tesco_interactions.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import config
from src import intent_taxonomy as tax
from src.text_utils import clean_for_heuristic

# Stopwords for unigram informativeness (English + Twitter/support noise). Bigrams keep function words.
STOPWORDS = set("""
a an the and or but if then so of to in on at for with from by as is are was were be been being
i you he she it we they me my your our their his her its this that these those there here what which
who whom whose when where why how do does did doing have has had having will would can could should
shall may might must not no yes im ive dont cant wont didnt doesnt isnt arent id ill ive u ur ya youre
just get got getting one now still back going go get like really very much more most any some out up
down off over please pls thanks thank hi hello hey oh ok okay amp via dm rt http https co tesco
""".split())

TOKEN_RE = re.compile(r"[a-z']{2,}")


def load_corpus(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        alt = path.replace(".parquet", ".csv")
        if os.path.exists(alt):
            path = alt
        else:
            print(f"ERROR: corpus not found at {path}", file=sys.stderr)
            sys.exit(1)
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    return df


def analyze(df: pd.DataFrame) -> dict:
    usable = df[df["is_usable"] == True].copy()  # noqa: E712
    texts = usable["customer_text"].astype(str).tolist()
    redacted = usable["customer_redacted_text"].astype(str).tolist()

    primary_counts = Counter()
    raw_match_counts = Counter()
    ambiguous = 0
    unigrams = Counter()
    bigrams = Counter()
    examples: dict[str, list[str]] = {name: [] for name in tax.PRIORITY_ORDER}
    example_seen: dict[str, set] = {name: set() for name in tax.PRIORITY_ORDER}

    for text, red in zip(texts, redacted):
        res = tax.assign_intent(text)
        primary_counts[res["intent"]] += 1
        for m in res["matched"]:
            raw_match_counts[m] += 1
        if res["is_ambiguous"]:
            ambiguous += 1

        clean = clean_for_heuristic(text)
        toks = TOKEN_RE.findall(clean)
        for t in toks:
            if t not in STOPWORDS:
                unigrams[t] += 1
        for a, b in zip(toks, toks[1:]):
            bigrams[(a, b)] += 1

        # collect up to 6 short, distinct redacted examples per primary intent
        pi = res["intent"]
        if len(examples[pi]) < 6:
            snippet = " ".join(str(red).split())
            key = snippet.lower()[:80]
            if 0 < len(snippet) <= 160 and key not in example_seen[pi]:
                example_seen[pi].add(key)
                examples[pi].append(snippet)

    total = len(usable)
    return {
        "total_usable": total,
        "primary_counts": primary_counts,
        "raw_match_counts": raw_match_counts,
        "ambiguous": ambiguous,
        "unigrams": unigrams,
        "bigrams": bigrams,
        "examples": examples,
    }


def write_reports(a: dict, reports_dir: str) -> tuple[str, str]:
    os.makedirs(reports_dir, exist_ok=True)
    total = a["total_usable"]
    pc = a["primary_counts"]
    rc = a["raw_match_counts"]

    def pct(n):
        return f"{100.0 * n / total:.2f}%" if total else "0%"

    # JSON counts
    counts_json = {
        "total_usable_customer_messages": total,
        "primary_intent_distribution": {k: int(pc.get(k, 0)) for k in tax.PRIORITY_ORDER},
        "raw_keyword_match_counts": {k: int(rc.get(k, 0)) for k in tax.PRIORITY_ORDER if k != tax.OTHER_INTENT},
        "ambiguous_messages": a["ambiguous"],
        "taxonomy_version": tax.TAXONOMY_VERSION,
    }
    json_path = os.path.join(reports_dir, "tesco_intent_analysis.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(counts_json, fh, indent=2)

    lines: list[str] = []
    add = lines.append
    add("# Tesco Intent Analysis (evidence for the taxonomy)")
    add("")
    add(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by `scripts/analyze_tesco_intents.py`._")
    add("")
    add(f"- Usable customer messages analyzed: **{total:,}**")
    add(f"- Ambiguous messages (>=2 intents matched): **{a['ambiguous']:,}** ({pct(a['ambiguous'])})")
    add(f"- Taxonomy version: {tax.TAXONOMY_VERSION}")
    add("- Labels here are PROVISIONAL (deterministic keyword/priority rules), used only to size the taxonomy.")
    add("")
    add("## Primary-intent distribution (priority-resolved)")
    add("")
    add("| Intent | Count | % |")
    add("| --- | ---: | ---: |")
    for name in tax.PRIORITY_ORDER:
        add(f"| `{name}` | {pc.get(name, 0):,} | {pct(pc.get(name, 0))} |")
    add("")
    add("## Raw keyword coverage (message matches this intent's rule, before priority)")
    add("")
    add("| Intent | Messages matching rule | % |")
    add("| --- | ---: | ---: |")
    for name in tax.PRIORITY_ORDER:
        if name == tax.OTHER_INTENT:
            continue
        add(f"| `{name}` | {rc.get(name, 0):,} | {pct(rc.get(name, 0))} |")
    add("")
    add("## Top 30 unigrams (stopwords removed)")
    add("")
    add("`" + ", ".join(f"{w}:{c}" for w, c in a["unigrams"].most_common(30)) + "`")
    add("")
    add("## Top 30 bigrams (raw)")
    add("")
    add("`" + ", ".join(f"{w1}_{w2}:{c}" for (w1, w2), c in a["bigrams"].most_common(30)) + "`")
    add("")
    add("## Representative (redacted) examples per intent")
    for name in tax.PRIORITY_ORDER:
        add("")
        add(f"### `{name}`  (n={pc.get(name, 0):,})")
        exs = a["examples"].get(name, [])
        if not exs:
            add("- (no short examples captured)")
        for e in exs:
            add(f"- {e}")
    add("")
    md_path = os.path.join(reports_dir, "tesco_intent_analysis.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return md_path, json_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Tesco intents (evidence for taxonomy).")
    p.add_argument("--corpus", default=os.path.join(config.PROCESSED_DIR, "tesco_interactions.parquet"))
    p.add_argument("--reports-dir", default=config.REPORTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading corpus: {args.corpus}", flush=True)
    df = load_corpus(args.corpus)
    print(f"  rows={len(df):,}; usable={int((df['is_usable']==True).sum()):,}", flush=True)  # noqa: E712
    a = analyze(df)
    md_path, json_path = write_reports(a, args.reports_dir)

    print("\n===== INTENT ANALYSIS SUMMARY =====", flush=True)
    print(f"usable messages : {a['total_usable']:,}")
    print(f"ambiguous       : {a['ambiguous']:,}")
    print("primary distribution:")
    for name in tax.PRIORITY_ORDER:
        n = a["primary_counts"].get(name, 0)
        print(f"  {name:28s} {n:6,d}  ({100.0*n/max(a['total_usable'],1):5.2f}%)")
    print(f"\nreports:\n  {md_path}\n  {json_path}")


if __name__ == "__main__":
    main()
