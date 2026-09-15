"""Shared, lightweight text utilities for the Hiver AI support-agent project.

Single source of truth for:
  * light text normalization (whitespace only - preserves wording, mentions, negations,
    product names and order terms),
  * URL extraction into a separate field,
  * the multi-signal generic-reply heuristic (kept consistent with the Phase 2 analyzer),
  * a transparent PII redaction pass used to build annotation/evaluation-safe text.

Design principle: be conservative. We never stem, lemmatize, drop stopwords, or strip
negations/product/order terms. Original text is always preserved by the caller; these
helpers only produce additional derived fields.
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+|\bt\.co/\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")
DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"\w")

# Redirection / low-information phrases. A single hit is NOT enough to call a reply generic.
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
REDIRECT_RE = re.compile("(?:" + "|".join(re.escape(p) for p in REDIRECT_PHRASES) + ")", re.IGNORECASE)
ACTIONABLE_RE = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in ACTIONABLE_TERMS) + r")\b", re.IGNORECASE)
SHORT_REPLY_MAX_WORDS = 6

# PII redaction patterns (heuristic; applied to a separate redacted field only).
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().\-]{8,}\d)(?!\d)")
LONGNUM_RE = re.compile(r"(?<!@)\b\d{5,}\b")  # order/reference-like numbers, but never @mention ids


# --------------------------------------------------------------------------------------
# Normalization / extraction
# --------------------------------------------------------------------------------------
def normalize_text(text: object) -> str:
    """Light normalization: collapse repeated whitespace and strip ends. Nothing else.

    Mentions, URLs, punctuation, casing, negations and product/order terms are all kept.
    """
    if text is None:
        return ""
    return WS_RE.sub(" ", str(text)).strip()


def extract_urls(text: object) -> list[str]:
    """Return the list of URLs found in the text (order preserved)."""
    if text is None:
        return []
    return URL_RE.findall(str(text))


def clean_for_heuristic(text: object) -> str:
    """Aggressive clean used ONLY for heuristics/validity: lower-case, drop mentions+URLs."""
    if text is None:
        return ""
    t = URL_RE.sub(" ", str(text))
    t = MENTION_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip().lower()


def has_real_content(text: object) -> bool:
    """True if, after removing mentions and URLs, at least one word character remains.

    Used to reject messages that are only a mention, only a URL, or empty.
    """
    return bool(WORD_RE.search(clean_for_heuristic(text)))


# --------------------------------------------------------------------------------------
# Generic-reply heuristic (multi-signal)
# --------------------------------------------------------------------------------------
def reply_generic_signals(text: object) -> dict:
    """Compute transparent signals for whether a brand reply is low-information/generic."""
    raw = str(text or "")
    clean = clean_for_heuristic(raw)
    valid = len(clean) > 0
    n_words = len(clean.split())
    has_url = bool(URL_RE.search(raw))
    has_digit = bool(DIGIT_RE.search(clean))
    has_redirect = bool(REDIRECT_RE.search(clean))
    has_actionable = has_url or has_digit or bool(ACTIONABLE_RE.search(clean))
    is_short = n_words <= SHORT_REPLY_MAX_WORDS
    signals = int(has_redirect) + int(is_short) + int(not has_actionable)
    return {
        "valid": valid,
        "n_words": n_words,
        "has_url": has_url,
        "has_redirect": has_redirect,
        "has_actionable": has_actionable,
        "is_short": is_short,
        "signals": signals,
        "is_generic": valid and signals >= 2,
    }


def is_generic_reply(text: object) -> bool:
    return reply_generic_signals(text)["is_generic"]


def word_count(text: object) -> int:
    """Word count on the mention/URL-stripped content."""
    return len(clean_for_heuristic(text).split())


# --------------------------------------------------------------------------------------
# PII redaction (transparent, order-sensitive)
# --------------------------------------------------------------------------------------
def redact_text(text: object) -> tuple[str, dict]:
    """Return (redacted_text, counts) with URLs, emails, phones and long numbers masked.

    Redaction order (each replaced by a bracket token):
      1. URLs           -> [URL]
      2. emails         -> [EMAIL]
      3. phone-like     -> [PHONE]   (>= ~10 digits, allowing separators)
      4. long numbers   -> [NUM]     (>= 5 digit runs; order/reference numbers, never @mention ids)

    @mentions (including anonymized numeric customer handles like @12345) are preserved.
    """
    s = normalize_text(text)
    counts = {"url": 0, "email": 0, "phone": 0, "number": 0}
    s, counts["url"] = URL_RE.subn("[URL]", s)
    s, counts["email"] = EMAIL_RE.subn("[EMAIL]", s)
    s, counts["phone"] = PHONE_RE.subn("[PHONE]", s)
    s, counts["number"] = LONGNUM_RE.subn("[NUM]", s)
    return s, counts


if __name__ == "__main__":
    samples = [
        "@115712 Please send us a DM so we can help.",
        "@Tesco my order 123456789 hasn't arrived, call me on 07123 456789 or email me a@b.com",
        "@Tesco try resetting your Clubcard password at https://tesco.com/help and let us know",
        "@Tesco   ??? ",
    ]
    for s in samples:
        sig = reply_generic_signals(s)
        red, c = redact_text(s)
        print(repr(s))
        print("  generic:", sig["is_generic"], "| urls:", extract_urls(s), "| content:", has_real_content(s))
        print("  redacted:", red, "|", c)
