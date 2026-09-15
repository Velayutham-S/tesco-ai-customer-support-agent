"""Tesco intent taxonomy (Phase 4) - single source of truth.

Defines a small, Tesco-specific support-intent taxonomy plus a transparent, deterministic
rule-based labeller used to (a) size the taxonomy against the corpus (analysis) and
(b) assign PROVISIONAL labels to the golden set. The same module will back the later
intent classifier so intent names never get duplicated across the codebase.

Design notes:
  * Labels produced by `assign_intent` are PROVISIONAL (keyword/priority rules), not human
    ground truth. The golden set marks them as such.
  * Priority order resolves multi-keyword messages to a single primary intent: specific
    operational problems first, resolution asks (refund) next, broad complaint/info later,
    and `other_or_unclear` as the fallback.
  * Definitions describe *customer message types*. They deliberately do NOT assert Tesco
    business policies.
"""

from __future__ import annotations

import re

try:  # works both as `src.intent_taxonomy` and when run directly
    from src.text_utils import clean_for_heuristic
except ImportError:  # pragma: no cover
    from text_utils import clean_for_heuristic

TAXONOMY_VERSION = "1.0"
OTHER_INTENT = "other_or_unclear"

# Priority order: first matching pattern (top-down) wins as the primary intent.
PRIORITY_ORDER = [
    "clubcard_or_loyalty",
    "missing_or_incorrect_item",
    "delivery_issue",
    "online_order_issue",
    "product_availability",
    "refund_or_payment",
    "account_or_technical_issue",
    "store_experience",
    "complaint",
    "general_information",
    OTHER_INTENT,
]

# Regex keyword rules (matched against mention/URL-stripped, lower-cased customer text).
# Kept intentionally readable; each pattern is a set of Tesco-relevant surface signals.
_PATTERNS: dict[str, str] = {
    "clubcard_or_loyalty": r"\b(clubcard|club card|loyalty|reward|points|nectar)\b",
    "missing_or_incorrect_item": (
        r"\b(missing|substitut|replaced|wrong item|incorrect item|wrong order|"
        r"item[s]? missing|didn'?t (get|receive)|not received|never received|damaged|"
        r"squashed|smashed|short[- ]?dated|out of date item|missing item)\b"
    ),
    "delivery_issue": (
        r"\b(deliver|delivery|delivered|driver|slot|didn'?t (arrive|turn up|show)|"
        r"not (arrived|delivered|turned up)|no.show|late|arrive|dispatch|courier|van|"
        r"redeliver|delivery date)\b"
    ),
    "online_order_issue": (
        r"\b(online order|grocery order|my order|order number|place an order|placing an order|"
        r"amend|cancel (my )?order|checkout|basket|order (didn'?t|won'?t|hasn'?t)|orders?)\b"
    ),
    "product_availability": (
        # `availab\w*` matches available/availability/availabilities; `unavailable` handled explicitly
        # (its leading "un" blocks the \b before "availab"). Fixes the old dead `availab\b` token.
        r"\b(out of stock|in stock|stock|sold out|unavailable|availab\w*|when (will|are) .* back|"
        r"restock|discontinu|not available|no longer (sell|stock))\b"
    ),
    "refund_or_payment": (
        r"\b(refund|refunded|money back|reimburse|overcharg|over charged|charged (me|twice)?|"
        r"double charged|payment|paid|declined|voucher|gift card|e[- ]?coupon|debited|"
        r"card (declined|payment))\b"
    ),
    "account_or_technical_issue": (
        r"\b(log ?in|log ?in|sign ?in|password|reset|account|website|web site|app|site|"
        r"error|crash|glitch|not working|won'?t load|can'?t (log|access|sign))\b"
    ),
    "store_experience": (
        r"\b(store|branch|shop|in[- ]?store|staff|colleague|manager|queue|till|checkout|"
        r"car ?park|petrol|fuel|express|superstore|metro|aisle|shelf)\b"
    ),
    "complaint": (
        r"\b(complaint|complain|disgusting|appalling|disgrace|disgraceful|worst|terrible|"
        r"awful|rubbish|shocking|fuming|furious|mould|mouldy|rotten|inedible|expired|"
        r"out of date|poor (service|quality)|disappointed|unhappy|rude|unacceptable)\b"
    ),
    "general_information": (
        r"\b(how do i|how can i|how much|can i|could you|do you (have|sell|know|do)|"
        r"is there|are there|what time|opening (times|hours)|when (do|are|is) you|"
        r"what (is|are)|where (can|is|do)|price of|question|enquir|query|info(rmation)?)\b"
    ),
}
_COMPILED: dict[str, re.Pattern] = {k: re.compile(v, re.IGNORECASE) for k, v in _PATTERNS.items()}

# Human-readable metadata (used by the taxonomy report). Describes MESSAGE TYPES only.
INTENT_META: dict[str, dict] = {
    "clubcard_or_loyalty": {
        "definition": "Questions/problems about Clubcard, loyalty points, rewards or loyalty vouchers.",
        "includes": "Clubcard points not credited, missing loyalty vouchers, Clubcard account/app queries.",
        "excludes": "Refunds unrelated to loyalty (-> refund_or_payment); general vouchers used as refunds.",
        "expected_reply": "Acknowledge the loyalty issue and route to the correct Clubcard support channel/details.",
    },
    "missing_or_incorrect_item": {
        "definition": "An order arrived but items are missing, substituted, wrong, or damaged.",
        "includes": "Missing items, unwanted substitutions, wrong/incorrect items, damaged goods on arrival.",
        "excludes": "Whole order never arrived (-> delivery_issue); product simply out of stock (-> product_availability).",
        "expected_reply": "Apologise, request order details securely, and explain how the missing/incorrect item is handled.",
    },
    "delivery_issue": {
        "definition": "Problems with a home delivery: non-arrival, lateness, slots, or the driver.",
        "includes": "Delivery didn't arrive/late, missed slot, driver problem, redelivery, delivery date queries.",
        "excludes": "Order arrived but items wrong/missing (-> missing_or_incorrect_item); placing/amending an order (-> online_order_issue).",
        "expected_reply": "Acknowledge the delivery problem, gather order/slot details, and explain next steps.",
    },
    "online_order_issue": {
        "definition": "Placing, amending, cancelling or checking out an online grocery order.",
        "includes": "Cannot place/amend/cancel order, checkout/basket problems, order status queries.",
        "excludes": "Delivery of a placed order (-> delivery_issue); pure website/app login errors (-> account_or_technical_issue).",
        "expected_reply": "Guide the customer on the ordering step and request order details where needed.",
    },
    "product_availability": {
        "definition": "Whether a product is in stock, sold out, or (re)available.",
        "includes": "Out of stock, when back in stock, discontinued/no longer stocked queries.",
        "excludes": "Item missing from a delivered order (-> missing_or_incorrect_item).",
        "expected_reply": "Give availability guidance without inventing stock promises; suggest checking store/site.",
    },
    "refund_or_payment": {
        "definition": "Refunds, charges and payment problems (the money side).",
        "includes": "Refund requests/status, overcharging/double charge, payment declined, gift card/voucher money.",
        "excludes": "The underlying operational cause when that is the main point (-> delivery/missing/etc.).",
        "expected_reply": "Acknowledge the payment/refund concern and route to secure account verification.",
    },
    "account_or_technical_issue": {
        "definition": "Website/app/account technical problems (login, errors, crashes).",
        "includes": "Cannot log in, password reset, site/app errors or crashes, account access.",
        "excludes": "Ordering-flow problems (-> online_order_issue); Clubcard-specific (-> clubcard_or_loyalty).",
        "expected_reply": "Offer basic troubleshooting and route to the right technical/account channel.",
    },
    "store_experience": {
        "definition": "Experience at a physical store (staff, queues, tills, car park, fuel).",
        "includes": "In-store staff/service, queues/tills, car park, petrol station, shelf/aisle issues.",
        "excludes": "Online ordering/delivery (-> online_order_issue/delivery_issue).",
        "expected_reply": "Acknowledge the in-store experience and offer to pass feedback / route to the store team.",
    },
    "complaint": {
        "definition": "Product-quality or general dissatisfaction not tied to a specific operational category.",
        "includes": "Poor product quality (mouldy/rotten/out of date), strong dissatisfaction, general complaints.",
        "excludes": "Complaints that are clearly about delivery/missing/refund (-> those specific intents).",
        "expected_reply": "Empathise, apologise, and route the complaint to the appropriate resolution channel.",
    },
    "general_information": {
        "definition": "General questions/requests for information (not a specific fault).",
        "includes": "Opening times, do-you-sell/stock, prices, how-to and policy questions.",
        "excludes": "Anything describing a specific fault/problem (-> the relevant issue intent).",
        "expected_reply": "Answer the question factually or point to where the information can be found.",
    },
    OTHER_INTENT: {
        "definition": "Anything that does not clearly fit another intent, or is too ambiguous/short.",
        "includes": "Chit-chat, thanks, unclear or context-free messages, off-topic tweets.",
        "excludes": "Messages that clearly match a defined intent.",
        "expected_reply": "Ask a clarifying question or route to a human if intent cannot be determined.",
    },
}


def intent_matches(text: object) -> list[str]:
    """Return all intents whose keyword rule matches (excludes the fallback)."""
    clean = clean_for_heuristic(text)
    if not clean:
        return []
    return [name for name in PRIORITY_ORDER if name != OTHER_INTENT and _COMPILED[name].search(clean)]


def assign_intent(text: object) -> dict:
    """Deterministically assign a PROVISIONAL primary intent.

    Returns dict: intent (primary), matched (all matched intents), is_ambiguous (>=2 matched),
    rule (short human-readable reason).
    """
    matched = intent_matches(text)
    if not matched:
        return {"intent": OTHER_INTENT, "matched": [], "is_ambiguous": False,
                "rule": "no keyword rule matched"}
    # priority = first in PRIORITY_ORDER that matched
    primary = next(name for name in PRIORITY_ORDER if name in matched)
    return {
        "intent": primary,
        "matched": matched,
        "is_ambiguous": len(matched) >= 2,
        "rule": f"keyword rule '{primary}'" + (f"; also matched {', '.join(m for m in matched if m != primary)}"
                                               if len(matched) >= 2 else ""),
    }


if __name__ == "__main__":
    tests = [
        "@Tesco my delivery never turned up and items are missing",
        "@Tesco I want a refund, I was charged twice",
        "@Tesco do you still sell the finest coffee? out of stock everywhere",
        "@Tesco my clubcard points are missing",
        "@Tesco your app keeps crashing when I try to log in",
        "@Tesco the queue at the Express store was huge and staff were rude",
        "@Tesco thanks!",
    ]
    print("taxonomy version", TAXONOMY_VERSION, "| intents:", len(PRIORITY_ORDER))
    for t in tests:
        print(assign_intent(t)["intent"], "<-", t)
