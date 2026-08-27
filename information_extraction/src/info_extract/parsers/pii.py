"""Two-tier PII redaction for parsed documents.

Tier 1 (envelope): PII carried by the email itself — recipient/sender addresses,
routing and identity headers.

Tier 2 (content): PII inside the invoice body — postal addresses, email
addresses, phone numbers, card numbers, personal names.

Redaction is deterministic and value-stable: the same value always maps to the
same placeholder (``[EMAIL_1]``, ``[ADDRESS_2]``, ...) within one document, so
document structure and coreference survive while the raw value does not. The
redaction report only carries counts and placeholders — never the original
values.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# --- Categories -------------------------------------------------------------

EMAIL = "EMAIL"
PHONE = "PHONE"
ADDRESS = "ADDRESS"
CARD = "CARD"
PERSON = "PERSON"


# --- Header classification (tier 1) -----------------------------------------

#: Headers naming the recipient — always masked in full (this is the user).
RECIPIENT_HEADERS = frozenset(
    {
        "to",
        "cc",
        "bcc",
        "delivered-to",
        "envelope-to",
        "x-original-to",
        "x-forwarded-to",
        "x-rcpt-to",
        "x-envelope-to",
    }
)

#: Headers naming the sender — the domain is a vendor signal worth keeping.
SENDER_HEADERS = frozenset({"from", "reply-to", "sender", "x-sender"})

#: Routing/identity headers with no extraction value that leak identifiers.
ROUTING_HEADERS = frozenset(
    {
        "received",
        "return-path",
        "errors-to",
        "message-id",
        "in-reply-to",
        "references",
        "dkim-signature",
        "domainkey-signature",
        "authentication-results",
        "arc-authentication-results",
        "arc-message-signature",
        "arc-seal",
        "x-google-dkim-signature",
        "x-gm-message-state",
        "x-received",
        "x-google-smtp-source",
        "list-id",
        "list-unsubscribe",
        "list-unsubscribe-post",
        "feedback-id",
        "x-feedback-id",
        "x-report-abuse",
        "x-campaign-id",
        "x-sg-eid",
        "x-mandrill-user",
        "x-ses-outgoing",
        "received-spf",
        "x-originating-ip",
        "x-sender-ip",
        "x-original-sender",
        "x-authenticated-sender",
    }
)

#: Header name prefixes/substrings that mark routing metadata (SPF results,
#: ARC chains, Google/ESP internals) — these carry envelope addresses verbatim.
ROUTING_HEADER_PREFIXES = ("received", "arc-", "x-google", "x-gm-", "x-ms-exchange")
ROUTING_HEADER_SUBSTRINGS = ("-spf", "dkim")


def is_routing_header(name: str) -> bool:
    lname = name.lower()
    return (
        lname in ROUTING_HEADERS
        or lname.startswith(ROUTING_HEADER_PREFIXES)
        or any(s in lname for s in ROUTING_HEADER_SUBSTRINGS)
    )


# --- Patterns ---------------------------------------------------------------

# Matches plain and URL-encoded (%40) addresses; also `name [at] host` forms.
EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+(?:@|%40|\s?\[at\]\s?)[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    re.IGNORECASE,
)
EMAIL_PARTS_RE = re.compile(
    r"([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
    re.IGNORECASE,
)

# Requires separators between digit groups so order numbers and totals survive.
PHONE_RE = re.compile(
    r"""(?<![\w.])
        (?:(?:\+\d{1,3}|1)[\s.\-]\s?)?      # optional country code
        (?:\(\d{3}\)\s*|\d{3}[\s.\-])       # area code
        \d{3}[\s.\-]\d{4}
        (?!\d)""",
    re.VERBOSE,
)
TEL_URI_RE = re.compile(r"tel:\+?\d[\d\s().\-]{5,}\d", re.IGNORECASE)

# `**** **** **** 4321`, `xxxx-1234`, `•••• 1234`. `#` is deliberately not a mask
# char here: `#1234` is a store/order number and `&#8199;` is an HTML entity.
CARD_MASKED_RE = re.compile(r"(?:[*x•·]{2,}[\s\-–]?){1,4}\d{2,4}(?!\d)", re.IGNORECASE)
# A single mask char, as in `American Express *5678` — require a full 4 digits so
# footnote markers (`*2 items`) are left alone.
CARD_SHORT_MASK_RE = re.compile(r"(?<![\w*])[*x•·][\s\-]?\d{4}(?!\d)", re.IGNORECASE)
# `ending in 5678`, `ending with 1003` — keeps the phrasing, drops the digits.
CARD_ENDING_RE = re.compile(
    r"(ending\s+(?:in|with)\s+|ending\s+)(\d{4})(?!\d)", re.IGNORECASE
)
# Full PAN, validated with Luhn to avoid eating order numbers.
CARD_PAN_RE = re.compile(r"(?<![\d\-])(?:\d{4}[\s\-]?){3}\d{1,4}(?![\d\-])")

STREET_SUFFIXES = (
    "St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ct|Court|Dr|Drive|Ln|Lane|Way|"
    "Pl|Place|Ter|Terrace|Cir|Circle|Pkwy|Parkway|Hwy|Highway|Sq|Square|Trl|Trail|"
    "Loop|Row|Alley|Path|Walk|Crescent|Close|Mews"
)
UNIT_KEYWORDS = "#|Apt\\.?|Apartment|Suite|Ste\\.?|Unit|Fl\\.?|Floor|Rm\\.?|Room"
STREET_RE = re.compile(
    rf"(?<![\w])\d{{1,6}}[A-Za-z]?\s+(?:[\w.'\-]+\s+){{0,4}}(?:{STREET_SUFFIXES})\b\.?"
    rf"(?:\s*,?\s*(?:{UNIT_KEYWORDS})\s*[\w\-]+)?",
    re.IGNORECASE,
)
_CITY = r"[A-Z][A-Za-z.'\-]+(?:[ ][A-Z][A-Za-z.'\-]+){0,3}"
# "Madison, WI 53703" / "Springfield, CA, 62704"
CITY_STATE_ZIP_RE = re.compile(
    rf"(?<![\w]){_CITY},?\s+[A-Z]{{2}}\b,?\s+\d{{5}}(?:-\d{{4}})?(?![\d])"
)
# "Springfield 62704-1234 CA" — some vendors reorder city/zip/state
CITY_ZIP_STATE_RE = re.compile(
    rf"(?<![\w]){_CITY},?\s+\d{{5}}(?:-\d{{4}})?\s+[A-Z]{{2}}\b(?![\w])"
)
# ZIP+4 is specific enough to mask on its own.
ZIP4_RE = re.compile(r"(?<![\d\-])\d{5}-\d{4}(?![\d\-])")
PO_BOX_RE = re.compile(r"P\.?\s?O\.?\s*Box\s*\d+", re.IGNORECASE)
UNIT_LINE_RE = re.compile(rf"^(?:(?:{UNIT_KEYWORDS})\s*)?[\w\-#/]{{1,8}}$", re.IGNORECASE)
ADDRESS_TOKEN_RE = re.compile(r"\[ADDRESS_\d+\]")

# Name hints in the body.
GREETING_RE = re.compile(
    r"(?:^|\n|\.\s)\s*(?:Hi|Hello|Hey|Dear)[, ]+([A-Z][a-z'\-]+(?: [A-Z][a-z'\-]+)?)\b"
)
# Case-sensitive on purpose: the capture must look like a capitalized name, so
# only the label alternation is case-insensitive (via a scoped flag). A global
# re.IGNORECASE here would let a label like "Shipping address" be captured as a
# name and then masked throughout the body.
THANKS_RE = re.compile(
    r"(?i:thanks|thank you)[^,\n]{0,40},\s*([A-Z][a-z'\-]+(?: [A-Z][a-z'\-]+)?)"
    r"(?=[!?.,;\n]|$)"
)
SHIP_TO_RE = re.compile(
    r"(?i:ship(?:ping|ped)?\s+(?:to|address)|bill(?:ing)?\s+(?:to|address)|"
    r"deliver(?:y|ed)?\s+(?:to|address)|mailing\s+address|sold\s+to|attn\.?|"
    r"recipient|(?:customer\s+)?name)\s*:?\s*\n?\s*"
    r"([A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){1,2})\b"
)

#: Words that are roles, providers, or document labels — never masked as names.
#: Guards the name heuristics: a mislearned common word would be masked
#: everywhere in the body and destroy line-item text.
NAME_STOPWORDS = frozenset(
    {
        "address",
        "arrives",
        "card",
        "delivery",
        "details",
        "express",
        "free",
        "gift",
        "ground",
        "invoice",
        "item",
        "items",
        "method",
        "payment",
        "shipping",
        "standard",
        "store",
        "subtotal",
        "summary",
        "thank",
        "thanks",
        "total",
        "tracking",
        "account",
        "admin",
        "billing",
        "care",
        "contact",
        "customer",
        "email",
        "gmail",
        "help",
        "hotmail",
        "icloud",
        "info",
        "mail",
        "member",
        "news",
        "newsletter",
        "noreply",
        "notification",
        "notifications",
        "order",
        "orders",
        "outlook",
        "receipt",
        "receipts",
        "reply",
        "sales",
        "service",
        "shop",
        "support",
        "team",
        "update",
        "updates",
        "user",
        "yahoo",
    }
)

MIN_DERIVED_NAME_LEN = 4  # tokens mined from an email local part
MIN_EXPLICIT_NAME_LEN = 3  # tokens from a display name / "Ship to" block


@dataclass(frozen=True)
class PIIPolicy:
    """Which PII categories to mask, and how aggressively."""

    mask_emails: bool = True
    mask_phones: bool = True
    mask_addresses: bool = True
    mask_card_numbers: bool = True
    mask_names: bool = True
    #: Keep the domain of sender headers (``help@vendor.example`` -> ``[EMAIL_1]@vendor.example``)
    #: so vendor identity remains extractable.
    preserve_sender_domain: bool = True
    #: Drop routing/identity headers outright instead of redacting them.
    drop_routing_headers: bool = True
    #: Additional names to mask (e.g. household members not named in headers).
    extra_names: tuple[str, ...] = ()


@dataclass
class RedactionReport:
    """Counts of what was masked. Deliberately holds no original values."""

    by_category: dict[str, int] = field(default_factory=dict)
    placeholders: dict[str, str] = field(default_factory=dict)
    dropped_headers: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.by_category.values())

    def to_dict(self) -> dict:
        return {
            "total_unique_values": self.total,
            "by_category": dict(self.by_category),
            "placeholders": dict(self.placeholders),
            "dropped_headers": list(self.dropped_headers),
        }


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class PIIRedactor:
    """Stateful, single-document redactor shared across tier 1 and tier 2.

    Sharing one instance across a document keeps placeholder numbering
    consistent between headers and body (the recipient address in ``To:`` gets
    the same token as the same address appearing in the invoice footer).
    """

    def __init__(self, policy: PIIPolicy | None = None):
        self.policy = policy or PIIPolicy()
        self._tokens: dict[tuple[str, str], str] = {}
        self._counts: dict[str, int] = {}
        self._dropped_headers: list[str] = []
        # lowercase name token -> placeholder (all tokens of one identity share it)
        self._name_tokens: dict[str, str] = {}
        for name in self.policy.extra_names:
            self.learn_name(name)

    # --- placeholders -------------------------------------------------------

    def _placeholder(self, category: str, value: str) -> str:
        key = (category, " ".join(value.split()).lower())
        token = self._tokens.get(key)
        if token is None:
            self._counts[category] = self._counts.get(category, 0) + 1
            token = f"[{category}_{self._counts[category]}]"
            self._tokens[key] = token
        return token

    @property
    def report(self) -> RedactionReport:
        by_category: dict[str, int] = {}
        placeholders: dict[str, str] = {}
        for (category, _value), token in self._tokens.items():
            by_category[category] = by_category.get(category, 0) + 1
            placeholders[token] = category
        return RedactionReport(
            by_category=by_category,
            placeholders=placeholders,
            dropped_headers=list(self._dropped_headers),
        )

    # --- name discovery -----------------------------------------------------

    def learn_name(self, name: str | None, *, min_len: int = MIN_EXPLICIT_NAME_LEN) -> None:
        """Register a personal name so all of its tokens get masked in text."""
        if not name or not self.policy.mask_names:
            return
        tokens = [
            t.lower()
            for t in re.split(r"[^A-Za-z]+", name)
            if len(t) >= min_len and t.lower() not in NAME_STOPWORDS
        ]
        if not tokens:
            return
        placeholder = self._placeholder(PERSON, " ".join(tokens))
        for token in tokens:
            self._name_tokens.setdefault(token, placeholder)

    def learn_recipient(self, header_value: str | None) -> None:
        """Mine names from a recipient header: display name and email local part."""
        if not header_value or not self.policy.mask_names:
            return
        for match in EMAIL_PARTS_RE.finditer(header_value):
            self.learn_name(match.group(1), min_len=MIN_DERIVED_NAME_LEN)
        display = EMAIL_RE.sub(" ", header_value)
        display = re.sub(r"[<>\"']", " ", display)
        self.learn_name(display)

    def learn_names_from_text(self, text: str | None) -> None:
        """Mine names from body structure: greetings and ship-to/bill-to blocks."""
        if not text or not self.policy.mask_names:
            return
        for pattern in (SHIP_TO_RE, GREETING_RE, THANKS_RE):
            for match in pattern.finditer(text):
                self.learn_name(match.group(1))

    # --- tier 1: headers ----------------------------------------------------

    def redact_recipient(self, value: str | None) -> str | None:
        """Mask a recipient header in full."""
        if value is None:
            return None
        if not self.policy.mask_emails:
            return self.redact_text(value)
        value = EMAIL_RE.sub(lambda m: self._placeholder(EMAIL, m.group(0)), value)
        return self.redact_text(value)

    def redact_sender(self, value: str | None) -> str | None:
        """Mask a sender header, optionally keeping the (vendor) domain."""
        if value is None:
            return None
        if self.policy.mask_emails:
            if self.policy.preserve_sender_domain:
                value = EMAIL_PARTS_RE.sub(
                    lambda m: f"{self._placeholder(EMAIL, m.group(0))}@{m.group(2)}", value
                )
            else:
                value = EMAIL_RE.sub(lambda m: self._placeholder(EMAIL, m.group(0)), value)
        return self.redact_text(value)

    def redact_headers(self, headers: Iterable[tuple[str, str]]) -> dict[str, str]:
        """Redact a header list, dropping routing/identity headers."""
        out: dict[str, str] = {}
        for name, value in headers:
            lname = name.lower()
            if self.policy.drop_routing_headers and is_routing_header(lname):
                if lname not in self._dropped_headers:
                    self._dropped_headers.append(lname)
                continue
            if lname in RECIPIENT_HEADERS:
                out[name] = self.redact_recipient(value) or ""
            elif lname in SENDER_HEADERS:
                out[name] = self.redact_sender(value) or ""
            else:
                out[name] = self.redact_text(value)
        return out

    # --- tier 2: content ----------------------------------------------------

    def redact_text(self, text: str | None) -> str:
        """Mask PII inside document content (or any free text)."""
        if not text:
            return text or ""

        # Defense in depth: pick up name cues in this text too, so a caller that
        # forgets learn_names_from_text() still gets names masked. Learning is
        # idempotent, so the parser's up-front pass is unaffected.
        self.learn_names_from_text(text)

        if self.policy.mask_emails:
            text = EMAIL_RE.sub(lambda m: self._placeholder(EMAIL, m.group(0)), text)

        if self.policy.mask_card_numbers:
            text = CARD_MASKED_RE.sub(lambda m: self._placeholder(CARD, m.group(0)), text)
            text = CARD_SHORT_MASK_RE.sub(lambda m: self._placeholder(CARD, m.group(0)), text)
            text = CARD_ENDING_RE.sub(
                lambda m: m.group(1) + self._placeholder(CARD, m.group(2)), text
            )
            text = CARD_PAN_RE.sub(self._sub_pan, text)

        if self.policy.mask_phones:
            text = TEL_URI_RE.sub(lambda m: "tel:" + self._placeholder(PHONE, m.group(0)), text)
            text = PHONE_RE.sub(lambda m: self._placeholder(PHONE, m.group(0)), text)

        if self.policy.mask_addresses:
            text = PO_BOX_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = STREET_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = CITY_STATE_ZIP_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = CITY_ZIP_STATE_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = ZIP4_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = self._mask_interior_address_lines(text)

        if self.policy.mask_names and self._name_tokens:
            text = self._mask_names(text)

        return text

    def _sub_pan(self, match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) < 13 or not _luhn_valid(digits):
            return match.group(0)
        return self._placeholder(CARD, match.group(0))

    def _mask_interior_address_lines(self, text: str) -> str:
        """Mask short lines wedged between two masked address lines.

        Multi-line blocks like ``500 Oak Ave / 914 / Madison, WI 53703`` leave the
        bare apartment number behind after the street and city lines are masked.
        """
        lines = text.split("\n")
        for i in range(1, len(lines) - 1):
            stripped = lines[i].strip()
            if not stripped or not UNIT_LINE_RE.match(stripped):
                continue
            prev_token = ADDRESS_TOKEN_RE.search(lines[i - 1])
            if prev_token and ADDRESS_TOKEN_RE.search(lines[i + 1]):
                lines[i] = lines[i].replace(stripped, prev_token.group(0))
        return "\n".join(lines)

    def _mask_names(self, text: str) -> str:
        pattern = re.compile(
            r"(?<![\w@.])(" + "|".join(sorted(map(re.escape, self._name_tokens), key=len, reverse=True)) + r")(?![\w@])",
            re.IGNORECASE,
        )
        text = pattern.sub(lambda m: self._name_tokens[m.group(1).lower()], text)
        # "Jane Roe" -> "[PERSON_1] [PERSON_1]" -> "[PERSON_1]"
        return re.sub(r"(\[PERSON_\d+\])(?:[ \t]+\1)+", r"\1", text)
