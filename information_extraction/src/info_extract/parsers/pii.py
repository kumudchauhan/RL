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

import os
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
# A single line break is tolerated around the `@` because PDF text extraction
# wraps mid-address (`orders@\nvendor.example`).
_WRAP = r"[ \t]*(?:\n[ \t]*)?"
EMAIL_RE = re.compile(
    rf"[A-Za-z0-9._%+\-]+{_WRAP}(?:@|%40|\[at\]){_WRAP}[A-Za-z0-9.\-]+\.[A-Za-z]{{2,}}",
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
# E.164 with no separators, common on invoices: `+15105550142`
INTL_PHONE_RE = re.compile(r"(?<![\w+])\+\d{10,15}(?!\d)")

# `**** **** **** 4321`, `xxxx-1234`, `•••• 1234`. `#` is deliberately not a mask
# char here: `#1234` is a store/order number and `&#8199;` is an HTML entity.
CARD_MASKED_RE = re.compile(r"(?:[*x•·]{2,}[\s\-–]?){1,4}\d{2,4}(?!\d)", re.IGNORECASE)
# A single mask char, as in `American Express *5678` — require a full 4 digits so
# footnote markers (`*2 items`) are left alone.
CARD_SHORT_MASK_RE = re.compile(r"(?<![\w*])[*x•·][\s\-]?\d{4}(?!\d)", re.IGNORECASE)
# `ending in 5678`, `ending with 5678` — keeps the phrasing, drops the digits.
CARD_ENDING_RE = re.compile(
    r"(ending\s+(?:in|with)\s+|ending\s+)(\d{4})(?!\d)", re.IGNORECASE
)
# Full PAN, validated with Luhn to avoid eating order numbers.
CARD_PAN_RE = re.compile(r"(?<![\d\-])(?:\d{4}[\s\-]?){3}\d{1,4}(?![\d\-])")
#: Labels for device/product identifiers that also pass Luhn — see _sub_pan.
SERIAL_LABEL_RE = re.compile(
    r"(?i)\b(?:imei|serial(?:\s*(?:no|number|#))?|s\.?\s?no|sn|meid|vin)\b[:# ]*$"
)

STREET_SUFFIXES = (
    "St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ct|Court|Dr|Drive|Ln|Lane|Way|"
    "Pl|Place|Ter|Terrace|Cir|Circle|Pkwy|Parkway|Hwy|Highway|Sq|Square|Trl|Trail|"
    "Loop|Row|Alley|Path|Walk|Crescent|Close|Mews"
)
UNIT_KEYWORDS = "#|Apt\\.?|Apartment|Suite|Ste\\.?|Unit|Fl\\.?|Floor|Rm\\.?|Room"
# Single-line by design (`[ \t]` not `\s`): a street line never wraps, and letting
# it span line breaks made it swallow whole PDF blocks — "December 23, 2025 /
# DeepLearning.AI / 100 North Arlington Avenue" collapsed into one placeholder.
STREET_RE = re.compile(
    rf"(?<![\w])\d{{1,6}}[A-Za-z]?[ \t]+(?:[\w.'\-]+[ \t]+){{0,4}}(?:{STREET_SUFFIXES})\b\.?"
    rf"(?:[ \t]*,?[ \t]*(?:{UNIT_KEYWORDS})[ \t]*[\w\-]+)?",
    re.IGNORECASE,
)
# At least three letters per word: a two-letter "city" is really a code, and with
# `\s` gaps below that let "NA / NA / 12345.67" read as city/state/ZIP and swallow
# the invoice total.
_CITY = r"[A-Z][A-Za-z.'\-]{2,}(?:[ ][A-Z][A-Za-z.'\-]+){0,3}"
STATE_NAMES = (
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|"
    "Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|"
    "Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|"
    "New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|"
    "Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|"
    "Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming"
)
#: Two-letter code or spelled-out name: "Madison, WI" / "Springfield, Illinois".
_STATE = rf"(?:[A-Z]{{2}}|{STATE_NAMES})"
#: One gap between address fields — spaces, or a single line break where the
#: block wraps. Plain `\s+` let a match reach across unrelated lines.
_GAP = r"(?:[ \t]+|[ \t]*\n[ \t]*)"
# "Madison, WI 53703" / "Springfield, IL, 62704" / "Springfield, Illinois 62704"
CITY_STATE_ZIP_RE = re.compile(
    rf"(?<![\w]){_CITY},?{_GAP}{_STATE}\b,?{_GAP}\d{{5}}(?:-\d{{4}})?(?![\d])"
)
# "Springfield 62704-1234 IL" — some vendors reorder city/zip/state
CITY_ZIP_STATE_RE = re.compile(
    rf"(?<![\w]){_CITY},?{_GAP}\d{{5}}(?:-\d{{4}})?{_GAP}{_STATE}\b(?![\w])"
)
# A whole line of "53703 Madison" — ZIP-first ordering. Anchored to the full line so
# a 5-digit order number followed by a capitalized word is not swept up.
ZIP_CITY_LINE_RE = re.compile(rf"^[ \t]*\d{{5}}(?:-\d{{4}})?[ \t]+{_CITY}[ \t]*$", re.MULTILINE)
# ZIP+4 is specific enough to mask on its own.
ZIP4_RE = re.compile(r"(?<![\d\-])\d{5}-\d{4}(?![\d\-])")
#: A line holding nothing but a postal code. Masked only when adjacent to an
#: already-masked line or a country line — see _mask_bare_zip_lines.
BARE_ZIP_LINE_RE = re.compile(r"^[ \t]*\d{5}(?:-\d{4})?[ \t]*$")
COUNTRY_LINE_RE = re.compile(
    r"^[ \t]*(?:United States(?: of America)?|U\.?S\.?A\.?|US|Canada|Mexico|"
    r"United Kingdom|UK|India|Australia|Germany|France)[ \t]*,?[ \t]*$",
    re.IGNORECASE,
)
PO_BOX_RE = re.compile(r"P\.?\s?O\.?\s*Box\s*\d+", re.IGNORECASE)
UNIT_LINE_RE = re.compile(rf"^(?:(?:{UNIT_KEYWORDS})[ \t]*)?[\w\-#/]{{1,8}}$", re.IGNORECASE)
#: A line that is explicitly a unit designator, e.g. "Apt 5", "Suite 200E".
UNIT_PREFIXED_LINE_RE = re.compile(rf"^(?:{UNIT_KEYWORDS})[ \t]*[\w\-#/]{{1,10}}$", re.IGNORECASE)
#: A line holding only a state, e.g. "MA" / "California". PDF column layouts split
#: an address across one line per field.
STATE_LINE_RE = re.compile(rf"^(?:[A-Z]{{2}}|{STATE_NAMES})[ \t]*,?$")
#: A line holding only a city name, e.g. "Springfield" / "SPRINGFIELD".
CITY_LINE_RE = re.compile(rf"^(?:{_CITY}|[A-Z][A-Z.'\-]+(?:[ ][A-Z][A-Z.'\-]+){{0,3}})[ \t]*,?$")
ADDRESS_TOKEN_RE = re.compile(r"\[ADDRESS_\d+\]")
#: How many lines after a masked address line may still be treated as part of it.
MAX_ADDRESS_CONTINUATION_LINES = 4
#: How many stacked field labels to walk past when looking for a recipient label
#: above a name-shaped line.
MAX_NAME_LABEL_LOOKBACK = 4

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
    r"(?i:ship(?:ping|ped)?\s+(?:to|address)|bill(?:ing|ed)?\s+(?:to|address)|"
    r"deliver(?:y|ed)?\s+(?:to|address)|mailing\s+address|sold\s+to|attn\.?|"
    r"invoice\s+for|recipient|(?:customer\s+)?name)\s*:?\s*\n?[ \t]*"
    # The name itself must sit on one line: allowing \s here let the capture run
    # past the newline and learn the next field's label ("Jane Roe / Status")
    # as part of the name, which then masked that word document-wide.
    r"([A-Z][a-z'\-]+(?:[ \t]+[A-Z][a-z'\-]+){1,2})\b"
)

# --- Column-layout name discovery (mostly PDFs) ------------------------------
# PDF text extraction emits one line per visual field, so a "Bill to" label and
# its value can be several lines apart and the name is often ALL CAPS
# ("Customer Information / JANE ROE"). SHIP_TO_RE, which wants the value
# adjacent and Title Case, misses both shapes.

#: A line that is nothing but a person-shaped name, Title Case or ALL CAPS, with
#: an optional leading account/receipt id ("1000042 - JANE ROE").
NAME_LINE_RE = re.compile(
    r"^(?:[\d\-#]{2,12}[ \t]*[-–][ \t]*)?"
    r"((?:[A-Z][a-z'\-]+|[A-Z][A-Z'\-]+)(?:[ \t]+(?:[A-Z][a-z'\-]+|[A-Z][A-Z'\-]+)){1,2})"
    r"[ \t]*,?$"
)
#: Labels whose value is the document's recipient — i.e. a person, not the vendor.
RECIPIENT_LABEL_LINE_RE = re.compile(
    r"^[ \t]*(?:ship(?:ping|ped)?[ \t]+(?:to|address)|bill(?:ing|ed)?[ \t]+(?:to|address)|"
    r"deliver(?:y|ed)?[ \t]+(?:to|address)|sold[ \t]+to|mailing[ \t]+address|"
    r"customer(?:[ \t]+(?:information|details|name))?|client|patient|member|recipient|"
    r"attn\.?|invoice[ \t]+for|name)[ \t]*:?[ \t]*$",
    re.IGNORECASE,
)
#: Organisations get caught by NAME_LINE_RE too; these suffixes rule them out so a
#: vendor name under "Bill to" is not learned as a person and masked document-wide.
COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:Inc|Inc\.|LLC|L\.L\.C\.|Ltd|Ltd\.|Limited|Corp|Corp\.|Corporation|Co|Co\.|"
    r"Company|GmbH|PLC|LLP|PBC|Foundation|Trust|Association|University|College|School|"
    r"Hospital|Center|Centre|Bank|Group|Holdings|Partners|Store|Market|Services)\b",
    re.IGNORECASE,
)
#: A label line looks like a name line ("Issue Date", "Date Of Service"); learning
#: one would mask that word everywhere. Any of these words disqualifies the line.
NAME_LINE_REJECT_RE = re.compile(
    r"(?i)\b(?:date|invoice|receipt|order|number|no|status|total|amount|due|paid|price|"
    r"qty|quantity|item|items|description|provider|vendor|seller|merchant|method|"
    r"payment|card|tax|subtotal|discount|shipping|delivery|address|phone|email|page|"
    r"time|balance|summary|detail|details|terms|notes|service|services|ref|reference)\b"
)

#: Words that are roles, providers, or document labels — never masked as names.
#: Guards the name heuristics: a mislearned common word would be masked
#: everywhere in the body and destroy line-item text.
NAME_STOPWORDS = frozenset(
    {
        "address",
        "anonymous",
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
        "friend",
        "guest",
        "madam",
        "madame",
        "sir",
        "there",
        "valued",
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
        "none",
        "noreply",
        "null",
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
        "unknown",
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

    @classmethod
    def from_env(cls, **overrides) -> PIIPolicy:
        """Build a policy, seeding ``extra_names`` from ``INFO_EXTRACT_PII_NAMES``.

        The name heuristics rely on document structure, and some layouts give
        them nothing to work with — a PDF whose "Client" label sits three lines
        above its value in a different text block, for instance. Naming the
        document owner explicitly (``INFO_EXTRACT_PII_NAMES="Jane Q Doe,J Doe"``)
        is the reliable backstop for those.
        """
        raw = os.environ.get("INFO_EXTRACT_PII_NAMES", "")
        names = tuple(n.strip() for n in raw.split(",") if n.strip())
        if names:
            overrides["extra_names"] = tuple(overrides.get("extra_names", ())) + names
        return cls(**overrides)


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
                if self._is_plausible_name(match.group(1)):
                    self.learn_name(match.group(1))
        self._learn_names_from_layout(text)

    @staticmethod
    def _is_plausible_name(candidate: str) -> bool:
        """Reject captures that are really labels or organisations.

        A label captured as a name is the worst failure mode here: the word then
        gets masked everywhere in the document. "Customer Name / Doc No" in a
        column layout put "Doc No" where the value should be, which masked every
        ``DOC/...`` order reference in the invoice.
        """
        return not (
            NAME_LINE_REJECT_RE.search(candidate) or COMPANY_SUFFIX_RE.search(candidate)
        )

    def _learn_names_from_layout(self, text: str) -> None:
        """Mine names from one-field-per-line layouts (PDF column extraction).

        A name-shaped line counts as a person when either

        * the next non-empty line is a street address — the strongest signal
          available, and label-independent; or
        * a recipient label ("Bill to", "Customer Information", "Client") sits
          within a few lines above it, with only other labels in between.
        """
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            match = NAME_LINE_RE.match(stripped)
            if not match:
                continue
            candidate = match.group(1)
            if not self._is_plausible_name(candidate):
                continue
            if self._street_follows(lines, i) or self._recipient_label_precedes(lines, i):
                self.learn_name(candidate)

    @staticmethod
    def _street_follows(lines: list[str], i: int) -> bool:
        for nxt in lines[i + 1 : i + 3]:
            stripped = nxt.strip()
            if not stripped:
                continue
            return bool(STREET_RE.match(stripped) or PO_BOX_RE.match(stripped))
        return False

    @staticmethod
    def _recipient_label_precedes(lines: list[str], i: int) -> bool:
        seen = 0
        for prev in reversed(lines[:i]):
            stripped = prev.strip()
            if not stripped:
                continue
            if RECIPIENT_LABEL_LINE_RE.match(stripped):
                return True
            # Only walk past other field labels — a column layout stacks them.
            if seen >= MAX_NAME_LABEL_LOOKBACK or not NAME_LINE_REJECT_RE.search(stripped):
                return False
            seen += 1
        return False

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
            text = INTL_PHONE_RE.sub(lambda m: self._placeholder(PHONE, m.group(0)), text)
            text = PHONE_RE.sub(lambda m: self._placeholder(PHONE, m.group(0)), text)

        if self.policy.mask_addresses:
            text = PO_BOX_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = STREET_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = CITY_STATE_ZIP_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = CITY_ZIP_STATE_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = ZIP_CITY_LINE_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = ZIP4_RE.sub(lambda m: self._placeholder(ADDRESS, m.group(0)), text)
            text = self._mask_bare_zip_lines(text)
            text = self._mask_interior_address_lines(text)

        if self.policy.mask_names and self._name_tokens:
            text = self._mask_names(text)

        return text

    def _sub_pan(self, match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) < 13 or not _luhn_valid(digits):
            return match.group(0)
        # IMEIs and some serial numbers are Luhn-valid too, and they identify the
        # product, not the payer — the label right before them says which it is.
        if SERIAL_LABEL_RE.search(match.string[max(0, match.start() - 24) : match.start()]):
            return match.group(0)
        return self._placeholder(CARD, match.group(0))

    def _mask_bare_zip_lines(self, text: str) -> str:
        """Mask a lone postal-code line that sits next to a country line.

        Some invoices print only the postal code and country of the recipient
        ("Bill to / <name> / 62704 / United States"), so there is no street or
        city for the address patterns to anchor to. A line that is nothing but a
        postal code, directly beside a country line, is not a total or an order
        number — masking it is safe.
        """
        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not BARE_ZIP_LINE_RE.match(stripped):
                continue
            neighbours = lines[i + 1 : i + 2] + lines[max(0, i - 1) : i]
            if any(COUNTRY_LINE_RE.match(n.strip()) for n in neighbours):
                lines[i] = line.replace(stripped, self._placeholder(ADDRESS, stripped))
        return "\n".join(lines)

    def _mask_interior_address_lines(self, text: str) -> str:
        """Mask address lines left stranded around a masked multi-line block.

        Street and city lines are masked independently, so blocks like
        ``500 Oak Ave / 914 / Madison, WI 53703`` or ``123 Main St. / Apt 5``
        leave the apartment line behind, and PDF column layouts split an address
        into one line per field (``123 Main St / Springfield / IL / 62704``)
        where only the street line matches on its own.

        Masking therefore continues line by line after a masked address for as
        long as each line still looks like part of an address, capped by
        :data:`MAX_ADDRESS_CONTINUATION_LINES` so a stray match cannot run away.
        """
        lines = text.split("\n")
        token: str | None = None
        run = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            found = ADDRESS_TOKEN_RE.search(line)
            if found:
                # A masked line restarts the run; the token carries forward so the
                # whole block collapses to one placeholder.
                token, run = found.group(0), 0
                continue
            if token is None or run >= MAX_ADDRESS_CONTINUATION_LINES:
                token = None
                continue
            if not self._is_address_continuation(lines, i, stripped):
                token = None
                continue
            lines[i] = line.replace(stripped, token)
            run += 1
        return "\n".join(lines)

    @staticmethod
    def _is_address_continuation(lines: list[str], i: int, stripped: str) -> bool:
        """Whether line ``i`` is a leftover piece of the address block above it."""
        if UNIT_PREFIXED_LINE_RE.match(stripped):  # "Apt 5"
            return True
        if BARE_ZIP_LINE_RE.match(stripped):  # "62704"
            return True
        if STATE_LINE_RE.match(stripped):  # "MA" / "California"
            return True
        # A bare token ("914") or a lone city ("Springfield") is too generic to
        # mask on the preceding line alone — require the block to continue below.
        if UNIT_LINE_RE.match(stripped) or CITY_LINE_RE.match(stripped):
            for nxt in lines[i + 1 : i + 2]:
                after = nxt.strip()
                return bool(
                    ADDRESS_TOKEN_RE.search(nxt)
                    or BARE_ZIP_LINE_RE.match(after)
                    or STATE_LINE_RE.match(after)
                    or COUNTRY_LINE_RE.match(after)
                )
        return False

    def _mask_names(self, text: str) -> str:
        pattern = re.compile(
            r"(?<![\w@.])(" + "|".join(sorted(map(re.escape, self._name_tokens), key=len, reverse=True)) + r")(?![\w@])",
            re.IGNORECASE,
        )
        text = pattern.sub(lambda m: self._name_tokens[m.group(1).lower()], text)
        # "Jane Roe" -> "[PERSON_1] [PERSON_1]" -> "[PERSON_1]"
        return re.sub(r"(\[PERSON_\d+\])(?:[ \t]+\1)+", r"\1", text)
