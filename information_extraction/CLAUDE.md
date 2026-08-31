# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An information extraction pipeline that parses invoice emails and order confirmations, extracts structured fields using Claude's tool-use API, and evaluates/optimizes extraction quality using a Harbor-style RLVR (Reinforcement Learning with Verifiable Rewards) framework. Composable verifiers produce granular reward signals suitable for training.

## Architecture

```
src/info_extract/
├── schemas.py              # Pydantic models: InvoiceExtraction, LineItem, ProductIdentifier,
│                           #   Discount, Fee, Payment — PII-free by construction
├── parsers/
│   ├── base.py             # ParsedDocument dataclass + DocumentParser ABC
│   ├── eml_parser.py       # .eml → ParsedDocument (stdlib email + BeautifulSoup)
│   ├── pdf_parser.py       # .pdf → ParsedDocument (pymupdf text layer)
│   └── pii.py              # Two-tier PII redaction (PIIPolicy + PIIRedactor)
├── agent/
│   ├── prompts.py          # System/user prompt templates
│   └── extraction_agent.py # Claude API wrapper using forced tool-use for structured output
├── verifiers/
│   ├── base.py             # Verifier ABC + VerificationResult
│   ├── field_verifier.py   # Scalar fields (vendor, service_provider, order_id, dates) — fuzzy
│   ├── numeric_verifier.py # Monetary fields (totals, fees, billed vs paid) — tolerance-based
│   ├── line_item_verifier.py # Line items — greedy bipartite matching with F1
│   ├── detail_verifier.py  # Repeated records — identifiers, taxonomy, coupons, fees, tenders
│   └── composite.py        # Weighted aggregation → RewardSignal
├── dataset/
│   ├── loader.py           # Reads annotations/, prunes dropped fields, parses source docs
│   └── tasks.py            # ExtractionTask dataclass
└── runner/
    └── evaluate.py         # Main harness + CLI entrypoint (extract-eval)
```

## Key Design Decisions

- **Claude tool-use** for extraction guarantees valid JSON matching the Pydantic schema
- **Granular, composable verifiers** produce per-field reward signals (not binary pass/fail)
- **Weighted composite reward**: field_accuracy=0.20, numeric_accuracy=0.30, line_item_f1=0.35,
  detail_accuracy=0.15
- **Applicability + renormalization**: a verifier sets `VerificationResult.applicable=False` when
  the ground truth states nothing it can check (no line items, no annotated UPCs). Inapplicable
  components are dropped from the composite and the remaining weights renormalized, so
  annotating detail incrementally never looks like a regression — and a blank annotation never
  scores 1.0. `RewardSignal.applied_components` records which components counted.
- **Partial credit scoring**: fuzzy string matching for vendor names, tolerance-based numeric comparison, weighted F1 for line items
- **Transcribe, never infer**: every schema field is optional, so anything the document does not
  print comes back `null`/`[]`. `currency` does *not* default to `"USD"`; a category is never
  derived from a product name; a missing quantity is never assumed to be 1.
- **Billed, paid, and due are separate fields**, never derived from one another: `total` is what
  the order was billed, `amount_paid` what changed hands, `amount_due` what is outstanding. On a
  split tender, an instalment/EMI plan, or a partial gift-card payment these genuinely differ,
  and a model that copies one into another is wrong in a way a single `total` field would hide.
- **Store vs platform**: `vendor` is the retailer the goods came from; `service_provider` is the
  platform that took or fulfilled the order (Instacart, DoorDash), and is null when the store
  billed directly. Both are scored, so filing the platform as the store costs two fields.
- **Fees stay separate**: `shipping_cost`, `delivery_fee`, `service_fee`, and `tip` each have a
  field; anything else printed goes to `fees[]` with its label verbatim rather than being summed
  into one of them.
- **Report-don't-punish for unannotated detail**: `DetailVerifier` scores only values the ground
  truth states, and lists detail the model extracted beyond the annotation as
  `unverifiable_extras`.
- **Rollout capture mode**: stores full prompt/response for RLVR training data generation
- **PII masked at the parser**, before any content reaches the API or a rollout file
- **PII-free output schema** (second layer, see below): no field exists that could hold a name,
  address, phone, email, or card number — not even masked or synthetic

## PII Redaction

`parsers/pii.py` masks PII in two tiers against one shared `PIIRedactor` per document, so a
value seen in the envelope and again in the body maps to the same placeholder:

1. **Envelope (tier 1)** — for `.eml`: recipient addresses masked in full (`[EMAIL_1]`); sender
   addresses keep the vendor domain (`[EMAIL_2]@vendor.example`) so vendor extraction still works;
   routing and identity headers (`Received*`, `*-SPF`, `DKIM-*`, `Message-ID`,
   `List-Unsubscribe`, ...) are dropped outright. For `.pdf`: the document-info dictionary
   stands in for the envelope — `author`/`title`/`subject` are redacted and
   `creator`/`producer` (browser and OS build that printed the receipt) are dropped as the
   analogue of routing headers, listed in `metadata["dropped_info_keys"]`.
2. **Content (tier 2)** — postal addresses, emails, phone numbers, card numbers (`**** 4321`,
   `ending in 5678`, Luhn-valid PANs), and personal names in `text_body` and `html_body`.

Both parsers also expose a redacted `metadata["source_name"]`, and that — not `source_path` —
is the filename shown to the model, because filenames carry names and order ids
(`invoice_2023-04-12-Roe.pdf`).

Details:

- Names are discovered per document from recipient headers (display name + email local part),
  PDF `author` info, body cues (`Hi <name>`, `Ship to:`, `NAME:`), and line layout, then masked
  wherever they appear — including the subject line. `NAME_STOPWORDS` and
  `NAME_LINE_REJECT_RE`/`COMPANY_SUFFIX_RE` block role words, document labels, and
  organisations from being learned as names, since a mislearned common word would be masked
  across the body.
- **PDF layouts need line-level rules.** Text extraction emits one line per visual field, so a
  `Bill to` label can sit several lines from its value, names are often ALL CAPS
  (`JANE ROE`), and an address arrives as `123 Main St / Springfield / IL / 62704`
  where only the street line matches on its own. Redaction therefore also treats a name-shaped
  line as a person when a street line follows it or a recipient label precedes it, and
  continues masking after a masked address line while each following line still looks like part
  of an address (unit, city, state, postal code), capped at
  `MAX_ADDRESS_CONTINUATION_LINES`. Address patterns allow at most one line break per field gap
  — plain `\s+` let a match reach across unrelated lines and swallow an invoice total.
- Placeholders are stable per value (`[EMAIL_1]`, `[ADDRESS_2]`, ...), which preserves document
  structure and coreference for the model.
- `ParsedDocument.redaction_report` carries counts and placeholder names only — never the
  original values.
- Configure via `PIIPolicy` (per-category toggles, `preserve_sender_domain`, `extra_names`):
  `EmlParser(pii_policy=PIIPolicy(mask_addresses=False))`. Disable entirely with
  `EmlParser(redact_pii=False)` or `extract-eval --no-pii-redaction` (debugging only).
- Both parsers default to `PIIPolicy.from_env()`, which seeds `extra_names` from
  `INFO_EXTRACT_PII_NAMES` (comma-separated). The structural heuristics cannot cover every
  layout — some PDFs put a label and its value in unrelated text blocks — so naming the
  document owner explicitly is the reliable backstop.
- **Second layer — the schema itself.** Redaction is not the only guard. `schemas.py` has no
  `shipping_address`, `billing_address`, `payment.last_four`, or free-text `notes` field, so
  there is nowhere for personal data to land in a result file or a rollout even in masked form:
  - `assert_pii_free_schema()` runs at import and fails if any field name contains a fragment in
    `FORBIDDEN_FIELD_FRAGMENTS` (`address`, `phone`, `email`, `last_four`, `customer`, ...).
  - `model_config = ConfigDict(extra="forbid")` on `ExtractionModel` rejects unknown keys, so a
    removed field cannot be reintroduced through a response or an annotation.
  - A `model_validator(mode="after")` strips surviving redaction placeholders
    (`[PERSON_1]`, `[ADDRESS_2]`, ...) from string values — `None` for optional fields, `""` for
    required ones.
  - `annotations/` still holds the real values locally. `dataset/loader.py` prunes them
    (`DROPPED_ANNOTATION_KEYS`, `DROPPED_PAYMENT_KEYS`, `prune_annotation`) and logs only the
    field *names* it dropped; `unexpected_keys()` makes an unrecognised key a loud `ValueError`
    rather than a silent pass-through.
- Order numbers, totals, dates, and product names are deliberately preserved: phone patterns
  require digit-group separators, PAN masking requires a Luhn check and skips `IMEI:`/`Serial:`
  labelled values (IMEIs are Luhn-valid too), and `#1234`-style store numbers and `&#8199;`
  HTML entities are excluded from card patterns.

## Commands

```bash
# Install dependencies
uv sync

# Install with dev tools (pytest, ruff)
uv sync --extra dev

# Install PDF support (pymupdf) — required to parse .pdf invoices
uv sync --extra pdf

# Run extraction evaluation (requires ANTHROPIC_API_KEY)
uv run extract-eval

# Run with rollout capture for RLVR training
uv run extract-eval --capture-rollouts

# Run without PII masking (debugging only — sends raw text to the API)
uv run extract-eval --no-pii-redaction

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/
```

## Data (gitignored)

- `invoices/` — personal invoice/receipt documents (.eml, .pdf). Not tracked in git.
- `annotations/` — ground truth JSON files matching the InvoiceExtraction schema. Not tracked in git.
- `results/` — evaluation output JSON files. Not tracked in git.

### Annotation Format

Each annotation file in `annotations/` is a JSON file with a `_meta` key (source_file, annotator, date) plus all fields from the `InvoiceExtraction` schema. See `annotations/_template.json` for the template.

Existing annotations may still carry `shipping_address`, `billing_address`, `notes`, and
`payment.last_four` from the v1 schema — the loader prunes those on load, so annotation files
need no editing. A v2 singular `payment` object is lifted into `payments[0]` by
`migrate_annotation`, which prints what it renamed; annotations are never edited to follow a
schema change. Any *other* unrecognised key is a `ValueError`, which is deliberate: a typo in a
field name would otherwise silently score as "not annotated". Line-item detail
(`upc`/`asin`/`department`/`discounts`/...) is optional per annotation; whatever is absent simply
is not scored.

## Extraction Schema (fields extracted)

- `vendor` (the store/retailer), `service_provider` (Instacart/DoorDash-style platform, null if
  the store billed directly), `order_id`, `order_date`, `delivery_date`, `currency`
- money — `subtotal`, `tax`, `discount`, `total` (billed for the order), `amount_paid` (actually
  paid, only if printed), `amount_due` (outstanding balance, only if printed)
- fees — `shipping_cost`, `delivery_fee`, `service_fee`, `tip`, plus `fees[]`
  (`Fee(label, amount)`) for any other printed charge, label verbatim
- `discounts[]` (order level) — `Discount(description, code, source, amount, percentage)`, where
  `source` is the issuer as printed (`Groupon`, `manufacturer coupon`, ...)
- `payments[]` — one `Payment` per tender line: `method` (`PaymentMethod` enum: credit_card,
  debit_card, gift_card, store_credit, apple_pay, google_pay, paypal, venmo, cash, check, ebt,
  bank_transfer, buy_now_pay_later, other), `card_type` (brand as printed), `amount` charged to
  that instrument, and `installment_count`/`installment_amount` when the document states an
  instalment or EMI plan. **No card number, no `last_four`.**
- `line_items[]`
  - `product_name`, `quantity` (float — weighed goods print `1.24 lb`), `quantity_unit`,
    `unit_price`, `total_price`
  - every identifier the document prints: `sku`, `upc`, `asin`, `product_number`, plus
    `other_identifiers[]` (`ProductIdentifier(label, value)`) for anything else (`PLU`, `ISBN`,
    `Style #`) with the label kept verbatim
  - taxonomy **only as printed**: `department` (section header, e.g. `PRODUCE`) and `category`
    (e.g. `Beverages`, `Makeup`) — never inferred from the product name
  - `discounts[]` — per-item coupons and markdowns

**No address, name, phone, email, or card-number field exists** — see PII Redaction above.

## Configuration

- `harbor.toml` — Harbor project/dataset/agent/verifier config
- `pyproject.toml` — uv project config, dependencies, CLI entrypoint

## Dependencies

- `anthropic` — Claude API client
- `pydantic` — schema validation and JSON schema generation
- `beautifulsoup4` + `lxml` — HTML parsing for email bodies
- `pymupdf` — PDF text-layer extraction (optional `pdf` extra; imported lazily so the rest of
  the pipeline works without it)
- `pytest` + `ruff` — dev tools

## Testing

221 tests covering schemas, parsers, PII redaction, the dataset loader, and all verifiers. Tests
run without API access (verifiers tested with synthetic data, parsers/redaction tested against synthetic text
plus the actual .eml and .pdf files in `invoices/`). `tests/test_pii.py` includes a leak test
asserting the recipient's address never survives in any field of any sample document;
`tests/test_schemas.py` asserts the schema and its generated JSON schema expose no PII-shaped
property and that placeholders are scrubbed; `tests/test_loader.py` asserts a pruned annotation's
ground truth serializes with no dropped value in it and that a v2 `payment` annotation still
loads;
`tests/test_pdf_parser.py` covers the PDF column-layout shapes and skips cleanly when pymupdf
or the sample PDFs are absent.

`tests/test_verifiers.py::TestSchemaCoverage` is a drift guard, not a behaviour test: it probes
every field of `InvoiceExtraction`, `LineItem`, `Payment`, `Discount`, and `Fee` — stating the
field in the ground truth, omitting it in the prediction — and fails if no verifier's score
moves. Companion tests assert the probe tables cover `model_fields` exactly, so adding a schema
field without scoring it (the failure mode that silently drops a field from the reward) breaks
the suite, naming the field in the test id.

**No test hardcodes a value from the real documents.** Sample-driven tests glob `invoices/`
and assert structural properties (every sample yields a subject/sender/body; digit runs in a
subject survive redaction), so filenames, subjects, order ids, amounts, and vendor names from
the private corpus never enter the repository. Fixtures use synthetic stand-ins
(`Jane Roe`, `123 Main St / Springfield / IL / 62704`, `vendor.example`, 555-01xx phones).

Note: `uv run pytest` may resolve a system pytest; use `uv run python -m pytest -q` (and
`uv run python -m ruff check src/ tests/`).
