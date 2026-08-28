# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An information extraction pipeline that parses invoice emails and order confirmations, extracts structured fields using Claude's tool-use API, and evaluates/optimizes extraction quality using a Harbor-style RLVR (Reinforcement Learning with Verifiable Rewards) framework. Composable verifiers produce granular reward signals suitable for training.

## Architecture

```
src/info_extract/
├── schemas.py              # Pydantic models: InvoiceExtraction, LineItem, Address, PaymentInfo
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
│   ├── field_verifier.py   # Scalar fields (vendor, order_id, dates) — exact/fuzzy match
│   ├── numeric_verifier.py # Monetary fields (total, tax, etc.) — tolerance-based
│   ├── line_item_verifier.py # Line items — greedy bipartite matching with F1
│   └── composite.py        # Weighted aggregation → RewardSignal
├── dataset/
│   ├── loader.py           # Reads annotations/ + parses source docs into tasks
│   └── tasks.py            # ExtractionTask dataclass
└── runner/
    └── evaluate.py         # Main harness + CLI entrypoint (extract-eval)
```

## Key Design Decisions

- **Claude tool-use** for extraction guarantees valid JSON matching the Pydantic schema
- **Granular, composable verifiers** produce per-field reward signals (not binary pass/fail)
- **Weighted composite reward**: field_accuracy=0.25, numeric_accuracy=0.35, line_item_f1=0.40
- **Partial credit scoring**: fuzzy string matching for vendor names, tolerance-based numeric comparison, weighted F1 for line items
- **Rollout capture mode**: stores full prompt/response for RLVR training data generation
- **PII masked at the parser**, before any content reaches the API or a rollout file

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
- **Consequence:** `shipping_address`, `billing_address`, and `payment.last_four` are not
  extractable from redacted text — the model returns null for them. No verifier scores those
  fields (`FieldVerifier` covers vendor/order_id/dates/currency), so composite reward is
  unaffected. Note that files in `annotations/` still hold the real values.
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

## Extraction Schema (fields extracted)

- `vendor`, `order_id`, `order_date`, `delivery_date`
- `shipping_address`, `billing_address` (street, city, state, zip_code, country)
- `line_items[]` (product_name, quantity, unit_price, total_price, sku)
- `subtotal`, `tax`, `shipping_cost`, `discount`, `total`, `currency`
- `payment` (method, last_four, card_type)

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

98 tests covering schemas, parsers, PII redaction, and all verifiers. Tests run without API
access (verifiers tested with synthetic data, parsers/redaction tested against synthetic text
plus the actual .eml and .pdf files in `invoices/`). `tests/test_pii.py` includes a leak test
asserting the recipient's address never survives in any field of any sample document;
`tests/test_pdf_parser.py` covers the PDF column-layout shapes and skips cleanly when pymupdf
or the sample PDFs are absent.

**No test hardcodes a value from the real documents.** Sample-driven tests glob `invoices/`
and assert structural properties (every sample yields a subject/sender/body; digit runs in a
subject survive redaction), so filenames, subjects, order ids, amounts, and vendor names from
the private corpus never enter the repository. Fixtures use synthetic stand-ins
(`Jane Roe`, `123 Main St / Springfield / IL / 62704`, `vendor.example`, 555-01xx phones).

Note: `uv run pytest` may resolve a system pytest; use `uv run python -m pytest -q` (and
`uv run python -m ruff check src/ tests/`).
