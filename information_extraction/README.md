# Information Extraction Pipeline

Structured data extraction from invoice emails and order confirmations using Claude's tool-use API, with a Harbor RLVR evaluation framework that produces granular reward signals via composable verifiers.

## What It Does

Takes raw `.eml` email files (order confirmations, receipts) and extracts key information out of it:
- Vendor, order ID, order date, invoice date, total spent
- Line items information (product, quantity, price, SKU)

Evaluates extraction quality against human-annotated ground truth using weighted verifiers (field accuracy, numeric accuracy, line item F1).

## Setup

```bash
cd information_extraction
uv sync
```

## Usage

```bash
# Run evaluation against annotated samples
export ANTHROPIC_API_KEY=sk-...
uv run extract-eval

# With rollout capture for RLVR training data
uv run extract-eval --capture-rollouts

# Custom model
uv run extract-eval --model claude-haiku-4-20250514

# Disable PII masking (debugging only — sends raw document text to the API)
uv run extract-eval --no-pii-redaction
```

## PII Redaction

Documents are redacted in the parser, before any text reaches the API or a rollout file. Two
tiers, one shared redactor per document:

1. **Envelope** — recipient addresses masked in full; sender addresses keep only the vendor
   domain (`[EMAIL_2]@vendor.example`); routing/identity headers (`Received*`, `*-SPF`, `DKIM-*`,
   `Message-ID`, `List-Unsubscribe`, ...) are dropped.
2. **Invoice content** — postal addresses, emails, phone numbers, card numbers, and personal
   names in both the text and HTML bodies.

Values are replaced with stable placeholders (`[PERSON_1]`, `[ADDRESS_2]`, `[CARD_1]`) so
document structure survives, and `ParsedDocument.redaction_report` records counts only — never
the original values. Extraction-relevant content (order numbers, totals, dates, product names)
is deliberately preserved; `shipping_address`, `billing_address`, and `payment.last_four`
become unextractable by design, and no verifier scores them.

Configure per category with `PIIPolicy`:

```python
from info_extract.parsers.eml_parser import EmlParser
from info_extract.parsers.pii import PIIPolicy

parser = EmlParser(pii_policy=PIIPolicy(mask_addresses=False, extra_names=("Ravi",)))
```

## Run Tests

```bash
uv sync --extra dev
uv run pytest
```

## Project Structure

```
src/info_extract/
├── schemas.py              # Pydantic extraction schema
├── parsers/                # Document parsing (.eml) + two-tier PII redaction
├── agent/                  # Claude API extraction (tool-use)
├── verifiers/              # Field, numeric, line item verifiers + composite reward
├── dataset/                # Annotation loader + task management
└── runner/                 # Evaluation harness + CLI entrypoint
```

## Reward Weights

| Verifier | Weight | Difficulty |
|----------|--------|------------|
| Field accuracy (fuzzy string) | 0.25 | Easiest — vendor, dates, IDs |
| Numeric accuracy (tolerance) | 0.35 | Harder — must be exact to the cent |
| Line item F1 (greedy match) | 0.40 | Hardest — multi-field matching |

## Summary Report

After each evaluation run, a human-readable report is generated (`results/report_*.txt`) that includes:

- **Per-task field breakdown** — every field scored with `+` (correct), `~` (partial), `x` (wrong) alongside predicted vs expected values
- **Equal vs weighted scoring comparison** — side-by-side table showing how uniform 0.33/0.33/0.33 weighting compares to the actual weighted scoring
- **Explanation of why weighted scoring matters** — equal weighting inflates scores by over-crediting easy fields; weighted scoring reflects real-world extraction difficulty

Example excerpt:

```
  Task                         Equal (0.33 each)    Weighted (0.25/0.35/0.40)
  ---------------------------- -------------------- --------------------
  cosmetics_order_order            0.9500               0.9450
  store_receipt_receipt            0.7100               0.6575
  ---------------------------- -------------------- --------------------
  MEAN                           0.8300               0.8013
```
