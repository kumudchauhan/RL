# Information Extraction Pipeline

Structured data extraction from invoice emails and order confirmations using Claude's tool-use API, with a Harbor RLVR evaluation framework that produces granular reward signals via composable verifiers.

## At a Glance

|  |  |
|---|---|
| **What** | Parses `.eml` and `.pdf` invoices, then extracts store and platform, order, billed-vs-paid totals, fees, tender lines, and fully detailed line items as schema-valid JSON via Claude tool-use |
| **Why** | Turns extraction quality into a *verifiable, granular reward* — the signal an RLVR training loop would need |
| **How it's scored** | Weighted composite: field 0.20 · numeric 0.30 · line-item F1 0.35 · line-item detail 0.15 |
| **Privacy** | Two-tier PII masking inside the parser, plus an output schema with no field for a name, address, phone, email, or card number — so there is nothing to leak, masked or otherwise |
| **Status** | Evaluation and verifier framework works end to end; policy optimization not yet implemented |
| **Stack** | Python 3.11+, uv, `anthropic`, Pydantic, pymupdf; 221 tests that run with no API key |

**Contents** — [Current state](#current-project-state) · [Not yet implemented](#not-yet-implemented) ·
[Next milestone](#next-technical-milestone) · [Setup](#setup) · [Usage](#usage) ·
[Technical map](#technical-map) · [PII redaction](#pii-redaction) ·
[Reward weights](#reward-weights) · [Summary report](#summary-report) · [Tests](#run-tests)

## Current Project State

This project currently implements an RLVR-oriented **evaluation and verifier framework** for
structured information extraction. The system:

1. Parses invoice / order-confirmation emails and PDF receipts.
2. Redacts PII in the parser, before any text leaves the machine.
3. Uses Claude tool-use to produce structured extraction.
4. Evaluates predictions against annotated ground truth.
5. Computes granular rewards using field, numeric, and line-item verifiers.
6. Produces a weighted composite reward.
7. Supports rollout capture intended for future RLVR training.

## Not Yet Implemented

Policy optimization / RL training is **not** yet part of the demonstrated pipeline. The
rollout format exists and is captured, but nothing consumes it to update a policy yet.

## Next Technical Milestone

Determine whether the current reward/verifier design can be used as a reliable training
signal for policy optimization, and establish a baseline against which RLVR improvements can
be measured.

Concretely, that means answering:

- Is the composite reward **discriminative** — do better extractions reliably score higher,
  and is the signal smooth enough to learn from rather than near-binary?
- Is it **stable** — same input, same reward, no drift from fuzzy-match thresholds or
  tolerance edges?
- Is it **gameable** — can a degenerate output (empty line items, copied totals) score well?
- What is the **baseline**: prompt-only extraction scored on a frozen annotated set, reported
  per component, so any later RLVR run has something to beat.

## What It Extracts

Input is a raw `.eml` order confirmation or a `.pdf` receipt. Output is an `InvoiceExtraction`
(see `schemas.py`), so every prediction is validated on arrival:

- **Document** — `vendor` (the store the goods came from), `service_provider` (the platform that
  took or fulfilled the order, when the document names one — Instacart, DoorDash), order id,
  order date, delivery date, currency
- **Money, billed and paid kept apart** — `subtotal`, `tax`, `discount`, `total` (what the order
  was billed), `amount_paid` (what actually changed hands), `amount_due` (what is still
  outstanding). They diverge on a split-tender or instalment receipt, so none is derived from
  another
- **Fees, each on its own line** — `shipping_cost`, `delivery_fee`, `service_fee`, `tip`, plus
  `fees[{label, amount}]` for whatever else the document charges (`Bag Fee`, `Bottle Deposit`,
  `Small Order Fee`) with the label kept as printed
- **Discounts and coupons** (order level and per line) — printed description, code, issuer
  (`Groupon`, `manufacturer coupon`, ...), amount, percentage
- **Payment** — `payments[]`, one entry per tender line: method (`credit_card`, `debit_card`,
  `apple_pay`, `google_pay`, `paypal`, `gift_card`, `ebt`, `buy_now_pay_later`, ...), card brand
  as printed, the amount that went onto that instrument, and `installment_count` /
  `installment_amount` when the document states an instalment or EMI plan
- **Line items**
  - product name, quantity (fractional allowed, e.g. `1.24 lb`), unit of measure,
    unit price, line total
  - **every identifier the document prints** — `upc`, `sku`, `asin`, `product_number`, plus
    `other_identifiers[{label, value}]` for anything else (`PLU`, `ISBN`, `Style #`) with the
    label kept as printed
  - **taxonomy the document itself states** — `department` (a section header such as
    `PRODUCE`) and `category` (`Beverages`, `Makeup`)
  - per-item discounts and coupons

Two rules are enforced rather than requested:

- **Nothing is invented.** Every field is optional, so anything the document does not print
  comes back `null` (or `[]`). A category is never inferred from a product name, and a quantity
  is never assumed to be 1.
- **No personal data, in any form.** The schema has *no* field for a name, address, phone
  number, email, or card number — not even a masked or synthetic one. `assert_pii_free_schema()`
  runs at import, `extra="forbid"` rejects any key that is not in the schema, and any redaction
  placeholder that survives into a response (`[PERSON_1]`, `[ADDRESS_2]`) is stripped from the
  output. See [PII Redaction](#pii-redaction).

## Setup

```bash
cd information_extraction
uv sync

# PDF invoices need pymupdf
uv sync --extra pdf
```

The pipeline reads two directories that are **gitignored on purpose**, because the corpus is
personal documents:

- `invoices/` — source `.eml` / `.pdf` documents
- `annotations/` — one ground-truth JSON per document: a `_meta` block (`source_file`,
  `annotator`, `date`) plus the fields of `InvoiceExtraction`

Results land in `results/` (also gitignored). To try it, drop your own documents and matching
annotations into those directories. The test suite does not need them: it runs on synthetic
fixtures, and the few sample-driven parser tests skip cleanly when `invoices/` is empty.

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

## Technical Map

### Data flow

```
  invoices/*.eml, *.pdf
          │
          ▼
  ┌─────────────────────────────┐
  │ parsers/ + pii.py           │   EmlParser | PdfParser
  │ parse → mask PII            │   → ParsedDocument
  └─────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────┐
  │ dataset/loader.py           │   ◄── annotations/*.json (ground truth)
  │ pair, prune dropped fields  │   → ExtractionTask
  └─────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────┐
  │ agent/extraction_agent.py   │   Claude, forced tool-use
  │ prompt → structured output  │   → InvoiceExtraction (prediction)
  └─────────────────────────────┘
          │
          ▼
  ┌─────────────────────────────┐
  │ verifiers/                  │   field · numeric · line-item · detail
  │ prediction vs ground truth  │   → composite.py → RewardSignal
  └─────────────────────────────┘
          │
          ▼
  results/run_*.json  +  results/report_*.txt
  (+ one rollout per task with --capture-rollouts)
```

### Modules

| Path | Responsibility | Key types |
|------|----------------|-----------|
| `schemas.py` | Extraction target as Pydantic models; source of the tool JSON schema; PII-free by construction (import-time guard, `extra="forbid"`, placeholder scrubbing) | `InvoiceExtraction`, `LineItem`, `ProductIdentifier`, `Discount`, `PaymentInfo` |
| `parsers/base.py` | Parser interface (`can_handle` / `parse`) and the parsed-document contract | `DocumentParser`, `ParsedDocument` |
| `parsers/eml_parser.py` | `.eml` → text/HTML body, subject, sender, headers | `EmlParser` |
| `parsers/pdf_parser.py` | `.pdf` text layer via pymupdf (lazy import, optional extra) | `PdfParser` |
| `parsers/pii.py` | Two-tier PII masking with stable placeholders, one redactor per document | `PIIPolicy`, `PIIRedactor` |
| `dataset/loader.py` | Match each source document to its annotation, prune dropped fields, migrate older annotation shapes, build the eval set | `DatasetLoader`, `prune_annotation`, `migrate_annotation` |
| `dataset/tasks.py` | One evaluation instance; serializes to Harbor task format | `ExtractionTask` |
| `agent/prompts.py` | System + user prompt templates | — |
| `agent/extraction_agent.py` | Claude call with `tool_choice={"type": "tool", ...}`; rollout capture | `ExtractionAgent` |
| `verifiers/base.py` | Verifier interface and normalized scoring | `Verifier`, `VerificationResult` |
| `verifiers/field_verifier.py` | Scalar fields — vendor, service provider, order id, dates, currency — exact and fuzzy match | `FieldVerifier` |
| `verifiers/numeric_verifier.py` | Monetary fields, incl. each fee and billed vs paid — tolerance-based comparison | `NumericVerifier` |
| `verifiers/line_item_verifier.py` | Line items — greedy bipartite matching on name/quantity/total, weighted F1 | `LineItemVerifier` |
| `verifiers/detail_verifier.py` | Repeated records — per-item identifiers, taxonomy, unit price, coupons, plus order-level fee and tender lines | `DetailVerifier` |
| `verifiers/composite.py` | Weighted aggregation into a single training signal | `CompositeVerifier`, `RewardSignal` |
| `runner/evaluate.py` | Harness + `extract-eval` CLI; writes JSON results and the text report | `EvaluationRunner` |

### Contracts between stages

- **`ParsedDocument`** — `format`, `subject`, `sender`, `text_body`, `html_body`, `metadata`
  (incl. redacted `source_name`), `redaction_report` (counts and placeholder names only).
- **`ExtractionTask`** — `task_id`, `source_file`, `parsed_document`, `ground_truth`;
  `to_harbor_task()` emits the Harbor-shaped input/expected_output pair.
- **`InvoiceExtraction`** — the same model serves as the API tool schema and as the type of
  both prediction and ground truth, so a prediction is schema-valid by construction.
- **`VerificationResult`** — `score`, `max_score`, `normalized_score`, `applicable`, plus a
  `details` dict carrying the per-field evidence the report prints.
- **`RewardSignal`** — `overall_reward`, `component_rewards`, `applied_components`, `details`;
  `to_harbor_reward()` is the RLVR-facing shape.
- **Rollout record** (`--capture-rollouts`) — `system`, `input`, `output`, `raw_response`,
  `model`, `temperature`, stored alongside the reward in `results/run_*.json`.

### How the reward is composed

Each verifier returns a raw score and a maximum, normalized to `[0, 1]`. The composite is a
weighted sum over the components that had something to verify:

```
overall_reward = 0.20 * field_accuracy
               + 0.30 * numeric_accuracy
               + 0.35 * line_item_f1
               + 0.15 * detail_accuracy
```

A verifier reports `applicable=False` when the ground truth states nothing it can check — an
annotation with no line items, or one that records no UPCs. Those components are dropped and the
remaining weights renormalized, so annotating detail incrementally never reads as a regression,
and a blank annotation never scores as a perfect one.

Weights are overridable — `CompositeVerifier(weights={...})` — and the reasoning behind the
defaults is in [Reward Weights](#reward-weights).

### Extension points

- **New document format** — subclass `DocumentParser`, implement `can_handle` / `parse`, return
  a `ParsedDocument` with PII already masked.
- **New reward component** — subclass `Verifier`, add it to `CompositeVerifier.verifiers`, and
  give it a weight.
- **New PII category** — extend `PIIPolicy` with a toggle and handle it in `PIIRedactor`.

## PII Redaction

Documents are redacted in the parser, before any text reaches the API or a rollout file. Two
tiers, one shared redactor per document:

1. **Envelope** — for email: recipient addresses masked in full; sender addresses keep only the
   vendor domain (`[EMAIL_2]@vendor.example`); routing/identity headers (`Received*`, `*-SPF`,
   `DKIM-*`, `Message-ID`, `List-Unsubscribe`, ...) are dropped. For PDF, the document-info
   dictionary plays the same role: `author`/`title` are redacted and `creator`/`producer`,
   which fingerprint the device that printed the receipt, are dropped.
2. **Invoice content** — postal addresses, emails, phone numbers, card numbers, and personal
   names in both the text and HTML bodies. PDFs additionally get line-level rules, since text
   extraction splits an address into one line per field and prints names in ALL CAPS.

Values are replaced with stable placeholders (`[PERSON_1]`, `[ADDRESS_2]`, `[CARD_1]`) so
document structure survives, and `ParsedDocument.redaction_report` records counts only — never
the original values. Extraction-relevant content (order numbers, totals, dates, product names)
is deliberately preserved.

Redaction is the first of two layers. The second is the schema itself: there is no
`shipping_address`, `billing_address`, `payment.last_four`, or free-text `notes` field to put a
value in, so nothing personal can reach a result file or a captured rollout even in masked or
synthetic form. `assert_pii_free_schema()` fails at import if such a field is ever added,
`extra="forbid"` rejects unknown keys, and any placeholder that survives into a model response
is stripped by `ExtractionModel`. Annotations still hold the real values locally; the loader
prunes them (`prune_annotation`) and logs the field *names* it dropped, never the values.

Configure per category with `PIIPolicy`:

```python
from info_extract.parsers.eml_parser import EmlParser
from info_extract.parsers.pii import PIIPolicy

parser = EmlParser(pii_policy=PIIPolicy(mask_addresses=False, extra_names=("Jane Roe",)))
```

Parsers default to `PIIPolicy.from_env()`, so the document owner's name can be supplied out of
band as a backstop for layouts the heuristics cannot read:

```bash
export INFO_EXTRACT_PII_NAMES="Jane Q Doe,J Doe"
```

## Reward Weights

| Verifier | Weight | Difficulty |
|----------|--------|------------|
| Field accuracy (fuzzy string) | 0.20 | Easiest — vendor, service provider, dates, IDs |
| Numeric accuracy (tolerance) | 0.30 | Harder — every fee and total, exact to the cent, billed kept apart from paid |
| Line item F1 (greedy match) | 0.35 | Hard — name + quantity + line total, matched item by item |
| Detail accuracy (repeated records) | 0.15 | Hardest — UPC/SKU/ASIN/product number, printed taxonomy, unit price, coupons, fee lines, tender lines |

The weights are deliberate: the hardest signal carries the most reward, so a model cannot look
good by getting only the easy fields right. Override them with
`CompositeVerifier(weights={...})` — every run's report also shows what equal weighting would
have scored, for comparison.

## Summary Report

After each evaluation run, a human-readable report is generated (`results/report_*.txt`) that includes:

- **Per-task field breakdown** — every field scored with `+` (correct), `~` (partial), `x` (wrong) alongside predicted vs expected values
- **Detail breakdown** — per identifier/taxonomy/coupon/fee/tender field: how many values the annotation states and how many were transcribed, plus a count of detail the model extracted that the annotation does not yet cover (reported, never scored)
- **Equal vs weighted scoring comparison** — side-by-side table showing how uniform weighting compares to the actual weighted scoring, over the same set of applicable components
- **Explanation of why weighted scoring matters** — equal weighting inflates scores by over-crediting easy fields; weighted scoring reflects real-world extraction difficulty

Example excerpt:

```
  Field Accuracy:
    x vendor           0.00  (pred: "Instacart")
    x service_provider 0.00  (pred: None)
    + order_id         1.00  (pred: "IC-77")

  Numeric Accuracy:
    + delivery_fee     1.00  (pred: $3.99, expected: $3.99)
    + total            1.00  (pred: $51.30, expected: $51.30)
    x amount_paid      0.00  (pred: None, expected: $51.30)

  Detail (identifiers/taxonomy/coupons/fees/payments): 0.732 over 6 annotated value(s)
    + department           1.00  (1 annotated)
    x fees                 0.00  (1 annotated)
    + other_identifiers    1.00  (1 annotated)
    ~ payments             0.39  (1 annotated)
    i extracted but not annotated (unscored): category x1

  Task                         Equal (0.25 each)    Weighted (0.20/0.30/0.35/0.15)
  ---------------------------- -------------------- --------------------
  instacart_split_tender         0.7973               0.8370
  ---------------------------- -------------------- --------------------
  MEAN                           0.7973               0.8370
```

That excerpt is a delivery order the model got mostly right and misread in the ways this schema
exists to catch: it filed the platform as the store, collapsed a gift-card-plus-Visa split into
one payment, dropped a labelled fee line, and inferred a `category` the receipt never printed
(reported, not scored).

## Run Tests

221 tests cover the schema, both parsers, PII redaction, the annotation loader, and every
verifier, including drift guards that fail if a schema field is ever added without a verifier
that scores it. No API key and no
network access required — verifiers run on synthetic extractions and redaction is tested against
synthetic documents, so a fresh clone is green.

```bash
uv sync --extra dev
uv run python -m pytest -q            # `uv run pytest` may pick up a system pytest
uv run python -m ruff check src/ tests/
```
