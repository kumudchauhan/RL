# Information Extraction Pipeline

Structured data extraction from invoice emails and order confirmations using Claude's tool-use API, with a Harbor RLVR evaluation framework that produces granular reward signals via composable verifiers.

## At a Glance

|  |  |
|---|---|
| **What** | Parses `.eml` and `.pdf` invoices, then extracts vendor, order, totals, and line items as schema-valid JSON via Claude tool-use |
| **Why** | Turns extraction quality into a *verifiable, granular reward* — the signal an RLVR training loop would need |
| **How it's scored** | Weighted composite: field accuracy 0.25 · numeric accuracy 0.35 · line-item F1 0.40 |
| **Privacy** | Two-tier PII masking inside the parser — nothing unmasked reaches the API, and the document corpus stays out of git |
| **Status** | Evaluation and verifier framework works end to end; policy optimization not yet implemented |
| **Stack** | Python 3.11+, uv, `anthropic`, Pydantic, pymupdf; 98 tests that run with no API key |

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

- **Document** — vendor, order id, order date, delivery date, currency
- **Line items** — product name, quantity, unit price, total price, SKU
- **Money** — subtotal, tax, shipping cost, discount, total
- **Payment** — method, card type, last four
- **Addresses** — shipping and billing

The last two are masked before the document ever reaches the model, so they come back `null`
by design and no verifier scores them — see [PII Redaction](#pii-redaction).

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
  │ pair document + annotation  │   → ExtractionTask
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
  │ verifiers/                  │   field · numeric · line-item
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
| `schemas.py` | Extraction target as Pydantic models; also the source of the tool JSON schema | `InvoiceExtraction`, `LineItem`, `Address`, `PaymentInfo` |
| `parsers/base.py` | Parser interface (`can_handle` / `parse`) and the parsed-document contract | `DocumentParser`, `ParsedDocument` |
| `parsers/eml_parser.py` | `.eml` → text/HTML body, subject, sender, headers | `EmlParser` |
| `parsers/pdf_parser.py` | `.pdf` text layer via pymupdf (lazy import, optional extra) | `PdfParser` |
| `parsers/pii.py` | Two-tier PII masking with stable placeholders, one redactor per document | `PIIPolicy`, `PIIRedactor` |
| `dataset/loader.py` | Match each source document to its annotation, build the eval set | `DatasetLoader` |
| `dataset/tasks.py` | One evaluation instance; serializes to Harbor task format | `ExtractionTask` |
| `agent/prompts.py` | System + user prompt templates | — |
| `agent/extraction_agent.py` | Claude call with `tool_choice={"type": "tool", ...}`; rollout capture | `ExtractionAgent` |
| `verifiers/base.py` | Verifier interface and normalized scoring | `Verifier`, `VerificationResult` |
| `verifiers/field_verifier.py` | Scalar fields — exact and fuzzy string match | `FieldVerifier` |
| `verifiers/numeric_verifier.py` | Monetary fields — tolerance-based comparison | `NumericVerifier` |
| `verifiers/line_item_verifier.py` | Line items — greedy bipartite matching, weighted F1 | `LineItemVerifier` |
| `verifiers/composite.py` | Weighted aggregation into a single training signal | `CompositeVerifier`, `RewardSignal` |
| `runner/evaluate.py` | Harness + `extract-eval` CLI; writes JSON results and the text report | `EvaluationRunner` |

### Contracts between stages

- **`ParsedDocument`** — `format`, `subject`, `sender`, `text_body`, `html_body`, `metadata`
  (incl. redacted `source_name`), `redaction_report` (counts and placeholder names only).
- **`ExtractionTask`** — `task_id`, `source_file`, `parsed_document`, `ground_truth`;
  `to_harbor_task()` emits the Harbor-shaped input/expected_output pair.
- **`InvoiceExtraction`** — the same model serves as the API tool schema and as the type of
  both prediction and ground truth, so a prediction is schema-valid by construction.
- **`VerificationResult`** — `score`, `max_score`, `normalized_score`, plus a `details` dict
  carrying the per-field evidence the report prints.
- **`RewardSignal`** — `overall_reward`, `component_rewards`, `details`;
  `to_harbor_reward()` is the RLVR-facing shape.
- **Rollout record** (`--capture-rollouts`) — `system`, `input`, `output`, `raw_response`,
  `model`, `temperature`, stored alongside the reward in `results/run_*.json`.

### How the reward is composed

Each verifier returns a raw score and a maximum, normalized to `[0, 1]`. The composite is a
fixed weighted sum:

```
overall_reward = 0.25 * field_accuracy
               + 0.35 * numeric_accuracy
               + 0.40 * line_item_f1
```

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
is deliberately preserved; `shipping_address`, `billing_address`, and `payment.last_four`
become unextractable by design, and no verifier scores them.

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
| Field accuracy (fuzzy string) | 0.25 | Easiest — vendor, dates, IDs |
| Numeric accuracy (tolerance) | 0.35 | Harder — must be exact to the cent |
| Line item F1 (greedy match) | 0.40 | Hardest — multi-field matching |

The weights are deliberate: the hardest signal carries the most reward, so a model cannot look
good by getting only the easy fields right. Override them with
`CompositeVerifier(weights={...})` — every run's report also shows what equal weighting would
have scored, for comparison.

## Summary Report

After each evaluation run, a human-readable report is generated (`results/report_*.txt`) that includes:

- **Per-task field breakdown** — every field scored with `+` (correct), `~` (partial), `x` (wrong) alongside predicted vs expected values
- **Equal vs weighted scoring comparison** — side-by-side table showing how uniform 0.33/0.33/0.33 weighting compares to the actual weighted scoring
- **Explanation of why weighted scoring matters** — equal weighting inflates scores by over-crediting easy fields; weighted scoring reflects real-world extraction difficulty

Example excerpt:

```
  Task                         Equal (0.33 each)    Weighted (0.25/0.35/0.40)
  ---------------------------- -------------------- --------------------
  email_order_sample               0.9500               0.9450
  pdf_receipt_sample               0.7100               0.6575
  ---------------------------- -------------------- --------------------
  MEAN                           0.8300               0.8013
```

## Run Tests

98 tests cover the schema, both parsers, PII redaction, and every verifier. No API key and no
network access required — verifiers run on synthetic extractions and redaction is tested against
synthetic documents, so a fresh clone is green.

```bash
uv sync --extra dev
uv run python -m pytest -q            # `uv run pytest` may pick up a system pytest
uv run python -m ruff check src/ tests/
```
