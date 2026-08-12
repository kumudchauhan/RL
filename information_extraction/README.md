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
├── parsers/                # Document parsing (.eml)
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
