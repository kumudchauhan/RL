# Credit Underwriting RLVR

**In Progress: Not a finished project**

A credit underwriting environment for RL from Verifiable Rewards: synthetic loan applications in,
a structured underwriting decision out, scored by composable verifiers into a single reward. No
LLM judges anywhere in the reward — every check is arithmetic, a set comparison, or a rerun of
the written credit policy.

**V1 Goal**
Build a self-contained financial RLVR simulation environment where an LLM produces a credit
decision, deterministic verifiers calculate reward, and a simple RL/post-training loop can
eventually optimize against that reward.
V1a = complete environment + reward loop — **done**
V1b = actual policy optimization — not started

## At a Glance

|  |  |
|---|---|
| **What** | Hands a policy a loan application and a written credit policy, then scores the decision it submits — label, DTI, stated reasons, and cited evidence |
| **Why** | Underwriting is a natural RLVR target: the arithmetic is exact and the credit policy is written down, so most of decision quality is *checkable* without a human or a judge model |
| **How it's scored** | Composite of four verifiers — decision 0.35 · DTI 0.25 · policy application 0.25 · evidence 0.15 — behind a schema gate |
| **Baselines** | Oracle scores **1.000**; a degenerate always-approve policy scores **0.274**. That gap is the room a real policy has to earn |
| **Privacy** | Every application is generated, not sampled — there is no borrower data in the repository |
| **Status** | V1a complete: environment, verifiers, and reward work end to end. Policy optimization not started |
| **Stack** | Python 3.11+, uv, `anthropic`, Pydantic; 122 tests that run with no API key |

**Note** All underwriting policies and applicant data in this project are synthetic and intended
solely for experimentation purposes.

**Contents** — [Applicant set](#the-applicant-set) · [Credit policy](#the-credit-policy) ·
[Output schema](#what-the-policy-must-output) · [Verifiers](#the-verifiers) ·
[Reward](#the-reward) · [Baselines](#baselines) · [Known limitations](#known-limitations) ·
[Setup](#setup) · [Layout](#layout)

## The Applicant Set

`scenarios/applicants.json` holds ten synthetic applicants. Each carries only **features** —
monthly income, monthly debt, credit utilization, credit score, employment years, requested
loan. Derived metrics are computed in code, never stored, so the data cannot disagree with
itself.

They are chosen to span the decision space and to put pressure on specific parts of it, not
sampled at random. DTI is `monthly_debt / monthly_income`; loan/AI is `requested_loan` over
annual income; the last column is what the rulebook below decides:

| ID | Scenario | DTI | Loan/AI | Score | Util | Yrs | What it probes | → |
|---|---|---|---|---|---|---|---|---|
| APP-001 | `baseline-borderline` | 0.400 | 0.26 | 720 | 0.42 | 5 | The reference profile — every signal middling, no obvious answer | APPROVE |
| APP-002 | `strong-approve` | 0.120 | 0.13 | 790 | 0.08 | 11 | Unambiguous approve; a policy that declines this is broken | APPROVE |
| APP-003 | `clear-decline` | 0.574 | 0.48 | 575 | 0.94 | 0.5 | Unambiguous decline; anchors the low end of the reward scale | DECLINE |
| APP-004 | `short-tenure-strong-credit` | 0.202 | 0.13 | 765 | 0.11 | 0.7 | Is thin job tenure treated as risk or as disqualifying? | REFER |
| APP-005 | `high-score-high-dti` | 0.493 | 0.35 | 781 | 0.31 | 9 | Score and affordability disagree — does a good score mask a bad DTI? | REFER |
| APP-006 | `low-score-no-debt` | 0.000 | 0.14 | 618 | 0.09 | 8 | The inverse conflict, plus the zero-debt edge case | DECLINE |
| APP-007 | `revolving-stress` | 0.239 | 0.21 | 668 | 0.91 | 6 | Comfortable DTI but maxed revolving credit | DECLINE |
| APP-008 | `oversized-request` | 0.250 | 0.93 | 742 | 0.22 | 4 | Good borrower, wrong loan size — should refer, not decline | REFER |
| APP-009 | `dti-threshold-boundary` | 0.430 | 0.24 | 700 | 0.55 | 3 | Sits *exactly* on 0.43 and on the 700 score floor — catches tolerance and comparison-operator bugs | APPROVE |
| APP-010 | `high-income-high-obligations` | 0.550 | 0.30 | 758 | 0.63 | 14 | Large absolute income, thin margin — tests the high-income halo | DECLINE |

Coverage is deliberately monotone across each axis so a reward curve can be read against it:
DTI runs 0.00 → 0.57, scores 575 → 790, utilization 0.08 → 0.94, tenure 0.5 → 14 years. The
outcomes come out 3 approve / 3 refer / 4 decline.

## The Credit Policy

`credit_policy.py` is the rulebook — hypothetical, deliberately small, and *ordered*, so exactly
one rule fires and the reason for a decision is never ambiguous.

**Hard declines.** Any one of these ends the application:

| Rule | Condition |
|---|---|
| `D1` | credit score below **620** |
| `D2` | DTI above **0.50** |
| `D3` | credit utilization above **0.90** |

**Auto-approve (`A1`).** All five gates must pass: score ≥ **700**, DTI ≤ **0.43**,
utilization ≤ **0.60**, employment ≥ **2 years**, requested loan ≤ **0.50 ×** annual income.

**Otherwise `REFER` (`R1`)**, naming exactly the gates that failed.

Two details make the rulebook checkable rather than merely plausible:

- **DTI is rounded half-up to two decimals, and the thresholds apply to that rounded figure.**
  Python's built-in `round` is banker's rounding (`round(0.125, 2)` is `0.12`), which is not what
  an underwriter means. The engine and the verifier share one `round2`, so boundary cases cannot
  disagree.
- **Every rule declares the features it depends on and the concept a sound rationale would
  mention.** A `D1` decline rests on `credit_score` alone; a `REFER` on loan size rests on
  `requested_loan` and `monthly_income`. Those declarations are what "cite your evidence" and
  "name your reasons" are graded against — the grading target is data attached to the rule, not
  a second copy of the policy living inside the verifier.

The prompt the model sees (`POLICY_TEXT`) is interpolated from the same constants the engine
compares against, so the prompt cannot drift from the rules it is being graded on.

## What the Policy Must Output

```json
{
  "decision": "REFER",
  "dti": 0.45,
  "key_factors": ["DTI at policy threshold", "high credit utilization"],
  "evidence": [
    {"feature": "monthly_income", "value": 7000},
    {"feature": "monthly_debt", "value": 3150}
  ]
}
```

`evidence` is structured rather than prose on purpose: a feature name and the value the policy
claims to have read. That is checkable against the application, which free text is not — so
"show your work" becomes a score instead of a hope.

## The Verifiers

Five independent checks, each returning a score in `[0, 1]` and a `detail` string saying why:

| Verifier | What it asks | How it scores |
|---|---|---|
| `schema` | Does the output parse and validate? | Gate: 0 or 1 |
| `dti` | Is the reported ratio the right ratio? | Compared in cents: exact 1.0 · 1¢ off 0.6 · ≤5¢ 0.2 · else 0.0 |
| `policy` | Does the decision follow from the model's own numbers, and does the rationale name the rule that fired? | 0.7 self-consistency + 0.3 naming |
| `decision` | Is the label the one the rulebook reaches? | exact 1.0 · one band off 0.3 · opposite 0.0 |
| `evidence` | Are the cited facts the ones this application turns on, with the values the file shows? | F1 over the fired rule's required features |

Three of those deserve their reasoning spelled out.

**`dti` is graded, not binary.** All-or-nothing would flatten a rounding slip and a model that
never did the division into the same signal. Comparison happens in integer cents, so no float
epsilon creeps into a threshold.

**`policy` scores against the model's *own* reported DTI, not the true one.** Rerunning the
rulebook on what the model said it computed asks a different question from "was the label
right": it catches a correct answer reached by guessing, and a correct answer defended with
unrelated reasoning. Scoring against the reported figure also keeps bad arithmetic charged once,
in `dti`, rather than twice.

**`evidence` is F1, not recall.** Recall alone is trivially gamed by citing all six features
every time. Requiring precision as well means a `D1` decline that cites everything scores 0.29,
not 1.0 — over-citing costs as much as under-citing.

## The Reward

```
total = 0.35·decision + 0.25·dti + 0.25·policy + 0.15·evidence
```

Schema is a **gate, not a slice of the weight**: output that does not validate scores 0.0
overall, and no other component is reported. The other four numbers are undefined for malformed
output, not merely low, and a malformed submission should not collect partial credit.

The decision carries the most weight because it is what a lender acts on. Arithmetic and
reasoning together carry as much, because a right answer for the wrong reason is not a signal
worth training on.

Every `Reward` keeps its component breakdown, so a low score can be attributed rather than just
observed:

```
APP-003  total 0.150
  schema     1.0  valid
  dti        0.0  wrong by 0.57 (said 0.00, is 0.57)
  policy     0.0  rule D1 on the reported DTI gives DECLINE, not APPROVE; factors miss credit_score
  decision   0.0  opposite — said APPROVE, rulebook says DECLINE (D1)
  evidence   1.0  cited 1, 1/1 required matched
```

## Baselines

Two offline policies bracket the reward, and both run with no API key — which is what makes the
reward itself testable:

| Policy | Mean reward | Role |
|---|---|---|
| `oracle` | **1.000** | The ceiling. Applies the rulebook exactly and cites exactly what it rests on. If this ever drops below 1.0, the verifiers disagree with the rulebook and the bug is in the reward, not in a policy |
| `always-approve` | **0.274** | The floor. Approves everything without reading the application, but still emits valid JSON. A real policy has to beat this to have learned anything |

```bash
uv run credit-eval                 # both baselines on the frozen set
uv run credit-eval --policy oracle
```

`ClaudePolicy` is the real thing — Claude Opus 5, adaptive thinking, a forced tool call for the
decision schema. It returns the raw tool input **unvalidated**, so a submission that does not
validate is scored as a failed episode rather than raising, and the rollout is still captured.

## Known Limitations

Written down because they are what the next milestone has to resolve, and an unexamined reward is
worse than a known-imperfect one.

- **`evidence` can pay out for the wrong reason.** It grades citations against the *true* rule's
  features, so a citation can score without being the reason the policy actually gave. This is
  measurable: `always-approve` submits APPROVE against APP-003's DECLINE and still collects full
  evidence credit, because it cites `credit_score` and `D1` happens to rest on `credit_score`
  alone. Same story on APP-006, where it also collects full `dti` credit for reporting `0.00`
  against a genuinely zero-debt applicant. Two lucky-credit paths, worth 0.15 and 0.40 on those
  two episodes; both are pinned by tests so the leak cannot widen unnoticed. The alternative —
  grading evidence against the model's own claimed rule — decomposes the gradient better but
  lets a policy pick a rule arbitrarily and cite it perfectly.
- **`key_factors` is checked by keyword, not comprehension.** `FACTOR_KEYWORDS` catches a
  rationale pointing at the wrong driver. It does not judge whether the prose is any good, and it
  is spoofable by a policy that lists the right nouns without the right reasoning.
- **Ten applicants is a design set, not a measurement set.** Enough to exercise every rule and
  every boundary; too few to say anything statistical about a reward curve.
- **`REFER` has no size dimension.** APP-008 is a good borrower asking for too much, and the
  right commercial answer is a smaller offer. The schema can only say `REFER`, so a counteroffer
  is currently unexpressible and therefore unscoreable.

## Next Technical Milestone

Establish whether this reward can serve as a training signal, and produce the baseline any RLVR
run would have to beat. Concretely: run `ClaudePolicy` on the frozen ten and report per
component; check that the composite is **discriminative** (better submissions reliably score
higher — the four-way ordering test in `test_reward.py` is the seed of this), **stable** (same
input, same reward), and **not gameable** beyond the two leaks above.

## Setup

```bash
uv sync --extra dev
uv run python -m pytest -q     # 122 tests, no API key needed
uv run ruff check .
```

## Layout

```
scenarios/
└── applicants.json  # The frozen applicant set
src/credit_rlvr/
├── schemas.py       # Applicant, UnderwritingDecision, VerifierScore, Reward, Rollout
├── credit_policy.py # The rulebook: thresholds, ordered rules, and the prompt text
├── generator.py     # Loads and validates the frozen set
├── policy.py        # ClaudePolicy + oracle and degenerate baselines
├── verifier.py      # The five verifiers
├── reward.py        # Weights, the schema gate, and the composite
└── environment.py   # Episode loop and the credit-eval CLI
tests/               # 122 tests, no network or API key
notebooks/           # Reward-shaping and calibration exploration
```

A note on the name `policy.py`: in RL, *policy* means the agent; in credit, it means the lender's
rulebook. Here `policy.py` is the agent and `credit_policy.py` is the rulebook.
