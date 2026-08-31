"""The credit policy the verifiers score against.

A decision is only checkable if the rulebook is written down, so it lives here as
data: ordered rules, each naming the features it depends on and the concept a
sound rationale would mention. The verifiers read those two lists — they are what
"cite your evidence" and "name your reasons" are graded against.

Note the name: in RL, *policy* means the agent (that is `policy.py`). Here it
means the lender's rulebook. This module is the rulebook.

`POLICY_TEXT` — what the model is told — is interpolated from the same constants
the engine compares against, so the prompt cannot drift from the rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .schemas import Applicant, Decision, round2

# --- Hard declines. Any single one of these ends the application. -------------
SCORE_FLOOR = 620
DTI_CEILING = 0.50
UTILIZATION_CEILING = 0.90

# --- Auto-approve gates. All five must pass. ----------------------------------
APPROVE_MIN_SCORE = 700
APPROVE_MAX_DTI = 0.43
APPROVE_MAX_UTILIZATION = 0.60
APPROVE_MIN_EMPLOYMENT_YEARS = 2.0
APPROVE_MAX_LOAN_TO_INCOME = 0.50

#: Words that count as naming a concept in a stated rationale. Coarse on purpose:
#: it catches a rationale pointing at the wrong driver, and does not pretend to
#: judge whether the prose is any good.
FACTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dti": ("dti", "debt to income", "debt-to-income", "debt ratio", "obligations"),
    "credit_score": ("score", "fico", "credit history", "credit file"),
    "utilization": ("utilization", "utilisation", "revolving"),
    "employment": ("employment", "tenure", "job", "time in role", "years at"),
    "loan_size": ("loan size", "loan amount", "amount requested", "requested loan",
                  "loan to income", "loan-to-income"),
}


@dataclass(frozen=True)
class Ruling:
    """The rulebook's verdict, plus everything needed to check a claim about it."""

    decision: Decision
    rule_id: str
    reason: str
    #: Features the fired rule actually depends on — the evidence target.
    required_features: tuple[str, ...]
    #: Concepts a sound rationale would mention — the key_factors target.
    concepts: tuple[str, ...]


@dataclass(frozen=True)
class Gate:
    """One auto-approve condition."""

    label: str
    concept: str
    features: tuple[str, ...]
    passes: Callable[[Applicant, float], bool]


#: Evaluated together; a REFER names exactly the gates that failed.
APPROVE_GATES: tuple[Gate, ...] = (
    Gate(
        "credit score",
        "credit_score",
        ("credit_score",),
        lambda applicant, dti: applicant.credit_score >= APPROVE_MIN_SCORE,
    ),
    Gate(
        "DTI",
        "dti",
        ("monthly_income", "monthly_debt"),
        lambda applicant, dti: dti <= APPROVE_MAX_DTI,
    ),
    Gate(
        "credit utilization",
        "utilization",
        ("credit_utilization",),
        lambda applicant, dti: applicant.credit_utilization <= APPROVE_MAX_UTILIZATION,
    ),
    Gate(
        "employment tenure",
        "employment",
        ("employment_years",),
        lambda applicant, dti: applicant.employment_years >= APPROVE_MIN_EMPLOYMENT_YEARS,
    ),
    Gate(
        "loan size",
        "loan_size",
        ("requested_loan", "monthly_income"),
        lambda applicant, dti: applicant.loan_to_income <= APPROVE_MAX_LOAN_TO_INCOME,
    ),
)


POLICY_TEXT = f"""\
Hard declines. If any one of these holds, the decision is DECLINE:
  D1  credit score below {SCORE_FLOOR}
  D2  DTI above {DTI_CEILING:.2f}
  D3  credit utilization above {UTILIZATION_CEILING:.2f}

Auto-approve. The decision is APPROVE only if all five of these hold:
  credit score is at least {APPROVE_MIN_SCORE}
  DTI is at most {APPROVE_MAX_DTI:.2f}
  credit utilization is at most {APPROVE_MAX_UTILIZATION:.2f}
  employment is at least {APPROVE_MIN_EMPLOYMENT_YEARS:.0f} years
  requested loan is at most {APPROVE_MAX_LOAN_TO_INCOME:.2f} x annual income

Otherwise the decision is REFER.

DTI is monthly_debt / monthly_income, rounded half-up to two decimals. The
thresholds are applied to that rounded figure, so a DTI of exactly \
{APPROVE_MAX_DTI:.2f} still clears the auto-approve limit.
Annual income is monthly_income x 12.
"""


def _dedupe(features: tuple[str, ...]) -> tuple[str, ...]:
    """Order-preserving dedupe — two gates can depend on the same feature."""
    return tuple(dict.fromkeys(features))


def evaluate(applicant: Applicant, dti: float | None = None) -> Ruling:
    """Apply the rulebook: hard declines first, then the auto-approve gates.

    `dti` overrides the applicant's own ratio. That is what lets the verifier
    re-run the rulebook on the DTI the model *reported* and ask whether the
    model's decision follows from the model's own arithmetic.
    """
    dti = applicant.dti if dti is None else round2(dti)

    if applicant.credit_score < SCORE_FLOOR:
        return Ruling(
            Decision.DECLINE,
            "D1",
            f"credit score {applicant.credit_score} is below the {SCORE_FLOOR} floor",
            ("credit_score",),
            ("credit_score",),
        )
    if dti > DTI_CEILING:
        return Ruling(
            Decision.DECLINE,
            "D2",
            f"DTI {dti:.2f} is above the {DTI_CEILING:.2f} ceiling",
            ("monthly_income", "monthly_debt"),
            ("dti",),
        )
    if applicant.credit_utilization > UTILIZATION_CEILING:
        return Ruling(
            Decision.DECLINE,
            "D3",
            f"credit utilization {applicant.credit_utilization:.2f} is above the "
            f"{UTILIZATION_CEILING:.2f} ceiling",
            ("credit_utilization",),
            ("utilization",),
        )

    failed = tuple(gate for gate in APPROVE_GATES if not gate.passes(applicant, dti))

    if not failed:
        return Ruling(
            Decision.APPROVE,
            "A1",
            "credit score, DTI, utilization, employment tenure and loan size are all "
            "inside the auto-approve limits",
            _dedupe(tuple(f for gate in APPROVE_GATES for f in gate.features)),
            tuple(gate.concept for gate in APPROVE_GATES),
        )

    return Ruling(
        Decision.REFER,
        "R1",
        "outside the auto-approve limits on " + ", ".join(gate.label for gate in failed),
        _dedupe(tuple(f for gate in failed for f in gate.features)),
        tuple(gate.concept for gate in failed),
    )
