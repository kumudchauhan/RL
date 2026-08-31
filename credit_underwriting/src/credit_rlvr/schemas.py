"""Pydantic models for the credit underwriting environment.

Defines the contract every other module speaks: the application handed to the
policy, the decision the policy returns, and the per-verifier scores the reward
is built from. Validating on arrival keeps malformed model output from reaching
the verifiers.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: The only features a policy may cite as evidence. Nothing else is in the file,
#: so anything else is a hallucination.
APPLICANT_FEATURES: tuple[str, ...] = (
    "monthly_income",
    "monthly_debt",
    "credit_utilization",
    "credit_score",
    "employment_years",
    "requested_loan",
)


def round2(value: float) -> float:
    """Round half-up to two decimals.

    The built-in `round` is banker's rounding, so `round(0.125, 2)` is 0.12. Ratios
    here land on policy thresholds, so the tie-break has to be the one a human
    underwriter would use — and it has to be the same in the policy engine and in
    the verifier, or the boundary cases disagree.
    """
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class Decision(StrEnum):
    """The three outcomes the policy can reach."""

    APPROVE = "APPROVE"
    REFER = "REFER"
    DECLINE = "DECLINE"


class Applicant(BaseModel):
    """One loan application.

    Features only — `dti` and `loan_to_income` are derived on read rather than
    stored, so the data cannot contradict itself.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    scenario: str
    monthly_income: float = Field(gt=0)
    monthly_debt: float = Field(ge=0)
    credit_utilization: float = Field(ge=0.0, le=1.0)
    credit_score: int = Field(ge=300, le=850)
    employment_years: float = Field(ge=0)
    requested_loan: float = Field(gt=0)

    @property
    def dti(self) -> float:
        """Debt-to-income, rounded the way the policy thresholds expect."""
        return round2(self.monthly_debt / self.monthly_income)

    @property
    def loan_to_income(self) -> float:
        """Requested loan as a multiple of annual income."""
        return round2(self.requested_loan / (self.monthly_income * 12))

    def value_of(self, feature: str) -> float:
        """Read one citable feature by name. Raises for anything not citable."""
        if feature not in APPLICANT_FEATURES:
            raise KeyError(feature)
        return getattr(self, feature)


class Evidence(BaseModel):
    """One citation: an applicant feature, and the value the policy read from it.

    Structured rather than prose so the verifier can check it against the
    application instead of taking the model's word for it.
    """

    model_config = ConfigDict(extra="forbid")

    feature: str
    value: float


class UnderwritingDecision(BaseModel):
    """What the policy must return. Every field is checkable."""

    model_config = ConfigDict(extra="forbid")

    decision: Decision
    dti: float = Field(ge=0)
    key_factors: list[str] = Field(min_length=1, max_length=4)
    evidence: list[Evidence] = Field(min_length=1, max_length=len(APPLICANT_FEATURES))

    @field_validator("key_factors")
    @classmethod
    def _factors_are_substantive(cls, factors: list[str]) -> list[str]:
        cleaned = [factor.strip() for factor in factors]
        if any(not factor for factor in cleaned):
            raise ValueError("key_factors entries must not be blank")
        return cleaned


class VerifierScore(BaseModel):
    """One verifier's verdict on one decision.

    `weight` is stamped on by the reward, not by the verifier — a verifier knows
    how well the policy did on its dimension, not how much that dimension counts.
    """

    name: str
    score: float = Field(ge=0.0, le=1.0)
    detail: str
    weight: float = Field(default=0.0, ge=0.0)

    @property
    def weighted(self) -> float:
        return self.score * self.weight


class Reward(BaseModel):
    """The scalar an RL loop would consume, plus the breakdown behind it."""

    applicant_id: str
    total: float = Field(ge=0.0, le=1.0)
    schema_ok: bool
    components: list[VerifierScore]

    def component(self, name: str) -> VerifierScore:
        """Look up one component by name. Raises if that verifier did not run."""
        for component in self.components:
            if component.name == name:
                return component
        raise KeyError(name)


class Rollout(BaseModel):
    """One episode, captured in the form a training loop would replay."""

    applicant: Applicant
    raw_output: dict
    reward: Reward
