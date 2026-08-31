"""Weighted composite reward.

Aggregates the verifier scores into a single scalar, keeping the per-component
breakdown alongside it so a low reward can be attributed rather than just
observed.

Schema is a gate, not a slice of the weight. Output that does not validate scores
0.0 overall — the other four numbers are undefined in that case, not merely low,
and a malformed submission should not be able to collect partial credit.
"""

from __future__ import annotations

from .schemas import Applicant, Reward
from .verifier import (
    verify_decision,
    verify_dti,
    verify_evidence,
    verify_policy,
    verify_schema,
)

#: Sums to 1.0 (enforced by the tests). The decision carries the most weight
#: because it is what a lender acts on; the arithmetic and the reasoning behind it
#: together carry as much, because a right answer for the wrong reason is not a
#: signal worth training on.
WEIGHTS: dict[str, float] = {
    "dti": 0.25,
    "policy": 0.25,
    "decision": 0.35,
    "evidence": 0.15,
}


def score_episode(applicant: Applicant, raw_output: dict | str) -> Reward:
    """Score one submission against one application."""
    schema_score, decision = verify_schema(raw_output)
    if decision is None:
        return Reward(
            applicant_id=applicant.id,
            total=0.0,
            schema_ok=False,
            components=[schema_score],
        )

    checks = [
        verify_dti(applicant, decision),
        verify_policy(applicant, decision),
        verify_decision(applicant, decision),
        verify_evidence(applicant, decision),
    ]
    weighted = [check.model_copy(update={"weight": WEIGHTS[check.name]}) for check in checks]
    total = sum(check.weighted for check in weighted)

    return Reward(
        applicant_id=applicant.id,
        total=round(total, 4),
        schema_ok=True,
        components=[schema_score, *weighted],
    )
