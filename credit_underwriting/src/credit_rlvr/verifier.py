"""Composable verifiers over an underwriting decision.

Five independent checks, each scoring one dimension in [0, 1] and saying in
`detail` why. None of them consults an LLM: every one is arithmetic, a set
comparison, or a rerun of the rulebook.

  schema     does the output parse and validate? (a gate — see reward.py)
  dti        is the reported ratio the right ratio?
  policy     does the decision follow from the model's own numbers, and does the
             rationale name the rule that fired?
  decision   is the label the one the rulebook reaches?
  evidence   are the cited facts the ones this application turns on, with the
             values the file actually shows?
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from . import credit_policy
from .credit_policy import FACTOR_KEYWORDS
from .schemas import (
    APPLICANT_FEATURES,
    Applicant,
    Decision,
    Evidence,
    UnderwritingDecision,
    VerifierScore,
    round2,
)

#: How close a cited value has to be to count as the file's value.
VALUE_TOLERANCE = 0.005

#: Pairs one band apart. Calling a REFER an APPROVE is a worse decision than the
#: rulebook's, but it is not the opposite of it.
_ADJACENT: frozenset[frozenset[Decision]] = frozenset(
    {
        frozenset({Decision.APPROVE, Decision.REFER}),
        frozenset({Decision.REFER, Decision.DECLINE}),
    }
)


def verify_schema(raw: dict | str) -> tuple[VerifierScore, UnderwritingDecision | None]:
    """Parse and validate the output. Returns the score and, on success, the decision.

    This runs first and gates the rest: if the output does not validate, the other
    four scores are undefined rather than merely low.
    """
    payload: object = raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return VerifierScore(name="schema", score=0.0, detail=f"not JSON: {exc}"), None

    if not isinstance(payload, dict):
        return (
            VerifierScore(
                name="schema", score=0.0, detail=f"expected an object, got {type(payload).__name__}"
            ),
            None,
        )

    try:
        decision = UnderwritingDecision(**payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "<root>"
        return (
            VerifierScore(name="schema", score=0.0, detail=f"{where}: {first['msg']}"),
            None,
        )

    return VerifierScore(name="schema", score=1.0, detail="valid"), decision


def verify_dti(applicant: Applicant, decision: UnderwritingDecision) -> VerifierScore:
    """Check the arithmetic, in cents, with partial credit for near misses.

    Graded rather than binary so the reward can distinguish a rounding slip from a
    model that never did the division — an all-or-nothing check would flatten the
    two into the same signal.
    """
    truth = round(applicant.dti * 100)
    reported = round(round2(decision.dti) * 100)
    error = abs(reported - truth)
    said = f"said {reported / 100:.2f}, is {truth / 100:.2f}"

    if error == 0:
        return VerifierScore(name="dti", score=1.0, detail=f"exact ({truth / 100:.2f})")
    if error <= 1:
        return VerifierScore(name="dti", score=0.6, detail=f"off by 0.01 — rounding slip ({said})")
    if error <= 5:
        return VerifierScore(name="dti", score=0.2, detail=f"off by {error / 100:.2f} ({said})")
    return VerifierScore(name="dti", score=0.0, detail=f"wrong by {error / 100:.2f} ({said})")


def verify_policy(applicant: Applicant, decision: UnderwritingDecision) -> VerifierScore:
    """Did the model *apply* the rulebook, or reach a label some other way?

    Two things are checkable without reference to the right answer. The decision
    has to follow from the DTI the model itself reported — that catches a correct
    label arrived at by guessing. And the stated factors have to name the rule
    that fired, which catches a correct label with unrelated reasoning.

    Scored against the model's own reported DTI, not the true one, so this stays
    an independent signal from `verify_dti`: bad arithmetic is charged there once,
    not twice.
    """
    ruling = credit_policy.evaluate(applicant, dti=decision.dti)
    consistent = ruling.decision == decision.decision
    named = names_concept(decision.key_factors, ruling.concepts)

    score = (0.7 if consistent else 0.0) + (0.3 if named else 0.0)
    notes = [
        f"rule {ruling.rule_id} on the reported DTI gives {ruling.decision}"
        + ("" if consistent else f", not {decision.decision}"),
        "factors name the driver" if named else f"factors miss {'/'.join(ruling.concepts)}",
    ]
    return VerifierScore(name="policy", score=score, detail="; ".join(notes))


def verify_decision(applicant: Applicant, decision: UnderwritingDecision) -> VerifierScore:
    """Compare the label against the rulebook, with credit for being one band off."""
    truth = credit_policy.evaluate(applicant)

    if decision.decision == truth.decision:
        return VerifierScore(
            name="decision", score=1.0, detail=f"{truth.decision} ({truth.rule_id})"
        )

    said = f"said {decision.decision}, rulebook says {truth.decision} ({truth.rule_id})"
    if frozenset({decision.decision, truth.decision}) in _ADJACENT:
        return VerifierScore(name="decision", score=0.3, detail=f"one band off — {said}")
    return VerifierScore(name="decision", score=0.0, detail=f"opposite — {said}")


def verify_evidence(applicant: Applicant, decision: UnderwritingDecision) -> VerifierScore:
    """F1 over the features the fired rule depends on.

    A citation counts only if the feature is real, its value matches the file, and
    the rule actually depends on it. So omitting a driver costs recall, and citing
    all six features regardless of the rule costs precision — neither
    under-citing nor shotgunning scores well.

    The target is the *true* rule's features. That couples this score to getting
    the decision right, which is a deliberate tradeoff: the question being asked
    is whether the facts this application actually turns on were surfaced.
    """
    truth = credit_policy.evaluate(applicant)
    required = set(truth.required_features)
    hits = {item.feature for item in decision.evidence if _is_hit(applicant, item, required)}

    precision = len(hits) / len(decision.evidence)
    recall = len(hits) / len(required)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    missed = sorted(required - hits)
    detail = f"cited {len(decision.evidence)}, {len(hits)}/{len(required)} required matched"
    if missed:
        detail += f"; missing {', '.join(missed)}"
    return VerifierScore(name="evidence", score=round(f1, 4), detail=detail)


def names_concept(factors: list[str], concepts: tuple[str, ...]) -> bool:
    """Does any stated factor mention any of these concepts?"""
    blob = " ".join(_normalize(factor) for factor in factors)
    return any(keyword in blob for concept in concepts for keyword in FACTOR_KEYWORDS[concept])


def _normalize(text: str) -> str:
    """Lowercase, and treat underscores and hyphens as spaces."""
    return re.sub(r"[_\-]+", " ", text.lower())


def _is_hit(applicant: Applicant, item: Evidence, required: set[str]) -> bool:
    if item.feature not in APPLICANT_FEATURES or item.feature not in required:
        return False
    return abs(item.value - applicant.value_of(item.feature)) <= VALUE_TOLERANCE
