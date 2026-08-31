"""Composite reward: the weights, the schema gate, and the baseline bounds."""

from __future__ import annotations

import pytest

from credit_rlvr.generator import load_applicants
from credit_rlvr.policy import AlwaysApprovePolicy, OraclePolicy
from credit_rlvr.reward import WEIGHTS, score_episode

from .test_schemas import make_applicant
from .test_verifier import decision_for


@pytest.fixture(scope="module")
def applicants():
    return load_applicants()


class TestWeights:
    def test_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_cover_every_scored_verifier(self):
        assert set(WEIGHTS) == {"dti", "policy", "decision", "evidence"}


class TestSchemaGate:
    def test_malformed_output_scores_zero_overall(self):
        reward = score_episode(make_applicant(), "{not json")
        assert reward.total == 0.0
        assert reward.schema_ok is False

    def test_malformed_output_reports_no_other_component(self):
        reward = score_episode(make_applicant(), {"decision": "APPROVE"})
        assert [c.name for c in reward.components] == ["schema"]

    def test_valid_output_reports_all_five_components(self):
        reward = score_episode(make_applicant(), decision_for().model_dump())
        assert [c.name for c in reward.components] == [
            "schema",
            "dti",
            "policy",
            "decision",
            "evidence",
        ]

    def test_schema_earns_no_weight_of_its_own(self):
        reward = score_episode(make_applicant(), decision_for().model_dump())
        assert reward.component("schema").weight == 0.0


class TestOracle:
    """The reward's ceiling. If this drops below 1.0, the bug is in the reward."""

    @pytest.mark.parametrize("index", range(10))
    def test_scores_a_perfect_one_on_every_applicant(self, applicants, index):
        applicant = applicants[index]
        reward = score_episode(applicant, OraclePolicy().decide(applicant))
        assert reward.total == 1.0, f"{applicant.id}: {[c.detail for c in reward.components]}"


class TestGameability:
    """The floor. A policy that ignores the application must not score well."""

    def test_always_approve_stays_well_under_the_oracle(self, applicants):
        policy = AlwaysApprovePolicy()
        rewards = [score_episode(a, policy.decide(a)).total for a in applicants]
        assert sum(rewards) / len(rewards) < 0.35

    def test_a_clear_decline_leaves_only_the_lucky_evidence_slice(self, applicants):
        """Pins a known leak in the evidence verifier.

        On APP-003 the degenerate policy submits APPROVE against a DECLINE, and
        still collects full evidence credit: it cites `credit_score`, which is
        exactly what rule D1 depends on. Evidence is graded against the *true*
        rule's features, so a citation can score without being the reason the
        policy gave. That is the reward's floor, not zero — asserted here so the
        leak cannot widen unnoticed.
        """
        applicant = next(a for a in applicants if a.id == "APP-003")
        reward = score_episode(applicant, AlwaysApprovePolicy().decide(applicant))
        assert reward.total == pytest.approx(WEIGHTS["evidence"])
        assert reward.component("evidence").score == 1.0
        for name in ("dti", "policy", "decision"):
            assert reward.component(name).score == 0.0

    def test_it_is_still_beaten_on_the_ones_it_gets_right(self, applicants):
        # APP-001 is a true APPROVE, so the degenerate policy gets the label free.
        # It should still lose the arithmetic, the reasoning and the evidence.
        applicant = next(a for a in applicants if a.id == "APP-001")
        reward = score_episode(applicant, AlwaysApprovePolicy().decide(applicant))
        assert reward.component("decision").score == 1.0
        assert reward.component("dti").score == 0.0
        assert reward.total < 0.6


class TestDiscrimination:
    """A better submission has to score higher, or there is nothing to learn from."""

    def test_reward_is_ordered_by_submission_quality(self):
        applicant = make_applicant(credit_score=500)  # DECLINE on D1
        perfect = {
            "decision": "DECLINE",
            "dti": 0.40,
            "key_factors": ["credit score below the floor"],
            "evidence": [{"feature": "credit_score", "value": 500}],
        }
        vague = {**perfect, "key_factors": ["not a good fit"]}
        wrong_math = {**vague, "dti": 0.99}
        wrong_label = {**wrong_math, "decision": "APPROVE"}

        totals = [score_episode(applicant, p).total for p in (perfect, vague, wrong_math, wrong_label)]
        assert totals == sorted(totals, reverse=True)
        assert len(set(totals)) == 4, f"not discriminative: {totals}"

    def test_reward_is_stable_across_repeated_scoring(self):
        applicant = make_applicant()
        payload = decision_for().model_dump()
        assert len({score_episode(applicant, payload).total for _ in range(5)}) == 1
