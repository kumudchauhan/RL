"""The five verifiers, one class each."""

from __future__ import annotations

import json

import pytest

from credit_rlvr.schemas import UnderwritingDecision
from credit_rlvr.verifier import (
    verify_decision,
    verify_dti,
    verify_evidence,
    verify_policy,
    verify_schema,
)

from .test_schemas import make_applicant


def decision_for(**overrides) -> UnderwritingDecision:
    """A well-formed APPROVE for the baseline applicant, with fields overridable."""
    payload = {
        "decision": "APPROVE",
        "dti": 0.40,
        "key_factors": ["DTI inside the auto-approve limit"],
        "evidence": [
            {"feature": "monthly_income", "value": 8000},
            {"feature": "monthly_debt", "value": 3200},
            {"feature": "credit_utilization", "value": 0.42},
            {"feature": "credit_score", "value": 720},
            {"feature": "employment_years", "value": 5},
            {"feature": "requested_loan", "value": 25000},
        ],
    }
    return UnderwritingDecision(**{**payload, **overrides})


class TestVerifySchema:
    def test_accepts_a_valid_dict(self):
        score, decision = verify_schema(decision_for().model_dump())
        assert (score.score, decision.decision) == (1.0, "APPROVE")

    def test_accepts_a_valid_json_string(self):
        score, decision = verify_schema(decision_for().model_dump_json())
        assert score.score == 1.0
        assert decision is not None

    def test_rejects_unparseable_json(self):
        score, decision = verify_schema("{not json")
        assert (score.score, decision) == (0.0, None)
        assert "not JSON" in score.detail

    def test_rejects_a_json_array(self):
        score, decision = verify_schema(json.dumps([1, 2]))
        assert (score.score, decision) == (0.0, None)

    def test_rejects_a_missing_field_and_says_which(self):
        payload = decision_for().model_dump()
        del payload["evidence"]
        score, decision = verify_schema(payload)
        assert (score.score, decision) == (0.0, None)
        assert "evidence" in score.detail

    def test_rejects_a_bad_enum_value(self):
        score, decision = verify_schema({**decision_for().model_dump(), "decision": "MAYBE"})
        assert (score.score, decision) == (0.0, None)

    def test_rejects_an_error_payload_from_a_failed_call(self):
        score, decision = verify_schema({"error": "no tool call"})
        assert (score.score, decision) == (0.0, None)


class TestVerifyDti:
    applicant = make_applicant()  # true DTI 0.40

    def test_exact_scores_full(self):
        assert verify_dti(self.applicant, decision_for(dti=0.40)).score == 1.0

    def test_trailing_zero_is_still_exact(self):
        assert verify_dti(self.applicant, decision_for(dti=0.4)).score == 1.0

    def test_one_cent_off_is_a_rounding_slip(self):
        score = verify_dti(self.applicant, decision_for(dti=0.41))
        assert score.score == 0.6
        assert "rounding slip" in score.detail

    def test_a_few_cents_off_scores_little(self):
        assert verify_dti(self.applicant, decision_for(dti=0.44)).score == 0.2

    def test_far_off_scores_nothing(self):
        assert verify_dti(self.applicant, decision_for(dti=0.75)).score == 0.0

    def test_unrounded_input_is_rounded_before_comparing(self):
        assert verify_dti(self.applicant, decision_for(dti=0.4000001)).score == 1.0

    def test_grading_is_monotone_in_the_error(self):
        errors = [0.40, 0.41, 0.44, 0.90]
        scores = [verify_dti(self.applicant, decision_for(dti=e)).score for e in errors]
        assert scores == sorted(scores, reverse=True)


class TestVerifyDecision:
    applicant = make_applicant()  # rulebook says APPROVE

    def test_matching_label_scores_full(self):
        assert verify_decision(self.applicant, decision_for(decision="APPROVE")).score == 1.0

    def test_one_band_off_gets_partial_credit(self):
        score = verify_decision(self.applicant, decision_for(decision="REFER"))
        assert score.score == 0.3
        assert "one band off" in score.detail

    def test_the_opposite_label_scores_nothing(self):
        score = verify_decision(self.applicant, decision_for(decision="DECLINE"))
        assert score.score == 0.0
        assert "opposite" in score.detail

    def test_refer_is_one_band_from_decline(self):
        applicant = make_applicant(credit_score=500)  # rulebook says DECLINE
        assert verify_decision(applicant, decision_for(decision="REFER")).score == 0.3


class TestVerifyPolicy:
    applicant = make_applicant()  # true DTI 0.40 -> APPROVE

    def test_consistent_and_well_named_scores_full(self):
        decision = decision_for(decision="APPROVE", dti=0.40, key_factors=["DTI is comfortable"])
        assert verify_policy(self.applicant, decision).score == 1.0

    def test_label_that_contradicts_the_reported_dti_loses_the_larger_share(self):
        # Says 0.40 (which approves) but submits DECLINE.
        decision = decision_for(decision="DECLINE", dti=0.40, key_factors=["DTI too high"])
        assert verify_policy(self.applicant, decision).score == pytest.approx(0.3)

    def test_rationale_that_names_nothing_relevant_loses_the_smaller_share(self):
        decision = decision_for(decision="APPROVE", key_factors=["the vibes are good"])
        assert verify_policy(self.applicant, decision).score == pytest.approx(0.7)

    def test_wrong_arithmetic_is_not_charged_here_when_the_label_follows_from_it(self):
        # Reports 0.80: wrong, but DECLINE does follow from 0.80, and the factor
        # names DTI. verify_dti charges the arithmetic; this verifier should not.
        decision = decision_for(decision="DECLINE", dti=0.80, key_factors=["DTI above ceiling"])
        assert verify_policy(self.applicant, decision).score == 1.0

    def test_concept_matching_ignores_case_and_underscores(self):
        applicant = make_applicant(credit_score=500)
        decision = decision_for(decision="DECLINE", dti=0.40, key_factors=["Credit_Score too low"])
        assert verify_policy(applicant, decision).score == 1.0

    def test_naming_the_wrong_driver_loses_the_naming_share(self):
        applicant = make_applicant(credit_utilization=0.95)  # declines on utilization
        decision = decision_for(decision="DECLINE", dti=0.40, key_factors=["employment tenure"])
        assert verify_policy(applicant, decision).score == pytest.approx(0.7)


class TestVerifyEvidence:
    def test_exactly_the_required_features_scores_full(self):
        applicant = make_applicant(credit_score=500)  # D1: needs credit_score only
        decision = decision_for(evidence=[{"feature": "credit_score", "value": 500}])
        assert verify_evidence(applicant, decision).score == 1.0

    def test_approve_needs_all_six(self):
        assert verify_evidence(make_applicant(), decision_for()).score == 1.0

    def test_citing_everything_for_a_single_feature_rule_costs_precision(self):
        applicant = make_applicant(credit_score=500)  # D1: needs credit_score only
        all_six = [
            {"feature": "monthly_income", "value": 8000},
            {"feature": "monthly_debt", "value": 3200},
            {"feature": "credit_utilization", "value": 0.42},
            {"feature": "credit_score", "value": 500},
            {"feature": "employment_years", "value": 5},
            {"feature": "requested_loan", "value": 25000},
        ]
        score = verify_evidence(applicant, decision_for(evidence=all_six))
        # precision 1/6, recall 1/1
        assert score.score == pytest.approx(2 * (1 / 6) / ((1 / 6) + 1), abs=1e-4)
        assert score.score < 0.3

    def test_omitting_a_required_feature_costs_recall(self):
        applicant = make_applicant(monthly_income=10000, monthly_debt=5100)  # D2
        decision = decision_for(
            decision="DECLINE", dti=0.51, evidence=[{"feature": "monthly_debt", "value": 5100}]
        )
        assert verify_evidence(applicant, decision).score == pytest.approx(2 / 3, abs=1e-4)

    def test_a_wrong_value_is_not_a_hit(self):
        applicant = make_applicant(credit_score=500)
        decision = decision_for(evidence=[{"feature": "credit_score", "value": 700}])
        score = verify_evidence(applicant, decision)
        assert score.score == 0.0
        assert "missing credit_score" in score.detail

    def test_a_feature_that_does_not_exist_is_not_a_hit(self):
        applicant = make_applicant(credit_score=500)
        decision = decision_for(evidence=[{"feature": "bankruptcies", "value": 0}])
        assert verify_evidence(applicant, decision).score == 0.0

    def test_a_value_within_tolerance_is_a_hit(self):
        applicant = make_applicant(credit_utilization=0.95)  # D3
        decision = decision_for(evidence=[{"feature": "credit_utilization", "value": 0.9502}])
        assert verify_evidence(applicant, decision).score == 1.0

    def test_duplicate_citations_cost_precision(self):
        applicant = make_applicant(credit_score=500)
        item = {"feature": "credit_score", "value": 500}
        decision = decision_for(evidence=[item, item])
        assert verify_evidence(applicant, decision).score == pytest.approx(2 / 3, abs=1e-4)
