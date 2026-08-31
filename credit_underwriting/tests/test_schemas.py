"""Schema and derived-ratio tests."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from credit_rlvr.schemas import Applicant, UnderwritingDecision, round2


def make_applicant(**overrides) -> Applicant:
    fields = {
        "id": "T-001",
        "scenario": "test",
        "monthly_income": 8000,
        "monthly_debt": 3200,
        "credit_utilization": 0.42,
        "credit_score": 720,
        "employment_years": 5,
        "requested_loan": 25000,
    }
    return Applicant(**{**fields, **overrides})


class TestRound2:
    def test_rounds_half_up_not_bankers(self):
        # round(0.125, 2) is 0.12 in Python; an underwriter would say 0.13.
        assert round2(0.125) == 0.13
        assert round2(0.135) == 0.14

    def test_leaves_exact_values_alone(self):
        assert round2(0.43) == 0.43
        assert round2(0.0) == 0.0


class TestApplicant:
    def test_derives_dti(self):
        assert make_applicant().dti == 0.40

    def test_derives_loan_to_income(self):
        assert make_applicant().loan_to_income == 0.26

    def test_zero_debt_gives_zero_dti(self):
        assert make_applicant(monthly_debt=0).dti == 0.0

    def test_rounds_dti_to_two_decimals(self):
        # 3550 / 7200 == 0.49305...
        assert make_applicant(monthly_income=7200, monthly_debt=3550).dti == 0.49

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            make_applicant().monthly_income = 1

    def test_rejects_zero_income(self):
        with pytest.raises(ValidationError):
            make_applicant(monthly_income=0)

    def test_rejects_utilization_above_one(self):
        with pytest.raises(ValidationError):
            make_applicant(credit_utilization=1.4)

    def test_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            make_applicant(co_applicant_income=4000)

    def test_value_of_reads_citable_features(self):
        assert make_applicant().value_of("credit_score") == 720

    def test_value_of_rejects_non_citable(self):
        with pytest.raises(KeyError):
            make_applicant().value_of("scenario")


class TestUnderwritingDecision:
    payload: ClassVar[dict] = {
        "decision": "REFER",
        "dti": 0.45,
        "key_factors": ["DTI at policy threshold", "high credit utilization"],
        "evidence": [{"feature": "monthly_debt", "value": 3200}],
    }

    def test_accepts_a_well_formed_decision(self):
        assert UnderwritingDecision(**self.payload).decision == "REFER"

    def test_rejects_unknown_decision_label(self):
        with pytest.raises(ValidationError):
            UnderwritingDecision(**{**self.payload, "decision": "MAYBE"})

    def test_rejects_blank_key_factor(self):
        with pytest.raises(ValidationError):
            UnderwritingDecision(**{**self.payload, "key_factors": ["  "]})

    def test_strips_key_factors(self):
        decision = UnderwritingDecision(**{**self.payload, "key_factors": ["  high DTI "]})
        assert decision.key_factors == ["high DTI"]

    def test_rejects_empty_key_factors(self):
        with pytest.raises(ValidationError):
            UnderwritingDecision(**{**self.payload, "key_factors": []})

    def test_rejects_empty_evidence(self):
        with pytest.raises(ValidationError):
            UnderwritingDecision(**{**self.payload, "evidence": []})

    def test_rejects_more_evidence_than_there_are_features(self):
        item = {"feature": "credit_score", "value": 720}
        with pytest.raises(ValidationError):
            UnderwritingDecision(**{**self.payload, "evidence": [item] * 7})

    def test_rejects_extra_top_level_key(self):
        with pytest.raises(ValidationError):
            UnderwritingDecision(**{**self.payload, "confidence": 0.9})

    def test_rejects_extra_evidence_key(self):
        payload = {**self.payload, "evidence": [{"feature": "monthly_debt", "value": 1, "why": "x"}]}
        with pytest.raises(ValidationError):
            UnderwritingDecision(**payload)
