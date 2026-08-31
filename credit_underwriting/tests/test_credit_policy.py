"""The rulebook, pinned against the frozen applicant set."""

from __future__ import annotations

import pytest

from credit_rlvr import credit_policy
from credit_rlvr.generator import load_applicants
from credit_rlvr.schemas import Decision

from .test_schemas import make_applicant

#: The decision the rulebook must reach for each of the ten. This table is the
#: regression guard: change a threshold and it tells you exactly who moved.
EXPECTED = {
    "APP-001": (Decision.APPROVE, "A1"),
    "APP-002": (Decision.APPROVE, "A1"),
    "APP-003": (Decision.DECLINE, "D1"),
    "APP-004": (Decision.REFER, "R1"),
    "APP-005": (Decision.REFER, "R1"),
    "APP-006": (Decision.DECLINE, "D1"),
    "APP-007": (Decision.DECLINE, "D3"),
    "APP-008": (Decision.REFER, "R1"),
    "APP-009": (Decision.APPROVE, "A1"),
    "APP-010": (Decision.DECLINE, "D2"),
}


@pytest.fixture(scope="module")
def applicants():
    return {applicant.id: applicant for applicant in load_applicants()}


class TestFrozenSet:
    def test_loads_ten_valid_applicants(self, applicants):
        assert len(applicants) == 10

    @pytest.mark.parametrize("applicant_id", sorted(EXPECTED))
    def test_reaches_the_expected_ruling(self, applicants, applicant_id):
        expected_decision, expected_rule = EXPECTED[applicant_id]
        ruling = credit_policy.evaluate(applicants[applicant_id])
        assert (ruling.decision, ruling.rule_id) == (expected_decision, expected_rule)

    def test_covers_all_three_outcomes(self, applicants):
        reached = {credit_policy.evaluate(a).decision for a in applicants.values()}
        assert reached == {Decision.APPROVE, Decision.REFER, Decision.DECLINE}


class TestHardDeclines:
    def test_score_floor_is_exclusive(self):
        assert credit_policy.evaluate(make_applicant(credit_score=620)).decision != Decision.DECLINE
        assert credit_policy.evaluate(make_applicant(credit_score=619)).rule_id == "D1"

    def test_dti_ceiling_is_exclusive(self):
        at_ceiling = make_applicant(monthly_income=10000, monthly_debt=5000)  # 0.50
        assert credit_policy.evaluate(at_ceiling).decision == Decision.REFER
        over = make_applicant(monthly_income=10000, monthly_debt=5100)  # 0.51
        assert credit_policy.evaluate(over).rule_id == "D2"

    def test_utilization_ceiling_is_exclusive(self):
        assert credit_policy.evaluate(make_applicant(credit_utilization=0.90)).rule_id != "D3"
        assert credit_policy.evaluate(make_applicant(credit_utilization=0.91)).rule_id == "D3"

    def test_score_floor_outranks_everything_else(self):
        # Both D1 and D3 hold; the ordered rulebook must report D1.
        applicant = make_applicant(credit_score=500, credit_utilization=0.99)
        assert credit_policy.evaluate(applicant).rule_id == "D1"


class TestApproveGates:
    def test_dti_at_the_approve_limit_still_approves(self, applicants):
        # APP-009 sits on 0.43 exactly, and on the 700 score floor.
        applicant = applicants["APP-009"]
        assert applicant.dti == credit_policy.APPROVE_MAX_DTI
        assert credit_policy.evaluate(applicant).decision == Decision.APPROVE

    def test_a_hair_over_the_approve_dti_refers(self):
        applicant = make_applicant(monthly_income=7000, monthly_debt=3020)  # 0.43142 -> 0.43
        assert credit_policy.evaluate(applicant).decision == Decision.APPROVE
        applicant = make_applicant(monthly_income=7000, monthly_debt=3060)  # 0.4371 -> 0.44
        assert credit_policy.evaluate(applicant).decision == Decision.REFER

    def test_short_tenure_alone_refers(self):
        ruling = credit_policy.evaluate(make_applicant(monthly_debt=1000, employment_years=1.0))
        assert ruling.decision == Decision.REFER
        assert ruling.required_features == ("employment_years",)
        assert ruling.concepts == ("employment",)

    def test_refer_names_every_failing_gate(self):
        applicant = make_applicant(
            monthly_debt=3300,  # DTI 0.41 -> passes; utilization and tenure fail
            credit_utilization=0.75,
            employment_years=0.5,
        )
        ruling = credit_policy.evaluate(applicant)
        assert ruling.decision == Decision.REFER
        assert set(ruling.concepts) == {"utilization", "employment"}
        assert set(ruling.required_features) == {"credit_utilization", "employment_years"}

    def test_oversized_loan_refers_and_cites_both_inputs(self, applicants):
        ruling = credit_policy.evaluate(applicants["APP-008"])
        assert ruling.decision == Decision.REFER
        assert set(ruling.required_features) == {"requested_loan", "monthly_income"}

    def test_approve_requires_every_feature_as_evidence(self, applicants):
        ruling = credit_policy.evaluate(applicants["APP-002"])
        assert len(ruling.required_features) == 6


class TestReportedDtiOverride:
    def test_override_changes_the_ruling(self):
        applicant = make_applicant()  # true DTI 0.40 -> APPROVE
        assert credit_policy.evaluate(applicant).decision == Decision.APPROVE
        assert credit_policy.evaluate(applicant, dti=0.62).rule_id == "D2"

    def test_override_is_rounded_the_same_way(self):
        applicant = make_applicant()
        assert credit_policy.evaluate(applicant, dti=0.504).decision == Decision.REFER
        assert credit_policy.evaluate(applicant, dti=0.505).rule_id == "D2"


class TestPolicyText:
    """The prompt is interpolated from the constants, so it cannot drift."""

    @pytest.mark.parametrize(
        "threshold",
        ["620", "0.50", "0.90", "700", "0.43", "0.60", "2 years"],
    )
    def test_states_each_threshold(self, threshold):
        assert threshold in credit_policy.POLICY_TEXT

    def test_states_every_outcome(self):
        for label in ("APPROVE", "REFER", "DECLINE"):
            assert label in credit_policy.POLICY_TEXT
