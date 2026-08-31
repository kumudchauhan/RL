"""The episode loop, the summary, and the Claude policy's request shape."""

from __future__ import annotations

from typing import ClassVar

import pytest

from credit_rlvr.environment import CreditEnvironment, summarize
from credit_rlvr.policy import AlwaysApprovePolicy, ClaudePolicy, OraclePolicy
from credit_rlvr.schemas import Applicant

from .test_schemas import make_applicant


class TestEnvironment:
    def test_loads_the_frozen_set_by_default(self):
        assert len(CreditEnvironment().applicants) == 10

    def test_runs_one_episode_into_a_rollout(self):
        applicant = make_applicant()
        rollout = CreditEnvironment([applicant]).run_episode(applicant, OraclePolicy())
        assert rollout.applicant == applicant
        assert rollout.reward.total == 1.0
        assert rollout.raw_output["decision"] == "APPROVE"

    def test_runs_every_applicant_in_file_order(self):
        environment = CreditEnvironment()
        rollouts = environment.run(OraclePolicy())
        assert [r.applicant.id for r in rollouts] == [a.id for a in environment.applicants]

    def test_captures_the_raw_output_even_when_it_is_invalid(self):
        class Broken:
            name = "broken"

            def decide(self, applicant: Applicant) -> dict:
                return {"error": "no tool call"}

        rollout = CreditEnvironment().run_episode(make_applicant(), Broken())
        assert rollout.raw_output == {"error": "no tool call"}
        assert rollout.reward.total == 0.0


class TestSummarize:
    def test_reports_the_oracle_at_one(self):
        summary = summarize(CreditEnvironment().run(OraclePolicy()))
        assert summary["episodes"] == 10
        assert summary["mean_reward"] == 1.0
        assert summary["decision"] == 1.0

    def test_separates_the_baselines(self):
        environment = CreditEnvironment()
        oracle = summarize(environment.run(OraclePolicy()))
        degenerate = summarize(environment.run(AlwaysApprovePolicy()))
        assert oracle["mean_reward"] > degenerate["mean_reward"]

    def test_handles_no_episodes(self):
        summary = summarize([])
        assert summary["episodes"] == 0
        assert summary["mean_reward"] == 0.0


class FakeBlock:
    type = "tool_use"

    def __init__(self, payload: dict) -> None:
        self.input = payload


class FakeResponse:
    stop_reason = "tool_use"

    def __init__(self, content: list) -> None:
        self.content = content


class FakeMessages:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.messages = FakeMessages(response)


class TestClaudePolicy:
    """Request shape only — no network, no API key."""

    payload: ClassVar[dict] = {
        "decision": "REFER",
        "dti": 0.45,
        "key_factors": ["DTI at policy threshold"],
        "evidence": [{"feature": "monthly_debt", "value": 3200}],
    }

    def build(self, content: list) -> tuple[ClaudePolicy, FakeClient]:
        client = FakeClient(FakeResponse(content))
        return ClaudePolicy(client=client), client

    def test_returns_the_tool_input_unvalidated(self):
        policy, _ = self.build([FakeBlock(self.payload)])
        assert policy.decide(make_applicant()) == self.payload

    def test_forces_the_submission_tool(self):
        policy, client = self.build([FakeBlock(self.payload)])
        policy.decide(make_applicant())
        assert client.messages.kwargs["tool_choice"] == {
            "type": "tool",
            "name": "submit_decision",
        }

    def test_sends_the_policy_in_the_system_prompt(self):
        policy, client = self.build([FakeBlock(self.payload)])
        policy.decide(make_applicant())
        system = client.messages.kwargs["system"]
        assert "DTI is monthly_debt / monthly_income" in system
        assert "620" in system

    def test_uses_adaptive_thinking_on_opus_5(self):
        policy, client = self.build([FakeBlock(self.payload)])
        policy.decide(make_applicant())
        assert client.messages.kwargs["model"] == "claude-opus-5"
        assert client.messages.kwargs["thinking"] == {"type": "adaptive"}
        assert client.messages.kwargs["output_config"] == {"effort": "high"}

    def test_a_missing_tool_call_becomes_a_scoreable_failure(self):
        policy, _ = self.build([])
        assert "no tool call" in policy.decide(make_applicant())["error"]

    def test_the_prompt_leaks_no_derived_ratio(self):
        applicant = make_applicant()
        rendered = ClaudePolicy.render(applicant)
        assert "0.40" not in rendered  # the DTI it has to compute for itself
        assert "0.26" not in rendered  # the loan-to-income it has to compute too
        assert "8000.00" in rendered

    def test_the_prompt_leaks_no_ground_truth(self):
        rendered = ClaudePolicy.render(make_applicant())
        for label in ("APPROVE", "REFER", "DECLINE", "test"):
            assert label not in rendered


@pytest.mark.parametrize("policy", [OraclePolicy(), AlwaysApprovePolicy()])
def test_baselines_emit_schema_valid_output(policy):
    environment = CreditEnvironment()
    assert all(r.reward.schema_ok for r in environment.run(policy))
