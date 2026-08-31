"""The underwriting policy under evaluation.

Wraps Claude tool-use to turn an application into a decision, and exposes the same
interface for baseline policies so they can be scored against each other on
identical episodes.

Every policy returns a **raw dict**, deliberately unvalidated. The environment
validates it, so `verify_schema` has real work to do and a policy that emits
nonsense is scored rather than raising.

Three implementations ship here:

  ClaudePolicy        the real thing
  OraclePolicy        applies the rulebook exactly — the reward's ceiling
  AlwaysApprovePolicy ignores the application — the reward's gameability probe

The two baselines need no API key, which is what makes the reward testable.
"""

from __future__ import annotations

from typing import Protocol

from . import credit_policy
from .credit_policy import POLICY_TEXT
from .schemas import Applicant, UnderwritingDecision

MODEL = "claude-opus-5"

SYSTEM_PROMPT = f"""\
You are a credit underwriter. You apply a written credit policy to a loan
application and submit a decision. You do not exercise discretion beyond the
policy, and you do not use knowledge about lending that the policy does not state.

The credit policy:

{POLICY_TEXT}
Submit your answer with the submit_decision tool:

- decision: APPROVE, REFER or DECLINE, per the policy above.
- dti: the ratio you computed, to two decimals.
- key_factors: one to four short phrases naming what drove the decision. Name the
  rule that actually decided it, not every fact you looked at.
- evidence: the application figures the deciding rule rests on, each as a feature
  name and the value shown in the application. Cite what the rule depends on and
  nothing more — a citation of an unrelated figure counts against you, as does
  omitting one the rule needs.
"""

USER_PROMPT_TEMPLATE = """\
Loan application {id}

monthly_income:      {monthly_income:.2f}
monthly_debt:        {monthly_debt:.2f}
credit_utilization:  {credit_utilization:.2f}
credit_score:        {credit_score}
employment_years:    {employment_years}
requested_loan:      {requested_loan:.2f}

Apply the policy and submit your decision.
"""


class Policy(Protocol):
    """Anything that can underwrite an application."""

    name: str

    def decide(self, applicant: Applicant) -> dict:
        """Return a candidate decision as a raw, unvalidated dict."""
        ...


class OraclePolicy:
    """Applies the rulebook exactly, and cites exactly what it rests on.

    The reward's ceiling: this should score 1.0 on every applicant. When it does
    not, the verifiers disagree with the rulebook and the bug is in the reward, not
    in a policy.
    """

    name = "oracle"

    def decide(self, applicant: Applicant) -> dict:
        ruling = credit_policy.evaluate(applicant)
        return {
            "decision": ruling.decision.value,
            "dti": applicant.dti,
            "key_factors": [ruling.reason],
            "evidence": [
                {"feature": feature, "value": applicant.value_of(feature)}
                for feature in ruling.required_features
            ],
        }


class AlwaysApprovePolicy:
    """Approves everything without reading the application.

    The gameability probe: a degenerate policy that still emits valid JSON. Its
    mean reward is the floor a real policy has to clear to have learned anything.
    """

    name = "always-approve"

    def decide(self, applicant: Applicant) -> dict:
        return {
            "decision": "APPROVE",
            "dti": 0.0,
            "key_factors": ["applicant looks creditworthy"],
            "evidence": [{"feature": "credit_score", "value": applicant.credit_score}],
        }


class ClaudePolicy:
    """Underwrites with Claude, via a forced tool call for the decision schema."""

    name = "claude"

    def __init__(
        self,
        model: str = MODEL,
        effort: str = "high",
        client: object | None = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self._tool = {
            "name": "submit_decision",
            "description": "Submit the underwriting decision for this application",
            "input_schema": UnderwritingDecision.model_json_schema(),
        }

    def decide(self, applicant: Applicant) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": self.render(applicant)}],
            tools=[self._tool],
            tool_choice={"type": "tool", "name": "submit_decision"},
        )
        block = next((b for b in response.content if b.type == "tool_use"), None)
        if block is None:
            # No submission is a failed episode, not an exception: let the schema
            # verifier score it 0 and keep the rollout.
            return {"error": f"no tool call (stop_reason={response.stop_reason})"}
        return dict(block.input)

    @staticmethod
    def render(applicant: Applicant) -> str:
        """Format the application. Features only — no derived ratios, no hints."""
        return USER_PROMPT_TEMPLATE.format(
            id=applicant.id,
            monthly_income=applicant.monthly_income,
            monthly_debt=applicant.monthly_debt,
            credit_utilization=applicant.credit_utilization,
            credit_score=applicant.credit_score,
            employment_years=applicant.employment_years,
            requested_loan=applicant.requested_loan,
        )
