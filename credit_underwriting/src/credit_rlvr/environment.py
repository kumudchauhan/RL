"""RLVR episode loop.

Draws an application from the generator, asks the policy for a decision, runs the
verifiers, and returns the reward together with the rollout record that a future
training loop would consume.
"""

from __future__ import annotations

import argparse

from .generator import load_applicants
from .policy import AlwaysApprovePolicy, OraclePolicy, Policy
from .reward import WEIGHTS, score_episode
from .schemas import Applicant, Rollout

BASELINES: dict[str, type] = {
    "oracle": OraclePolicy,
    "always-approve": AlwaysApprovePolicy,
}


class CreditEnvironment:
    """The applicant set, plus the generate → decide → verify → reward loop."""

    def __init__(self, applicants: list[Applicant] | None = None) -> None:
        self.applicants = applicants if applicants is not None else load_applicants()

    def run_episode(self, applicant: Applicant, policy: Policy) -> Rollout:
        """One application, one decision, one reward."""
        raw_output = policy.decide(applicant)
        return Rollout(
            applicant=applicant,
            raw_output=raw_output,
            reward=score_episode(applicant, raw_output),
        )

    def run(self, policy: Policy) -> list[Rollout]:
        """Every applicant, in file order."""
        return [self.run_episode(applicant, policy) for applicant in self.applicants]


def summarize(rollouts: list[Rollout]) -> dict[str, float]:
    """Mean total reward, plus a mean per component. Empty input gives zeroes."""
    if not rollouts:
        return {"episodes": 0, "mean_reward": 0.0, **{name: 0.0 for name in WEIGHTS}}

    summary: dict[str, float] = {
        "episodes": len(rollouts),
        "mean_reward": sum(r.reward.total for r in rollouts) / len(rollouts),
    }
    for name in WEIGHTS:
        scored = [r.reward for r in rollouts if r.reward.schema_ok]
        summary[name] = (
            sum(reward.component(name).score for reward in scored) / len(scored) if scored else 0.0
        )
    return summary


def _report(name: str, rollouts: list[Rollout]) -> None:
    print(f"\n{name}")
    print(f"  {'applicant':10}{'scenario':30}{'decision':11}{'reward':>7}")
    for rollout in rollouts:
        decision = rollout.raw_output.get("decision", "-") if rollout.reward.schema_ok else "INVALID"
        print(
            f"  {rollout.applicant.id:10}{rollout.applicant.scenario:30}"
            f"{decision:11}{rollout.reward.total:7.3f}"
        )
    summary = summarize(rollouts)
    parts = " · ".join(f"{key} {summary[key]:.3f}" for key in WEIGHTS)
    print(f"  mean reward {summary['mean_reward']:.3f}   ({parts})")


def main() -> None:
    """Score the offline baselines on the frozen set. No API key required."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=[*BASELINES, "all"],
        default="all",
        help="which baseline to score (default: all)",
    )
    args = parser.parse_args()

    environment = CreditEnvironment()
    chosen = BASELINES if args.policy == "all" else {args.policy: BASELINES[args.policy]}
    for name, factory in chosen.items():
        _report(name, environment.run(factory()))


if __name__ == "__main__":
    main()
