"""Composite verifier with weighted aggregation producing RewardSignal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ..schemas import InvoiceExtraction
from .base import VerificationResult, Verifier
from .field_verifier import FieldVerifier
from .line_item_verifier import LineItemVerifier
from .numeric_verifier import NumericVerifier


@dataclass
class RewardSignal:
    """Complete reward signal for RLVR training."""

    overall_reward: float  # Weighted composite score [0, 1]
    component_rewards: dict[str, float] = field(default_factory=dict)
    details: dict[str, VerificationResult] = field(default_factory=dict)

    def to_harbor_reward(self) -> dict:
        """Serialize to Harbor reward format."""
        return {
            "reward": self.overall_reward,
            "components": self.component_rewards,
            "metadata": {
                name: {
                    "score": r.score,
                    "max_score": r.max_score,
                    "normalized": r.normalized_score,
                    "details": r.details,
                }
                for name, r in self.details.items()
            },
        }


class CompositeVerifier:
    """Aggregates multiple verifiers into a weighted reward signal."""

    DEFAULT_WEIGHTS: ClassVar[dict[str, float]] = {
        "field_accuracy": 0.25,
        "numeric_accuracy": 0.35,
        "line_item_f1": 0.40,
    }

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.verifiers: list[Verifier] = [
            FieldVerifier(),
            NumericVerifier(),
            LineItemVerifier(),
        ]

    def compute_reward(
        self,
        prediction: InvoiceExtraction,
        ground_truth: InvoiceExtraction,
    ) -> RewardSignal:
        results: dict[str, VerificationResult] = {}
        component_rewards: dict[str, float] = {}

        for verifier in self.verifiers:
            result = verifier.verify(prediction, ground_truth)
            results[verifier.name] = result
            component_rewards[verifier.name] = result.normalized_score

        overall = sum(
            component_rewards.get(name, 0.0) * weight
            for name, weight in self.weights.items()
        )

        return RewardSignal(
            overall_reward=overall,
            component_rewards=component_rewards,
            details=results,
        )
