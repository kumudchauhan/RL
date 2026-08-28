"""Composite verifier with weighted aggregation producing RewardSignal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ..schemas import InvoiceExtraction
from .base import VerificationResult, Verifier
from .detail_verifier import DetailVerifier
from .field_verifier import FieldVerifier
from .line_item_verifier import LineItemVerifier
from .numeric_verifier import NumericVerifier


@dataclass
class RewardSignal:
    """Complete reward signal for RLVR training."""

    overall_reward: float  # Weighted composite score [0, 1]
    component_rewards: dict[str, float] = field(default_factory=dict)
    details: dict[str, VerificationResult] = field(default_factory=dict)
    #: Components that had something to verify, and therefore contributed to the composite.
    applied_components: list[str] = field(default_factory=list)

    def to_harbor_reward(self) -> dict:
        """Serialize to Harbor reward format."""
        return {
            "reward": self.overall_reward,
            "components": self.component_rewards,
            "applied_components": list(self.applied_components),
            "metadata": {
                name: {
                    "score": r.score,
                    "max_score": r.max_score,
                    "normalized": r.normalized_score,
                    "applicable": r.applicable,
                    "details": r.details,
                }
                for name, r in self.details.items()
            },
        }


class CompositeVerifier:
    """Aggregates multiple verifiers into a weighted reward signal."""

    DEFAULT_WEIGHTS: ClassVar[dict[str, float]] = {
        "field_accuracy": 0.20,
        "numeric_accuracy": 0.30,
        "line_item_f1": 0.35,
        "detail_accuracy": 0.15,
    }

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.verifiers: list[Verifier] = [
            FieldVerifier(),
            NumericVerifier(),
            LineItemVerifier(),
            DetailVerifier(),
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

        # Components with nothing to verify are dropped and the remaining weights
        # renormalized, so an unannotated field neither rewards nor punishes the model.
        applied = [name for name, r in results.items() if r.applicable and name in self.weights]
        weight_total = sum(self.weights[name] for name in applied)

        if weight_total > 0:
            overall = (
                sum(component_rewards[name] * self.weights[name] for name in applied)
                / weight_total
            )
        else:
            overall = 0.0

        return RewardSignal(
            overall_reward=overall,
            component_rewards=component_rewards,
            details=results,
            applied_components=applied,
        )
