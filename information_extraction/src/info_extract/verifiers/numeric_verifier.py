"""Verifier for monetary fields using tolerance-based comparison."""

from __future__ import annotations

from typing import ClassVar

from ..schemas import InvoiceExtraction
from .base import VerificationResult, Verifier


class NumericVerifier(Verifier):
    """Verifies numeric fields (subtotal, tax, shipping, total) with tolerance."""

    FIELDS: ClassVar[list[str]] = ["subtotal", "tax", "shipping_cost", "discount", "total"]

    def __init__(self, tolerance: float = 0.01):
        """tolerance: absolute dollar amount tolerance (default 1 cent)."""
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "numeric_accuracy"

    def verify(
        self,
        prediction: InvoiceExtraction,
        ground_truth: InvoiceExtraction,
    ) -> VerificationResult:
        details = {}
        total_score = 0.0
        total_fields = 0

        for field_name in self.FIELDS:
            true_val = getattr(ground_truth, field_name)
            if true_val is None:
                continue

            total_fields += 1
            pred_val = getattr(prediction, field_name)

            if pred_val is None:
                details[field_name] = {"score": 0.0, "pred": None, "expected": true_val}
                continue

            diff = abs(pred_val - true_val)
            if diff <= self.tolerance + 1e-9:
                score = 1.0
            elif true_val != 0:
                relative_error = diff / abs(true_val)
                score = max(0.0, 1.0 - relative_error)
            else:
                score = 0.0

            total_score += score
            details[field_name] = {
                "score": score,
                "pred": pred_val,
                "expected": true_val,
                "diff": diff,
            }

        return VerificationResult(
            verifier_name=self.name,
            score=total_score,
            max_score=float(total_fields) if total_fields > 0 else 1.0,
            details=details,
            applicable=total_fields > 0,
        )
