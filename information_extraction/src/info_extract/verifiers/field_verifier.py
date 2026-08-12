"""Verifier for scalar fields (vendor, order_id, dates) using exact/fuzzy matching."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import ClassVar

from ..schemas import InvoiceExtraction
from .base import VerificationResult, Verifier


class FieldVerifier(Verifier):
    """Verifies scalar string fields with exact/fuzzy matching."""

    FIELDS: ClassVar[list[str]] = ["vendor", "order_id", "order_date", "delivery_date", "currency"]

    def __init__(self, fuzzy_threshold: float = 0.85):
        self.fuzzy_threshold = fuzzy_threshold

    @property
    def name(self) -> str:
        return "field_accuracy"

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

            pred_str = str(pred_val).strip().lower()
            true_str = str(true_val).strip().lower()

            if pred_str == true_str:
                score = 1.0
            else:
                ratio = SequenceMatcher(None, pred_str, true_str).ratio()
                score = ratio if ratio >= self.fuzzy_threshold else 0.0

            total_score += score
            details[field_name] = {
                "score": score,
                "pred": pred_val,
                "expected": true_val,
            }

        return VerificationResult(
            verifier_name=self.name,
            score=total_score,
            max_score=float(total_fields) if total_fields > 0 else 1.0,
            details=details,
        )
