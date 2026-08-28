"""Verifier for scalar fields (vendor, order_id, dates, payment) using exact/fuzzy matching."""

from __future__ import annotations

from difflib import SequenceMatcher
from enum import Enum
from typing import ClassVar

from ..schemas import InvoiceExtraction
from .base import VerificationResult, Verifier


def resolve_field(obj: object, path: str) -> object | None:
    """Resolve a dotted field path, returning None if any step is missing.

    Enum values are unwrapped to their string value so ``payment.method`` compares as
    ``"credit_card"`` rather than ``"PaymentMethod.CREDIT_CARD"``.
    """
    value: object | None = obj
    for part in path.split("."):
        if value is None:
            return None
        value = getattr(value, part, None)
    return value.value if isinstance(value, Enum) else value


class FieldVerifier(Verifier):
    """Verifies scalar string fields with exact/fuzzy matching."""

    FIELDS: ClassVar[list[str]] = [
        "vendor",
        "order_id",
        "order_date",
        "delivery_date",
        "currency",
        "payment.method",
        "payment.card_type",
    ]

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
            true_val = resolve_field(ground_truth, field_name)
            if true_val is None:
                continue

            total_fields += 1
            pred_val = resolve_field(prediction, field_name)

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
            applicable=total_fields > 0,
        )
