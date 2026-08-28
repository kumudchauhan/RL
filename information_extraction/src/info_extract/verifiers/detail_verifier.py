"""Verifier for the per-item detail a receipt prints beyond name, quantity, and total.

``LineItemVerifier`` answers "did the model find the right items?". This one answers "for the
items it found, did it transcribe everything the document printed about them?" — identifiers
(UPC, SKU, ASIN, product number), the taxonomy the document itself states (department,
category), unit price and unit of measure, and per-item discounts/coupons. Order-level
discounts are scored here too, since they are the same kind of evidence.

Two deliberate asymmetries:

* **Only ground-truth-stated detail is scored.** A field the annotation leaves null is skipped,
  so annotating detail incrementally never looks like a regression.
* **Extras are reported, not punished.** When the prediction fills a field the annotation leaves
  empty, it is counted under ``unverifiable_extras`` instead of scored: on this corpus that
  usually means the annotation has not caught up, and guessing at which is which inside a
  reward function is exactly the kind of silent judgement a verifiable reward should avoid.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import ClassVar

from ..schemas import Discount, InvoiceExtraction, LineItem, ProductIdentifier
from .base import VerificationResult, Verifier
from .line_item_verifier import LineItemVerifier


def _normalize_id(value: str) -> str:
    """Identifiers compare ignoring case and the separators receipts sprinkle in."""
    return "".join(ch for ch in value.casefold() if ch not in " -_.\t")


class DetailVerifier(Verifier):
    """Scores line-item detail and coupon capture on matched line items."""

    IDENTIFIER_FIELDS: ClassVar[tuple[str, ...]] = ("sku", "upc", "asin", "product_number")
    TAXONOMY_FIELDS: ClassVar[tuple[str, ...]] = ("department", "category")
    EXACT_FIELDS: ClassVar[tuple[str, ...]] = ("quantity_unit",)

    def __init__(
        self,
        fuzzy_threshold: float = 0.85,
        price_tolerance: float = 0.01,
        matcher: LineItemVerifier | None = None,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.price_tolerance = price_tolerance
        self.matcher = matcher or LineItemVerifier()

    @property
    def name(self) -> str:
        return "detail_accuracy"

    # --- scalar comparisons -------------------------------------------------

    def _fuzzy_score(self, pred: str, truth: str) -> float:
        pred_str, true_str = pred.strip().casefold(), truth.strip().casefold()
        if pred_str == true_str:
            return 1.0
        ratio = SequenceMatcher(None, pred_str, true_str).ratio()
        return ratio if ratio >= self.fuzzy_threshold else 0.0

    def _amount_score(self, pred: float, truth: float) -> float:
        diff = abs(pred - truth)
        if diff <= self.price_tolerance + 1e-9:
            return 1.0
        if truth != 0:
            return max(0.0, 1.0 - diff / abs(truth))
        return 0.0

    # --- collection comparisons --------------------------------------------

    def _identifier_set_score(
        self, preds: list[ProductIdentifier], truths: list[ProductIdentifier]
    ) -> float:
        """Overlap of (label, value) pairs, penalising both misses and invented pairs."""
        pred_pairs = {(_normalize_id(p.label), _normalize_id(p.value)) for p in preds}
        true_pairs = {(_normalize_id(t.label), _normalize_id(t.value)) for t in truths}
        matched = len(pred_pairs & true_pairs)
        return matched / max(len(pred_pairs), len(true_pairs))

    def _discount_score(self, pred: Discount, truth: Discount) -> float:
        """Similarity of one discount to another, over the parts the truth states."""
        weighted: list[tuple[float, float]] = []
        if truth.amount is not None:
            weighted.append(
                (self._amount_score(pred.amount, truth.amount) if pred.amount is not None else 0.0, 0.4)
            )
        if truth.percentage is not None:
            weighted.append(
                (
                    1.0
                    if pred.percentage is not None
                    and abs(pred.percentage - truth.percentage) < 1e-9
                    else 0.0,
                    0.2,
                )
            )
        if truth.code is not None:
            weighted.append(
                (
                    1.0
                    if pred.code is not None and _normalize_id(pred.code) == _normalize_id(truth.code)
                    else 0.0,
                    0.3,
                )
            )
        if truth.source is not None:
            weighted.append(
                (self._fuzzy_score(pred.source, truth.source) if pred.source else 0.0, 0.2)
            )
        if truth.description is not None:
            weighted.append(
                (
                    self._fuzzy_score(pred.description, truth.description)
                    if pred.description
                    else 0.0,
                    0.2,
                )
            )
        if not weighted:
            return 1.0
        return sum(s * w for s, w in weighted) / sum(w for _, w in weighted)

    def _discount_list_score(self, preds: list[Discount], truths: list[Discount]) -> float:
        """Greedy best-pairing of predicted discounts to annotated ones."""
        candidates = sorted(
            (
                (self._discount_score(p, t), i, j)
                for i, p in enumerate(preds)
                for j, t in enumerate(truths)
            ),
            reverse=True,
        )
        used_pred: set[int] = set()
        used_true: set[int] = set()
        total = 0.0
        for score, i, j in candidates:
            if score <= 0 or i in used_pred or j in used_true:
                continue
            used_pred.add(i)
            used_true.add(j)
            total += score
        return total / max(len(preds), len(truths))

    # --- per-item scoring ---------------------------------------------------

    def _score_item(
        self,
        pred: LineItem,
        truth: LineItem,
        slots: list[tuple[str, float]],
        extras: dict[str, int],
    ) -> None:
        for field_name in (*self.IDENTIFIER_FIELDS, *self.EXACT_FIELDS):
            true_val = getattr(truth, field_name)
            pred_val = getattr(pred, field_name)
            if true_val is None:
                if pred_val is not None:
                    extras[field_name] = extras.get(field_name, 0) + 1
                continue
            matched = pred_val is not None and _normalize_id(pred_val) == _normalize_id(true_val)
            slots.append((field_name, 1.0 if matched else 0.0))

        for field_name in self.TAXONOMY_FIELDS:
            true_val = getattr(truth, field_name)
            pred_val = getattr(pred, field_name)
            if true_val is None:
                if pred_val is not None:
                    extras[field_name] = extras.get(field_name, 0) + 1
                continue
            slots.append((field_name, self._fuzzy_score(pred_val, true_val) if pred_val else 0.0))

        if truth.unit_price is None:
            if pred.unit_price is not None:
                extras["unit_price"] = extras.get("unit_price", 0) + 1
        else:
            slots.append(
                (
                    "unit_price",
                    self._amount_score(pred.unit_price, truth.unit_price)
                    if pred.unit_price is not None
                    else 0.0,
                )
            )

        if not truth.other_identifiers:
            if pred.other_identifiers:
                extras["other_identifiers"] = extras.get("other_identifiers", 0) + 1
        else:
            slots.append(
                (
                    "other_identifiers",
                    self._identifier_set_score(pred.other_identifiers, truth.other_identifiers),
                )
            )

        if not truth.discounts:
            if pred.discounts:
                extras["line_item_discounts"] = extras.get("line_item_discounts", 0) + 1
        else:
            slots.append(
                ("line_item_discounts", self._discount_list_score(pred.discounts, truth.discounts))
            )

    # --- verifier interface -------------------------------------------------

    def verify(
        self,
        prediction: InvoiceExtraction,
        ground_truth: InvoiceExtraction,
    ) -> VerificationResult:
        slots: list[tuple[str, float]] = []
        extras: dict[str, int] = {}

        pred_items, true_items = prediction.line_items, ground_truth.line_items
        matches = (
            self.matcher.greedy_match(pred_items, true_items)
            if pred_items and true_items
            else []
        )
        for pred_idx, truth_idx, _ in matches:
            self._score_item(pred_items[pred_idx], true_items[truth_idx], slots, extras)

        if not ground_truth.discounts:
            if prediction.discounts:
                extras["order_discounts"] = extras.get("order_discounts", 0) + 1
        else:
            slots.append(
                (
                    "order_discounts",
                    self._discount_list_score(prediction.discounts, ground_truth.discounts),
                )
            )

        per_field: dict[str, dict[str, float]] = {}
        for field_name, score in slots:
            entry = per_field.setdefault(field_name, {"scored": 0.0, "slots": 0.0})
            entry["scored"] += score
            entry["slots"] += 1

        total_score = sum(score for _, score in slots)
        details = {
            "scored_slots": len(slots),
            "per_field": {
                name: {
                    "slots": int(entry["slots"]),
                    "score": entry["scored"] / entry["slots"],
                }
                for name, entry in per_field.items()
            },
            "unverifiable_extras": extras,
            "matched_items": len(matches),
            "unmatched_ground_truth_items": max(0, len(true_items) - len(matches)),
        }
        if not slots:
            details["note"] = "ground truth states no line-item detail or coupons"

        return VerificationResult(
            verifier_name=self.name,
            score=total_score,
            max_score=float(len(slots)) if slots else 1.0,
            details=details,
            applicable=bool(slots),
        )
