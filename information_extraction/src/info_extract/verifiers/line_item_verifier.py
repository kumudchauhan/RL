"""Verifier for line items using greedy bipartite matching with F1 scoring."""

from __future__ import annotations

from difflib import SequenceMatcher

from ..schemas import InvoiceExtraction, LineItem
from .base import VerificationResult, Verifier


class LineItemVerifier(Verifier):
    """Verifies line items using greedy matching with weighted F1.

    Only the fields every receipt prints are matched here — name, quantity, line total.
    The richer per-item detail (identifiers, taxonomy, unit price, discounts) is scored
    separately by ``DetailVerifier``, which reuses this matching.
    """

    def __init__(self, name_threshold: float = 0.8, price_tolerance: float = 0.01):
        self.name_threshold = name_threshold
        self.price_tolerance = price_tolerance

    @property
    def name(self) -> str:
        return "line_item_f1"

    def match_score(self, pred: LineItem, truth: LineItem) -> float:
        """Score a single predicted item against a ground truth item.

        Name similarity always counts. Quantity and line total count only when the ground
        truth states them; their weight is redistributed otherwise, so an item the receipt
        prints without a quantity is not scored as a miss.
        """
        # Product name similarity (40% weight)
        name_sim = SequenceMatcher(
            None,
            pred.product_name.lower().strip(),
            truth.product_name.lower().strip(),
        ).ratio()
        weighted: list[tuple[float, float]] = [(name_sim, 0.4)]

        # Quantity match (20% weight)
        if truth.quantity is not None:
            qty_score = (
                1.0
                if pred.quantity is not None and abs(pred.quantity - truth.quantity) < 1e-9
                else 0.0
            )
            weighted.append((qty_score, 0.2))

        # Price match (40% weight)
        if truth.total_price is not None:
            if pred.total_price is None:
                price_score = 0.0
            else:
                price_diff = abs(pred.total_price - truth.total_price)
                if price_diff <= self.price_tolerance:
                    price_score = 1.0
                elif truth.total_price != 0:
                    price_score = max(0.0, 1.0 - price_diff / abs(truth.total_price))
                else:
                    price_score = 0.0
            weighted.append((price_score, 0.4))

        total_weight = sum(w for _, w in weighted)
        return sum(s * w for s, w in weighted) / total_weight

    def greedy_match(
        self, predictions: list[LineItem], truths: list[LineItem]
    ) -> list[tuple[int, int, float]]:
        """Greedy bipartite matching of predicted items to ground truth."""
        scores = []
        for i, pred in enumerate(predictions):
            for j, truth in enumerate(truths):
                s = self.match_score(pred, truth)
                if s > 0.3:
                    scores.append((s, i, j))

        scores.sort(reverse=True)
        matched_preds: set[int] = set()
        matched_truths: set[int] = set()
        matches = []

        for score, i, j in scores:
            if i not in matched_preds and j not in matched_truths:
                matches.append((i, j, score))
                matched_preds.add(i)
                matched_truths.add(j)

        return matches

    def verify(
        self,
        prediction: InvoiceExtraction,
        ground_truth: InvoiceExtraction,
    ) -> VerificationResult:
        pred_items = prediction.line_items
        true_items = ground_truth.line_items

        if not true_items:
            return VerificationResult(
                verifier_name=self.name,
                score=1.0 if not pred_items else 0.0,
                max_score=1.0,
                details={"note": "no ground truth line items"},
                applicable=False,
            )

        if not pred_items:
            return VerificationResult(
                verifier_name=self.name,
                score=0.0,
                max_score=1.0,
                details={
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "num_predicted": 0,
                    "num_ground_truth": len(true_items),
                },
            )

        matches = self.greedy_match(pred_items, true_items)

        precision = sum(s for _, _, s in matches) / len(pred_items)
        recall = sum(s for _, _, s in matches) / len(true_items)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return VerificationResult(
            verifier_name=self.name,
            score=f1,
            max_score=1.0,
            details={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "num_predicted": len(pred_items),
                "num_ground_truth": len(true_items),
                "num_matched": len(matches),
                "matches": [
                    {"pred_idx": i, "truth_idx": j, "score": s} for i, j, s in matches
                ],
            },
        )
