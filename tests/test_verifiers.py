"""Tests for verifiers."""

import pytest

from info_extract.schemas import InvoiceExtraction, LineItem
from info_extract.verifiers.composite import CompositeVerifier, RewardSignal
from info_extract.verifiers.field_verifier import FieldVerifier
from info_extract.verifiers.line_item_verifier import LineItemVerifier
from info_extract.verifiers.numeric_verifier import NumericVerifier


class TestFieldVerifier:
    def setup_method(self):
        self.verifier = FieldVerifier()

    def test_exact_match(self):
        pred = InvoiceExtraction(vendor="Amazon", order_id="123", total=50.0)
        truth = InvoiceExtraction(vendor="Amazon", order_id="123", total=50.0)
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score == 1.0

    def test_no_match(self):
        pred = InvoiceExtraction(vendor="eBay", order_id="999", total=50.0)
        truth = InvoiceExtraction(vendor="Amazon", order_id="123", total=50.0)
        result = self.verifier.verify(pred, truth)
        # currency still matches (both default "USD"), so 1/3 fields match
        assert result.details["vendor"]["score"] == 0.0
        assert result.details["order_id"]["score"] == 0.0

    def test_fuzzy_match_vendor(self):
        pred = InvoiceExtraction(vendor="Amazon.com", total=50.0)
        truth = InvoiceExtraction(vendor="Amazon.com Inc", total=50.0)
        result = self.verifier.verify(pred, truth)
        # "amazon.com" vs "amazon.com inc" - ratio should be above threshold
        assert result.normalized_score > 0.0

    def test_missing_prediction(self):
        pred = InvoiceExtraction(vendor="Amazon", total=50.0)
        truth = InvoiceExtraction(vendor="Amazon", order_id="123", total=50.0)
        result = self.verifier.verify(pred, truth)
        # vendor matches (1.0), order_id missing (0.0), currency matches (1.0)
        assert result.score == 2.0  # vendor + currency
        assert result.max_score == 3.0  # vendor + order_id + currency

    def test_null_ground_truth_skipped(self):
        pred = InvoiceExtraction(vendor="Amazon", order_id="123", total=50.0)
        truth = InvoiceExtraction(vendor="Amazon", total=50.0)
        result = self.verifier.verify(pred, truth)
        # order_id is None in ground truth, so not counted
        assert result.normalized_score == 1.0

    def test_case_insensitive(self):
        pred = InvoiceExtraction(vendor="AMAZON", total=50.0)
        truth = InvoiceExtraction(vendor="amazon", total=50.0)
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score == 1.0


class TestNumericVerifier:
    def setup_method(self):
        self.verifier = NumericVerifier(tolerance=0.01)

    def test_exact_match(self):
        pred = InvoiceExtraction(vendor="X", total=100.00, tax=8.50, subtotal=91.50)
        truth = InvoiceExtraction(vendor="X", total=100.00, tax=8.50, subtotal=91.50)
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score == 1.0

    def test_within_tolerance(self):
        pred = InvoiceExtraction(vendor="X", total=100.01)
        truth = InvoiceExtraction(vendor="X", total=100.00)
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score == 1.0

    def test_outside_tolerance(self):
        pred = InvoiceExtraction(vendor="X", total=110.00)
        truth = InvoiceExtraction(vendor="X", total=100.00)
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score < 1.0
        # 10% error -> score = 1.0 - 0.1 = 0.9
        assert result.normalized_score == pytest.approx(0.9, abs=0.01)

    def test_missing_prediction(self):
        pred = InvoiceExtraction(vendor="X", total=100.00)
        truth = InvoiceExtraction(vendor="X", total=100.00, tax=5.00)
        result = self.verifier.verify(pred, truth)
        # total matches (1.0), tax missing (0.0)
        assert result.score == 1.0
        assert result.max_score == 2.0

    def test_null_ground_truth_skipped(self):
        pred = InvoiceExtraction(vendor="X", total=100.00, tax=5.00)
        truth = InvoiceExtraction(vendor="X", total=100.00)
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score == 1.0


class TestLineItemVerifier:
    def setup_method(self):
        self.verifier = LineItemVerifier()

    def test_perfect_match(self):
        items = [LineItem(product_name="Widget", quantity=2, total_price=20.00)]
        pred = InvoiceExtraction(vendor="X", total=20.0, line_items=items)
        truth = InvoiceExtraction(vendor="X", total=20.0, line_items=items)
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score == 1.0

    def test_no_predictions(self):
        truth_items = [LineItem(product_name="Widget", quantity=1, total_price=10.00)]
        pred = InvoiceExtraction(vendor="X", total=10.0, line_items=[])
        truth = InvoiceExtraction(vendor="X", total=10.0, line_items=truth_items)
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score == 0.0

    def test_partial_match(self):
        pred_items = [
            LineItem(product_name="Widget", quantity=2, total_price=20.00),
            LineItem(product_name="Gadget", quantity=1, total_price=15.00),
        ]
        truth_items = [
            LineItem(product_name="Widget", quantity=2, total_price=20.00),
            LineItem(product_name="Gadget", quantity=1, total_price=15.00),
            LineItem(product_name="Thingamajig", quantity=1, total_price=5.00),
        ]
        pred = InvoiceExtraction(vendor="X", total=40.0, line_items=pred_items)
        truth = InvoiceExtraction(vendor="X", total=40.0, line_items=truth_items)
        result = self.verifier.verify(pred, truth)
        # Precision: 2/2 = 1.0, Recall: 2/3 ≈ 0.67, F1 ≈ 0.8
        assert 0.7 < result.normalized_score < 0.9

    def test_empty_ground_truth(self):
        pred = InvoiceExtraction(vendor="X", total=10.0, line_items=[])
        truth = InvoiceExtraction(vendor="X", total=10.0, line_items=[])
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score == 1.0

    def test_fuzzy_product_name(self):
        pred_items = [
            LineItem(product_name="Acme Satin Lipstick", quantity=1, total_price=16.00)
        ]
        truth_items = [
            LineItem(
                product_name="ACME COLLECTION Satin Hydrating Lipstick",
                quantity=1,
                total_price=16.00,
            )
        ]
        pred = InvoiceExtraction(vendor="X", total=16.0, line_items=pred_items)
        truth = InvoiceExtraction(vendor="X", total=16.0, line_items=truth_items)
        result = self.verifier.verify(pred, truth)
        # Should still get partial credit due to fuzzy matching
        assert result.normalized_score > 0.3


class TestCompositeVerifier:
    def setup_method(self):
        self.verifier = CompositeVerifier()

    def test_perfect_score(self):
        items = [LineItem(product_name="Widget", quantity=1, total_price=10.00)]
        invoice = InvoiceExtraction(
            vendor="TestCo",
            order_id="123",
            total=10.00,
            subtotal=10.00,
            line_items=items,
        )
        reward = self.verifier.compute_reward(invoice, invoice)
        assert isinstance(reward, RewardSignal)
        assert reward.overall_reward == pytest.approx(1.0, abs=0.01)

    def test_zero_score(self):
        pred = InvoiceExtraction(
            vendor="Wrong",
            order_id="999",
            total=999.99,
            subtotal=999.99,
            line_items=[LineItem(product_name="Nothing", quantity=99, total_price=999.99)],
        )
        truth = InvoiceExtraction(
            vendor="Correct",
            order_id="123",
            total=10.00,
            subtotal=10.00,
            line_items=[LineItem(product_name="Widget", quantity=1, total_price=10.00)],
        )
        reward = self.verifier.compute_reward(pred, truth)
        assert reward.overall_reward < 0.3

    def test_harbor_reward_format(self):
        invoice = InvoiceExtraction(vendor="Test", total=10.0)
        reward = self.verifier.compute_reward(invoice, invoice)
        harbor = reward.to_harbor_reward()
        assert "reward" in harbor
        assert "components" in harbor
        assert "metadata" in harbor
        assert isinstance(harbor["reward"], float)

    def test_custom_weights(self):
        weights = {"field_accuracy": 1.0, "numeric_accuracy": 0.0, "line_item_f1": 0.0}
        verifier = CompositeVerifier(weights=weights)
        pred = InvoiceExtraction(vendor="Amazon", total=999.0)
        truth = InvoiceExtraction(vendor="Amazon", total=10.0)
        reward = verifier.compute_reward(pred, truth)
        # Only field_accuracy matters, and vendor + currency match
        assert reward.overall_reward == pytest.approx(1.0, abs=0.01)
