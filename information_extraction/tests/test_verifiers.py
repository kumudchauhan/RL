"""Tests for verifiers."""

import pytest

from info_extract.schemas import (
    Discount,
    Fee,
    InvoiceExtraction,
    LineItem,
    Payment,
    PaymentMethod,
    ProductIdentifier,
)
from info_extract.verifiers.composite import CompositeVerifier, RewardSignal
from info_extract.verifiers.detail_verifier import DetailVerifier
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
        assert result.normalized_score == 0.0
        assert result.details["vendor"]["score"] == 0.0
        assert result.details["order_id"]["score"] == 0.0

    def test_fuzzy_match_vendor(self):
        pred = InvoiceExtraction(vendor="Acme Cosmetics", total=50.0)
        truth = InvoiceExtraction(vendor="Acme Cosmetics Inc", total=50.0)
        result = self.verifier.verify(pred, truth)
        # ratio ~0.88, above the 0.85 threshold, so partial credit
        assert 0.0 < result.normalized_score < 1.0

    def test_missing_prediction(self):
        pred = InvoiceExtraction(vendor="Amazon", total=50.0)
        truth = InvoiceExtraction(vendor="Amazon", order_id="123", total=50.0)
        result = self.verifier.verify(pred, truth)
        # vendor matches (1.0), order_id missing (0.0)
        assert result.score == 1.0
        assert result.max_score == 2.0

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

    def test_service_provider_is_scored_separately_from_the_store(self):
        """A delivery order names both: the store sold it, the platform delivered it."""
        pred = InvoiceExtraction(
            vendor="Corner Market", service_provider="Instacart", total=50.0
        )
        truth = InvoiceExtraction(
            vendor="Corner Market", service_provider="Instacart", total=50.0
        )
        result = self.verifier.verify(pred, truth)
        assert result.max_score == 2.0
        assert result.normalized_score == 1.0

    def test_platform_mistaken_for_the_store_costs_both_fields(self):
        pred = InvoiceExtraction(vendor="Instacart", service_provider="Corner Market", total=50.0)
        truth = InvoiceExtraction(
            vendor="Corner Market", service_provider="Instacart", total=50.0
        )
        result = self.verifier.verify(pred, truth)
        assert result.details["vendor"]["score"] == 0.0
        assert result.details["service_provider"]["score"] == 0.0

    def test_unstated_service_provider_is_not_scored(self):
        """A store that billed directly has no platform — inventing one is not punished here."""
        pred = InvoiceExtraction(vendor="X", service_provider="Instacart", total=50.0)
        truth = InvoiceExtraction(vendor="X", total=50.0)
        result = self.verifier.verify(pred, truth)
        assert "service_provider" not in result.details
        assert result.normalized_score == 1.0

    def test_nothing_to_verify_is_not_applicable(self):
        blank = InvoiceExtraction.model_construct(vendor=None, total=None)
        result = self.verifier.verify(blank, blank)
        assert result.applicable is False


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

    def test_each_fee_is_its_own_field(self):
        """Delivery, service, and tip are separate charges, so each is separately scored."""
        truth = InvoiceExtraction(
            vendor="X", total=48.20, delivery_fee=3.99, service_fee=2.01, tip=2.00
        )
        pred = InvoiceExtraction(
            vendor="X", total=48.20, delivery_fee=3.99, service_fee=2.01, tip=0.00
        )
        result = self.verifier.verify(pred, truth)
        assert result.max_score == 4.0
        assert result.details["delivery_fee"]["score"] == 1.0
        assert result.details["tip"]["score"] == 0.0

    def test_fees_lumped_into_one_field_scores_badly(self):
        """A model that adds the fees together gets neither field right."""
        truth = InvoiceExtraction(vendor="X", total=10.0, delivery_fee=3.99, service_fee=2.01)
        pred = InvoiceExtraction(vendor="X", total=10.0, delivery_fee=6.00, service_fee=0.0)
        result = self.verifier.verify(pred, truth)
        assert result.details["delivery_fee"]["score"] < 0.6
        assert result.details["service_fee"]["score"] == 0.0

    def test_billed_and_paid_are_scored_independently(self):
        """Split tender: the order was billed 100 but only 75 hit the card."""
        truth = InvoiceExtraction(vendor="X", total=100.00, amount_paid=75.00)
        pred = InvoiceExtraction(vendor="X", total=100.00, amount_paid=100.00)
        result = self.verifier.verify(pred, truth)
        assert result.details["total"]["score"] == 1.0
        assert result.details["amount_paid"]["score"] < 1.0

    def test_installment_balance_is_scored(self):
        truth = InvoiceExtraction(
            vendor="X", total=299.94, amount_paid=49.99, amount_due=249.95
        )
        result = self.verifier.verify(truth, truth)
        assert result.max_score == 3.0
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

    def test_unstated_quantity_is_not_a_miss(self):
        """A receipt line without a printed quantity must not be scored as a wrong quantity."""
        truth_items = [LineItem(product_name="Bananas", total_price=1.99)]
        pred_items = [LineItem(product_name="Bananas", total_price=1.99)]
        pred = InvoiceExtraction(vendor="X", total=1.99, line_items=pred_items)
        truth = InvoiceExtraction(vendor="X", total=1.99, line_items=truth_items)
        assert self.verifier.verify(pred, truth).normalized_score == pytest.approx(1.0)

    def test_fractional_quantity_matches(self):
        items = [LineItem(product_name="Bananas", quantity=1.24, total_price=1.99)]
        pred = InvoiceExtraction(vendor="X", total=1.99, line_items=items)
        truth = InvoiceExtraction(vendor="X", total=1.99, line_items=items)
        assert self.verifier.verify(pred, truth).normalized_score == pytest.approx(1.0)

    def test_unstated_line_total_falls_back_to_name_and_quantity(self):
        truth_items = [LineItem(product_name="Widget", quantity=2)]
        pred_items = [LineItem(product_name="Widget", quantity=2, total_price=99.0)]
        pred = InvoiceExtraction(vendor="X", total=99.0, line_items=pred_items)
        truth = InvoiceExtraction(vendor="X", total=99.0, line_items=truth_items)
        assert self.verifier.verify(pred, truth).normalized_score == pytest.approx(1.0)

    def test_empty_ground_truth_is_not_applicable(self):
        pred = InvoiceExtraction(vendor="X", total=10.0)
        truth = InvoiceExtraction(vendor="X", total=10.0)
        assert self.verifier.verify(pred, truth).applicable is False


class TestDetailVerifier:
    def setup_method(self):
        self.verifier = DetailVerifier()

    @staticmethod
    def _invoice(item: LineItem, **kwargs) -> InvoiceExtraction:
        return InvoiceExtraction(vendor="X", total=10.0, line_items=[item], **kwargs)

    def test_identifiers_scored_when_annotated(self):
        truth_item = LineItem(
            product_name="Sparkling Water",
            total_price=10.0,
            upc="012345678905",
            sku="SW-12",
        )
        pred = self._invoice(
            LineItem(
                product_name="Sparkling Water",
                total_price=10.0,
                upc="012345678905",
                sku="SW-12",
            )
        )
        result = self.verifier.verify(pred, self._invoice(truth_item))
        assert result.applicable is True
        assert result.max_score == 2.0
        assert result.normalized_score == 1.0

    def test_wrong_identifier_scores_zero(self):
        truth = self._invoice(
            LineItem(product_name="Water", total_price=10.0, upc="012345678905")
        )
        pred = self._invoice(LineItem(product_name="Water", total_price=10.0, upc="999999999999"))
        result = self.verifier.verify(pred, truth)
        assert result.details["per_field"]["upc"]["score"] == 0.0

    def test_identifier_separators_are_ignored(self):
        truth = self._invoice(LineItem(product_name="Water", total_price=10.0, sku="SW 12-A"))
        pred = self._invoice(LineItem(product_name="Water", total_price=10.0, sku="sw12a"))
        assert self.verifier.verify(pred, truth).normalized_score == 1.0

    def test_missing_annotated_identifier_scores_zero(self):
        truth = self._invoice(LineItem(product_name="Water", total_price=10.0, asin="B00ABCDEFG"))
        pred = self._invoice(LineItem(product_name="Water", total_price=10.0))
        assert self.verifier.verify(pred, truth).normalized_score == 0.0

    def test_taxonomy_is_fuzzy_matched(self):
        truth = self._invoice(
            LineItem(
                product_name="Water", total_price=10.0, department="GROCERY", category="Beverages"
            )
        )
        pred = self._invoice(
            LineItem(
                product_name="Water", total_price=10.0, department="grocery", category="Beverage"
            )
        )
        result = self.verifier.verify(pred, truth)
        assert result.details["per_field"]["department"]["score"] == 1.0
        assert result.details["per_field"]["category"]["score"] > 0.85

    def test_extras_are_reported_not_punished(self):
        """Detail the annotation does not cover is counted, not scored."""
        truth = self._invoice(LineItem(product_name="Water", total_price=10.0, sku="SW-12"))
        pred = self._invoice(
            LineItem(
                product_name="Water",
                total_price=10.0,
                sku="SW-12",
                upc="012345678905",
                category="Beverages",
            )
        )
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score == 1.0
        assert result.details["unverifiable_extras"] == {"upc": 1, "category": 1}

    def test_not_applicable_without_annotated_detail(self):
        truth = self._invoice(LineItem(product_name="Water", total_price=10.0))
        pred = self._invoice(LineItem(product_name="Water", total_price=10.0, upc="012345678905"))
        result = self.verifier.verify(pred, truth)
        assert result.applicable is False
        assert result.details["scored_slots"] == 0

    def test_other_identifiers_overlap(self):
        truth = self._invoice(
            LineItem(
                product_name="Bananas",
                total_price=1.99,
                other_identifiers=[
                    ProductIdentifier(label="PLU", value="4011"),
                    ProductIdentifier(label="Lot", value="A7"),
                ],
            )
        )
        pred = self._invoice(
            LineItem(
                product_name="Bananas",
                total_price=1.99,
                other_identifiers=[ProductIdentifier(label="PLU", value="4011")],
            )
        )
        result = self.verifier.verify(pred, truth)
        assert result.details["per_field"]["other_identifiers"]["score"] == pytest.approx(0.5)

    def test_unit_price_uses_tolerance(self):
        truth = self._invoice(LineItem(product_name="Water", total_price=10.0, unit_price=5.00))
        exact = self._invoice(LineItem(product_name="Water", total_price=10.0, unit_price=5.00))
        off = self._invoice(LineItem(product_name="Water", total_price=10.0, unit_price=6.00))
        assert self.verifier.verify(exact, truth).normalized_score == 1.0
        assert self.verifier.verify(off, truth).normalized_score < 1.0

    def test_line_item_coupon_matched_on_code_and_amount(self):
        coupon = Discount(description="GROUPON $2 OFF", code="GRPN2", source="Groupon", amount=2.0)
        truth = self._invoice(
            LineItem(product_name="Facial", total_price=10.0, discounts=[coupon])
        )
        pred = self._invoice(
            LineItem(product_name="Facial", total_price=10.0, discounts=[coupon.model_copy()])
        )
        assert self.verifier.verify(pred, truth).normalized_score == 1.0

    def test_missed_coupon_costs_score(self):
        truth = self._invoice(
            LineItem(
                product_name="Facial",
                total_price=10.0,
                discounts=[Discount(code="GRPN2", amount=2.0)],
            )
        )
        pred = self._invoice(LineItem(product_name="Facial", total_price=10.0))
        assert self.verifier.verify(pred, truth).normalized_score == 0.0

    def test_invented_extra_coupon_dilutes_the_score(self):
        truth = self._invoice(
            LineItem(
                product_name="Facial",
                total_price=10.0,
                discounts=[Discount(code="GRPN2", amount=2.0)],
            )
        )
        pred = self._invoice(
            LineItem(
                product_name="Facial",
                total_price=10.0,
                discounts=[
                    Discount(code="GRPN2", amount=2.0),
                    Discount(code="MADEUP", amount=5.0),
                ],
            )
        )
        assert self.verifier.verify(pred, truth).normalized_score == pytest.approx(0.5)

    def test_order_level_discounts_are_scored(self):
        coupon = Discount(description="10% off order", percentage=10.0)
        truth = InvoiceExtraction(vendor="X", total=10.0, discounts=[coupon])
        pred = InvoiceExtraction(vendor="X", total=10.0, discounts=[coupon.model_copy()])
        result = self.verifier.verify(pred, truth)
        assert result.details["per_field"]["order_discounts"]["score"] == 1.0

    def test_fee_lines_are_scored_on_label_and_amount(self):
        fees = [Fee(label="Bag Fee", amount=0.10), Fee(label="Bottle Deposit", amount=0.25)]
        truth = InvoiceExtraction(vendor="X", total=10.0, fees=fees)
        pred = InvoiceExtraction(vendor="X", total=10.0, fees=[f.model_copy() for f in fees])
        result = self.verifier.verify(pred, truth)
        assert result.details["per_field"]["fees"]["score"] == 1.0

    def test_fee_amount_on_the_wrong_label_is_not_a_match(self):
        truth = InvoiceExtraction(vendor="X", total=10.0, fees=[Fee(label="Bag Fee", amount=0.10)])
        pred = InvoiceExtraction(
            vendor="X", total=10.0, fees=[Fee(label="Bottle Deposit", amount=0.10)]
        )
        assert self.verifier.verify(pred, truth).normalized_score == 0.0

    def test_missed_fee_line_costs_score(self):
        truth = InvoiceExtraction(
            vendor="X",
            total=10.0,
            fees=[Fee(label="Bag Fee", amount=0.10), Fee(label="Small Order Fee", amount=1.99)],
        )
        pred = InvoiceExtraction(vendor="X", total=10.0, fees=[Fee(label="Bag Fee", amount=0.10)])
        assert self.verifier.verify(pred, truth).normalized_score == pytest.approx(0.5)

    def test_split_tender_scored_per_payment(self):
        payments = [
            Payment(method=PaymentMethod.GIFT_CARD, amount=25.00),
            Payment(method=PaymentMethod.CREDIT_CARD, card_type="Visa", amount=75.00),
        ]
        truth = InvoiceExtraction(vendor="X", total=100.0, payments=payments)
        pred = InvoiceExtraction(
            vendor="X", total=100.0, payments=[p.model_copy() for p in payments]
        )
        result = self.verifier.verify(pred, truth)
        assert result.details["per_field"]["payments"]["score"] == 1.0

    def test_collapsing_a_split_payment_into_one_costs_score(self):
        truth = InvoiceExtraction(
            vendor="X",
            total=100.0,
            payments=[
                Payment(method=PaymentMethod.GIFT_CARD, amount=25.00),
                Payment(method=PaymentMethod.CREDIT_CARD, amount=75.00),
            ],
        )
        pred = InvoiceExtraction(
            vendor="X",
            total=100.0,
            payments=[Payment(method=PaymentMethod.CREDIT_CARD, amount=100.00)],
        )
        assert self.verifier.verify(pred, truth).normalized_score < 0.5

    def test_wrong_payment_method_scores_zero(self):
        truth = InvoiceExtraction(
            vendor="X", total=10.0, payments=[Payment(method=PaymentMethod.CREDIT_CARD)]
        )
        pred = InvoiceExtraction(
            vendor="X", total=10.0, payments=[Payment(method=PaymentMethod.DEBIT_CARD)]
        )
        assert self.verifier.verify(pred, truth).normalized_score == 0.0

    def test_installment_plan_is_scored(self):
        plan = Payment(
            method=PaymentMethod.CREDIT_CARD,
            card_type="Visa",
            amount=49.99,
            installment_count=6,
            installment_amount=49.99,
        )
        truth = InvoiceExtraction(vendor="X", total=299.94, payments=[plan])
        matching = InvoiceExtraction(vendor="X", total=299.94, payments=[plan.model_copy()])
        assert self.verifier.verify(matching, truth).normalized_score == 1.0

        missed_plan = InvoiceExtraction(
            vendor="X",
            total=299.94,
            payments=[Payment(method=PaymentMethod.CREDIT_CARD, card_type="Visa", amount=49.99)],
        )
        assert self.verifier.verify(missed_plan, truth).normalized_score < 1.0

    def test_unannotated_payments_are_reported_not_punished(self):
        truth = InvoiceExtraction(vendor="X", total=10.0, discounts=[Discount(amount=1.0)])
        pred = InvoiceExtraction(
            vendor="X",
            total=10.0,
            discounts=[Discount(amount=1.0)],
            payments=[Payment(method=PaymentMethod.CASH)],
            fees=[Fee(label="Bag Fee", amount=0.10)],
        )
        result = self.verifier.verify(pred, truth)
        assert result.normalized_score == 1.0
        assert result.details["unverifiable_extras"] == {"payments": 1, "fees": 1}

    def test_detail_of_unmatched_items_is_not_counted(self):
        """An item the model never found is the line-item verifier's penalty, not this one's."""
        truth = self._invoice(LineItem(product_name="Water", total_price=10.0, sku="SW-12"))
        pred = InvoiceExtraction(
            vendor="X",
            total=10.0,
            line_items=[LineItem(product_name="Totally Different Thing", total_price=99.0)],
        )
        result = self.verifier.verify(pred, truth)
        assert result.applicable is False
        assert result.details["unmatched_ground_truth_items"] == 1


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
        # Only field_accuracy carries weight, and vendor matches
        assert reward.overall_reward == pytest.approx(1.0, abs=0.01)

    def test_default_weights_sum_to_one(self):
        assert sum(CompositeVerifier.DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_every_weight_has_a_verifier(self):
        names = {v.name for v in CompositeVerifier().verifiers}
        assert set(CompositeVerifier.DEFAULT_WEIGHTS) == names

    def test_inapplicable_components_are_dropped_and_weights_renormalized(self):
        """With no annotated line items or detail, the score is field+numeric only."""
        pred = InvoiceExtraction(vendor="Wrong", total=10.0)
        truth = InvoiceExtraction(vendor="Right", total=10.0)
        reward = self.verifier.compute_reward(pred, truth)

        assert reward.applied_components == ["field_accuracy", "numeric_accuracy"]
        weights = CompositeVerifier.DEFAULT_WEIGHTS
        expected = (0.0 * weights["field_accuracy"] + 1.0 * weights["numeric_accuracy"]) / (
            weights["field_accuracy"] + weights["numeric_accuracy"]
        )
        assert reward.overall_reward == pytest.approx(expected)

    def test_unannotated_detail_does_not_dilute_a_perfect_score(self):
        """Annotating detail incrementally must never look like a regression."""
        item = LineItem(product_name="Widget", quantity=1, total_price=10.00)
        invoice = InvoiceExtraction(vendor="TestCo", total=10.00, line_items=[item])
        reward = self.verifier.compute_reward(invoice, invoice)
        assert "detail_accuracy" not in reward.applied_components
        assert reward.overall_reward == pytest.approx(1.0)

    def test_detail_counts_once_annotated(self):
        truth = InvoiceExtraction(
            vendor="TestCo",
            total=10.00,
            line_items=[LineItem(product_name="Widget", total_price=10.00, upc="012345678905")],
        )
        pred = InvoiceExtraction(
            vendor="TestCo",
            total=10.00,
            line_items=[LineItem(product_name="Widget", total_price=10.00)],
        )
        reward = self.verifier.compute_reward(pred, truth)
        assert "detail_accuracy" in reward.applied_components
        assert reward.component_rewards["detail_accuracy"] == 0.0
        assert reward.overall_reward < 1.0

    def test_harbor_reward_reports_applicability(self):
        invoice = InvoiceExtraction(vendor="Test", total=10.0)
        harbor = self.verifier.compute_reward(invoice, invoice).to_harbor_reward()
        assert harbor["applied_components"] == ["field_accuracy", "numeric_accuracy"]
        assert harbor["metadata"]["line_item_f1"]["applicable"] is False
