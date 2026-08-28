"""Tests for Pydantic schemas."""

import pytest
from pydantic import BaseModel, ValidationError

from info_extract.schemas import (
    FORBIDDEN_FIELD_FRAGMENTS,
    Discount,
    InvoiceExtraction,
    LineItem,
    PaymentInfo,
    PaymentMethod,
    ProductIdentifier,
    assert_pii_free_schema,
)

MODELS: list[type[BaseModel]] = [
    InvoiceExtraction,
    LineItem,
    ProductIdentifier,
    Discount,
    PaymentInfo,
]


class TestLineItem:
    def test_minimal_states_nothing_it_was_not_given(self):
        item = LineItem(product_name="Widget")
        assert item.quantity is None
        assert item.quantity_unit is None
        assert item.unit_price is None
        assert item.total_price is None
        assert item.sku is None
        assert item.upc is None
        assert item.asin is None
        assert item.product_number is None
        assert item.department is None
        assert item.category is None
        assert item.other_identifiers == []
        assert item.discounts == []

    def test_full(self):
        item = LineItem(
            product_name="Sparkling Water 12pk",
            quantity=2,
            quantity_unit="ea",
            unit_price=5.49,
            total_price=10.98,
            sku="SW-12",
            upc="012345678905",
            asin="B00ABCDEFG",
            product_number="PN-9931",
            other_identifiers=[ProductIdentifier(label="PLU", value="4011")],
            department="GROCERY",
            category="Beverages",
            discounts=[
                Discount(
                    description="GROUPON $2 OFF",
                    code="GRPN2",
                    source="Groupon",
                    amount=2.00,
                )
            ],
        )
        assert item.upc == "012345678905"
        assert item.other_identifiers[0].label == "PLU"
        assert item.discounts[0].amount == 2.00

    def test_fractional_quantity(self):
        """Weighed goods print quantities like '1.24 lb'."""
        item = LineItem(product_name="Bananas", quantity=1.24, quantity_unit="lb")
        assert item.quantity == pytest.approx(1.24)


class TestPaymentInfo:
    def test_method_and_card_type_optional(self):
        payment = PaymentInfo()
        assert payment.method is None
        assert payment.card_type is None

    def test_wallet_methods_available(self):
        assert PaymentInfo(method=PaymentMethod.APPLE_PAY).method == "apple_pay"
        assert PaymentInfo(method=PaymentMethod.PAYPAL).method == "paypal"

    def test_no_card_number_field(self):
        assert "last_four" not in PaymentInfo.model_fields


class TestNoPIIInSchema:
    def test_no_field_name_could_hold_personal_data(self):
        offenders = [
            f"{model.__name__}.{field_name}"
            for model in MODELS
            for field_name in model.model_fields
            for fragment in FORBIDDEN_FIELD_FRAGMENTS
            if fragment in field_name.lower()
        ]
        assert offenders == []

    def test_import_time_guard_passes(self):
        assert_pii_free_schema()  # raises AssertionError if a PII field is ever added

    def test_tool_json_schema_exposes_no_pii_properties(self):
        """The JSON schema is what the model is asked to fill in — check it, not just the class."""
        schema = InvoiceExtraction.model_json_schema()
        names: list[str] = []

        def walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "properties" and isinstance(value, dict):
                        names.extend(value)
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(schema)
        assert names, "schema should expose properties"
        assert not [
            name
            for name in names
            for fragment in FORBIDDEN_FIELD_FRAGMENTS
            if fragment in name.lower()
        ]

    def test_unknown_field_is_rejected(self):
        """extra='forbid' is what stops a removed field from sneaking back in."""
        with pytest.raises(ValidationError):
            PaymentInfo(method=PaymentMethod.CREDIT_CARD, last_four="4242")
        with pytest.raises(ValidationError):
            InvoiceExtraction(
                vendor="X",
                total=1.0,
                shipping_address={"street": "123 Main St", "city": "Springfield"},
            )


class TestPlaceholderScrubbing:
    def test_placeholder_only_optional_field_becomes_null(self):
        item = LineItem(product_name="Widget", category="[ADDRESS_1]")
        assert item.category is None

    def test_placeholder_is_removed_from_surrounding_text(self):
        item = LineItem(product_name="Gift card for [PERSON_1]")
        assert item.product_name == "Gift card for"

    def test_placeholder_only_required_field_becomes_blank(self):
        invoice = InvoiceExtraction(vendor="[PERSON_2]", total=10.0)
        assert invoice.vendor == ""

    def test_nested_models_are_scrubbed(self):
        invoice = InvoiceExtraction(
            vendor="Store",
            total=10.0,
            line_items=[LineItem(product_name="Widget", sku="[CARD_1]")],
            discounts=[Discount(description="Coupon mailed to [ADDRESS_3]")],
        )
        assert invoice.line_items[0].sku is None
        assert invoice.discounts[0].description == "Coupon mailed to"

    def test_ordinary_bracketed_text_survives(self):
        item = LineItem(product_name="Widget [2-pack]")
        assert item.product_name == "Widget [2-pack]"


class TestInvoiceExtraction:
    def test_minimal(self):
        invoice = InvoiceExtraction(vendor="TestCo", total=100.00)
        assert invoice.currency is None  # not stated -> not invented
        assert invoice.line_items == []
        assert invoice.discounts == []
        assert invoice.order_id is None
        assert invoice.payment is None

    def test_full(self):
        invoice = InvoiceExtraction(
            vendor="Acme",
            order_id="123-456",
            order_date="2024-01-15",
            delivery_date="2024-01-20",
            line_items=[
                LineItem(product_name="Book", quantity=2, unit_price=10.00, total_price=20.00),
            ],
            subtotal=20.00,
            tax=1.80,
            shipping_cost=5.99,
            discount=1.00,
            discounts=[Discount(description="SPRING10", code="SPRING10", amount=1.00)],
            total=26.79,
            currency="USD",
            payment=PaymentInfo(method=PaymentMethod.CREDIT_CARD, card_type="Visa"),
        )
        assert len(invoice.line_items) == 1
        assert invoice.total == 26.79
        assert invoice.payment.card_type == "Visa"

    def test_json_schema_generation(self):
        schema = InvoiceExtraction.model_json_schema()
        assert "properties" in schema
        assert "vendor" in schema["properties"]
        assert "total" in schema["properties"]

    def test_roundtrip_serialization(self):
        invoice = InvoiceExtraction(
            vendor="Test",
            total=50.00,
            line_items=[LineItem(product_name="Item", total_price=50.00, upc="012345678905")],
        )
        restored = InvoiceExtraction(**invoice.model_dump())
        assert restored.vendor == "Test"
        assert restored.line_items[0].upc == "012345678905"

    def test_json_roundtrip(self):
        invoice = InvoiceExtraction(vendor="Shop", total=10.00)
        restored = InvoiceExtraction.model_validate_json(invoice.model_dump_json())
        assert restored.vendor == "Shop"
