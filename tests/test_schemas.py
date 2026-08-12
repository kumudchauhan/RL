"""Tests for Pydantic schemas."""


from info_extract.schemas import (
    Address,
    InvoiceExtraction,
    LineItem,
    PaymentInfo,
    PaymentMethod,
)


def test_line_item_minimal():
    item = LineItem(product_name="Widget", total_price=9.99)
    assert item.quantity == 1
    assert item.unit_price is None
    assert item.sku is None


def test_line_item_full():
    item = LineItem(
        product_name="Gadget",
        quantity=3,
        unit_price=5.00,
        total_price=15.00,
        sku="ABC123",
    )
    assert item.product_name == "Gadget"
    assert item.quantity == 3


def test_address():
    addr = Address(street="123 Main St", city="Springfield", state="IL", zip_code="62701")
    assert addr.country is None


def test_payment_info():
    payment = PaymentInfo(method=PaymentMethod.CREDIT_CARD, last_four="4242", card_type="Visa")
    assert payment.method == "credit_card"


def test_invoice_extraction_minimal():
    invoice = InvoiceExtraction(vendor="TestCo", total=100.00)
    assert invoice.currency == "USD"
    assert invoice.line_items == []
    assert invoice.order_id is None


def test_invoice_extraction_full():
    invoice = InvoiceExtraction(
        vendor="Amazon",
        order_id="123-456",
        order_date="2024-01-15",
        delivery_date="2024-01-20",
        shipping_address=Address(street="1 Main St", city="Seattle", state="WA", zip_code="98101"),
        line_items=[
            LineItem(product_name="Book", quantity=2, unit_price=10.00, total_price=20.00),
        ],
        subtotal=20.00,
        tax=1.80,
        shipping_cost=5.99,
        total=27.79,
        currency="USD",
        payment=PaymentInfo(method=PaymentMethod.CREDIT_CARD, last_four="1234", card_type="Visa"),
    )
    assert len(invoice.line_items) == 1
    assert invoice.total == 27.79


def test_json_schema_generation():
    schema = InvoiceExtraction.model_json_schema()
    assert "properties" in schema
    assert "vendor" in schema["properties"]
    assert "total" in schema["properties"]


def test_roundtrip_serialization():
    invoice = InvoiceExtraction(
        vendor="Test",
        total=50.00,
        line_items=[LineItem(product_name="Item", total_price=50.00)],
    )
    data = invoice.model_dump()
    restored = InvoiceExtraction(**data)
    assert restored.vendor == "Test"
    assert restored.line_items[0].product_name == "Item"


def test_json_roundtrip():
    invoice = InvoiceExtraction(vendor="Shop", total=10.00)
    json_str = invoice.model_dump_json()
    restored = InvoiceExtraction.model_validate_json(json_str)
    assert restored.vendor == "Shop"
