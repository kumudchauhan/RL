"""Pydantic models for invoice extraction output."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PaymentMethod(str, Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    GIFT_CARD = "gift_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    OTHER = "other"


class LineItem(BaseModel):
    product_name: str = Field(..., description="Name/description of the product")
    quantity: int = Field(1, description="Quantity ordered")
    unit_price: float | None = Field(None, description="Price per unit")
    total_price: float = Field(..., description="Total price for this line item")
    sku: str | None = Field(None, description="Product SKU or ASIN if available")


class Address(BaseModel):
    street: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    country: str | None = None


class PaymentInfo(BaseModel):
    method: PaymentMethod
    last_four: str | None = Field(None, description="Last 4 digits of card")
    card_type: str | None = Field(None, description="Visa, Mastercard, etc.")


class InvoiceExtraction(BaseModel):
    """Schema for extracted invoice/order confirmation data."""

    vendor: str = Field(..., description="Retailer or vendor name")
    order_id: str | None = Field(None, description="Order/confirmation number")
    order_date: str | None = Field(None, description="Date of order (ISO 8601: YYYY-MM-DD)")
    delivery_date: str | None = Field(
        None, description="Expected/actual delivery date (ISO 8601: YYYY-MM-DD)"
    )

    shipping_address: Address | None = None
    billing_address: Address | None = None

    line_items: list[LineItem] = Field(default_factory=list)

    subtotal: float | None = Field(None, description="Subtotal before tax/shipping")
    tax: float | None = Field(None, description="Tax amount")
    shipping_cost: float | None = Field(None, description="Shipping cost")
    discount: float | None = Field(None, description="Discount amount (positive value)")
    total: float = Field(..., description="Total amount charged")
    currency: str = Field("USD", description="ISO currency code")

    payment: PaymentInfo | None = None
    notes: str | None = Field(None, description="Any additional relevant notes")
