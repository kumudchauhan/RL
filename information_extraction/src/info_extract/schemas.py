"""Pydantic models for invoice extraction output.

Two properties are enforced here, not left to the prompt:

1. **The schema carries no personal data.** There is no field for a name, an address, a phone
   number, an email, or a card number — not even a redacted or synthetic one. A value that
   cannot be stored cannot leak into a result file or a captured rollout.
   :func:`assert_pii_free_schema` runs at import time, so adding such a field breaks the build.
2. **Nothing is invented.** Every field is optional unless the document always states it, so a
   value that is not printed on the receipt comes back ``null`` (or ``[]``) rather than guessed.
   Redaction placeholders that survive into a model response (``[PERSON_1]``, ``[ADDRESS_2]``)
   are stripped by :class:`ExtractionModel`, so masked values do not reach the output either.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Placeholders left behind by ``parsers.pii``. They mean "a value was removed here", so they
#: are never a legitimate extraction — see ``ExtractionModel._strip_placeholders``.
PLACEHOLDER_RE = re.compile(r"\[(?:PERSON|EMAIL|PHONE|ADDRESS|CARD)_\d+\]")

#: Field-name fragments that would turn this schema into a carrier for personal data.
#: Checked against every field of every model at import time.
FORBIDDEN_FIELD_FRAGMENTS: tuple[str, ...] = (
    "address",
    "street",
    "city",
    "state",
    "zip",
    "postal",
    "phone",
    "email",
    "last_four",
    "card_number",
    "ssn",
    "customer",
    "recipient",
    "buyer",
    "contact",
    "person",
    "birth",
)


def _allows_none(annotation: object) -> bool:
    return type(None) in get_args(annotation)


class ExtractionModel(BaseModel):
    """Base model for extraction output: strict about keys, hostile to placeholders."""

    #: An unexpected key is an error, not something to silently keep: it is the one way a
    #: field this schema deliberately omits could reappear in the output.
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _strip_placeholders(self) -> ExtractionModel:
        """Drop redaction placeholders from string values.

        The prompt already tells the model to treat ``[PERSON_1]`` as missing information;
        this makes it true regardless of what the model does.
        """
        for name, field_info in type(self).model_fields.items():
            value = getattr(self, name, None)
            if not isinstance(value, str) or not PLACEHOLDER_RE.search(value):
                continue
            cleaned = re.sub(r"\s{2,}", " ", PLACEHOLDER_RE.sub(" ", value)).strip(" ,;:|-/")
            replacement = cleaned or (None if _allows_none(field_info.annotation) else "")
            object.__setattr__(self, name, replacement)
        return self


class PaymentMethod(str, Enum):
    """How the order was paid for, as stated on the document."""

    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    GIFT_CARD = "gift_card"
    STORE_CREDIT = "store_credit"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    PAYPAL = "paypal"
    VENMO = "venmo"
    CASH = "cash"
    CHECK = "check"
    EBT = "ebt"
    BANK_TRANSFER = "bank_transfer"
    BUY_NOW_PAY_LATER = "buy_now_pay_later"
    OTHER = "other"


class Payment(ExtractionModel):
    """One payment instrument the document says the order was paid with.

    Orders split across instruments (gift card + credit card, two cards, EBT + debit) print one
    of these per tender line, so :attr:`InvoiceExtraction.payments` is a list. ``amount`` is what
    the document says went onto *this* instrument, which is why the sum can be less than the
    order total — an instalment/EMI plan charges only the first instalment now.

    There is deliberately no ``last_four`` field: the card number is masked in the parser, and
    a field for it would only invite a masked or invented value into the output.
    """

    method: PaymentMethod | None = Field(
        None, description="Payment method as stated on the document; null if it does not say"
    )
    card_type: str | None = Field(
        None,
        description=(
            "Card brand/network exactly as printed (Visa, Mastercard, AMEX, Discover); "
            "null if not printed"
        ),
    )
    amount: float | None = Field(
        None,
        description=(
            "Amount charged to this instrument, if the document prints a per-tender amount"
        ),
    )
    installment_count: int | None = Field(
        None,
        description=(
            "Number of instalments/EMIs, only if the document states a plan "
            "(e.g. '6 monthly payments'); null otherwise"
        ),
    )
    installment_amount: float | None = Field(
        None, description="Amount of each instalment/EMI, only if printed"
    )


class Fee(ExtractionModel):
    """A fee or surcharge line with no dedicated field, transcribed with its printed label.

    Delivery platforms invent fee lines constantly ('Regulatory Response Fee', 'Small Order
    Fee', 'Bag Fee', 'Bottle Deposit'). Rather than guess which dedicated field each belongs
    in — or drop it — the label is kept verbatim.
    """

    label: str = Field(
        ..., description="Fee label exactly as printed, e.g. 'Bag Fee', 'Bottle Deposit'"
    )
    amount: float | None = Field(None, description="Fee amount as printed")


class ProductIdentifier(ExtractionModel):
    """An identifier printed on the document that has no dedicated field."""

    label: str = Field(
        ..., description="Identifier label exactly as printed, e.g. 'ISBN', 'Style #', 'PLU'"
    )
    value: str = Field(..., description="Identifier value exactly as printed")


class Discount(ExtractionModel):
    """A discount, coupon, or promotion printed on the document."""

    description: str | None = Field(
        None, description="The discount line exactly as printed, e.g. 'GROUPON $10 OFF'"
    )
    code: str | None = Field(
        None, description="Coupon/promo code as printed; null if no code is shown"
    )
    source: str | None = Field(
        None,
        description=(
            "Who issued it, only if the document says so, e.g. 'Groupon', "
            "'manufacturer coupon', 'store'"
        ),
    )
    amount: float | None = Field(
        None, description="Amount saved as a positive number, if an amount is printed"
    )
    percentage: float | None = Field(
        None, description="Percentage saved, only if the document states a percentage"
    )


class LineItem(ExtractionModel):
    """One product line, transcribed from the document.

    Every field except ``product_name`` is optional: receipts differ wildly in what they print,
    and an absent field must stay absent rather than being inferred from the product name.
    """

    product_name: str = Field(..., description="Product name/description exactly as printed")

    quantity: float | None = Field(
        None, description="Quantity as printed (may be fractional, e.g. 1.24 lb); null if absent"
    )
    quantity_unit: str | None = Field(
        None, description="Unit of the quantity as printed, e.g. 'ea', 'lb', 'kg', 'oz'"
    )
    unit_price: float | None = Field(None, description="Price per unit, if printed")
    total_price: float | None = Field(None, description="Line total, if printed")

    # --- identifiers: capture every one the document prints ---
    sku: str | None = Field(None, description="SKU exactly as printed")
    upc: str | None = Field(None, description="UPC/EAN/barcode number exactly as printed")
    asin: str | None = Field(None, description="ASIN exactly as printed")
    product_number: str | None = Field(
        None, description="Vendor product/item/model number exactly as printed"
    )
    other_identifiers: list[ProductIdentifier] = Field(
        default_factory=list,
        description="Any further identifiers printed for this item, with their printed labels",
    )

    # --- taxonomy: only what the document itself states ---
    department: str | None = Field(
        None,
        description=(
            "Department/section the document groups this item under, e.g. 'PRODUCE', "
            "'GROCERY'; null unless printed"
        ),
    )
    category: str | None = Field(
        None,
        description=(
            "Category as printed for this item, e.g. 'Beverages', 'Makeup'; "
            "null unless printed — never infer it from the product name"
        ),
    )

    discounts: list[Discount] = Field(
        default_factory=list, description="Discounts/coupons printed against this line item"
    )


class InvoiceExtraction(ExtractionModel):
    """Schema for extracted invoice / order-confirmation / receipt data."""

    vendor: str = Field(
        ...,
        description=(
            "The store/retailer the goods were bought from, as printed (e.g. 'Costco', "
            "'Safeway'). For a marketplace or delivery order this is the store, not the "
            "platform — the platform goes in service_provider"
        ),
    )
    service_provider: str | None = Field(
        None,
        description=(
            "Platform/marketplace that took or fulfilled the order, when the document names "
            "one distinct from the store (e.g. 'Instacart', 'DoorDash', 'Uber Eats'); null if "
            "the store billed directly"
        ),
    )
    order_id: str | None = Field(None, description="Order/confirmation/receipt number")
    order_date: str | None = Field(None, description="Date of order (ISO 8601: YYYY-MM-DD)")
    delivery_date: str | None = Field(
        None, description="Expected/actual delivery date (ISO 8601: YYYY-MM-DD)"
    )

    line_items: list[LineItem] = Field(default_factory=list)

    # --- money: billed and paid are separate questions ---
    subtotal: float | None = Field(
        None, description="Items subtotal before tax, fees, and shipping"
    )
    tax: float | None = Field(None, description="Tax amount")
    shipping_cost: float | None = Field(None, description="Shipping cost")
    delivery_fee: float | None = Field(
        None, description="Delivery fee, when the document prints one separately from shipping"
    )
    service_fee: float | None = Field(
        None, description="Service fee (platform/booking/convenience fee), if printed"
    )
    tip: float | None = Field(None, description="Tip/gratuity, if printed")
    fees: list[Fee] = Field(
        default_factory=list,
        description="Any further fee/surcharge lines, with their labels exactly as printed",
    )
    discount: float | None = Field(
        None, description="Order-level discount total as a positive value"
    )
    discounts: list[Discount] = Field(
        default_factory=list, description="Order-level discounts/coupons as printed"
    )
    total: float = Field(
        ...,
        description=(
            "Total billed for the order — the invoice/order total as printed. Not necessarily "
            "what was paid at checkout"
        ),
    )
    amount_paid: float | None = Field(
        None,
        description=(
            "Total actually paid/charged, if the document states it (e.g. 'Total paid', "
            "'Amount charged'). Differs from total when payment is split across instruments, "
            "part-covered by a gift card, or spread over instalments"
        ),
    )
    amount_due: float | None = Field(
        None,
        description=(
            "Balance still outstanding, only if the document prints one "
            "(e.g. 'Balance due', remaining instalments)"
        ),
    )
    currency: str | None = Field(
        None,
        description=(
            "ISO currency code the document states or symbolises ('$' -> USD); "
            "null if the document gives no indication"
        ),
    )

    payments: list[Payment] = Field(
        default_factory=list,
        description=(
            "One entry per payment instrument/tender line the document shows; empty if the "
            "document does not say how it was paid"
        ),
    )


def assert_pii_free_schema() -> None:
    """Fail loudly if any model field could hold personal data.

    Called at import time: the guarantee that extraction output contains no names, addresses,
    phone numbers, emails, or card numbers rests on those fields not existing.
    """
    models: list[type[BaseModel]] = [
        InvoiceExtraction,
        LineItem,
        ProductIdentifier,
        Discount,
        Fee,
        Payment,
    ]
    offenders = [
        f"{model.__name__}.{field_name}"
        for model in models
        for field_name in model.model_fields
        for fragment in FORBIDDEN_FIELD_FRAGMENTS
        if fragment in field_name.lower()
    ]
    if offenders:
        raise AssertionError(
            "extraction schema must not contain personal-data fields: " + ", ".join(offenders)
        )


assert_pii_free_schema()
