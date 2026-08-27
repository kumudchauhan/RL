"""Tests for two-tier PII redaction."""

from pathlib import Path

import pytest

from info_extract.parsers.eml_parser import EmlParser
from info_extract.parsers.pii import PIIPolicy, PIIRedactor

INVOICES_DIR = Path(__file__).parent.parent / "invoices"


class TestTier1Envelope:
    def test_recipient_fully_masked(self):
        r = PIIRedactor()
        assert r.redact_recipient("jane.doe@example.com") == "[EMAIL_1]"

    def test_sender_keeps_vendor_domain(self):
        r = PIIRedactor()
        out = r.redact_sender('"Vendor.example" <help@vendor.example>')
        assert "help" not in out
        assert "vendor.example" in out

    def test_sender_domain_maskable(self):
        r = PIIRedactor(PIIPolicy(preserve_sender_domain=False))
        assert "vendor.example" not in r.redact_sender("help@vendor.example")

    def test_routing_headers_dropped(self):
        r = PIIRedactor()
        headers = r.redact_headers(
            [
                ("Subject", "Your receipt"),
                ("Received", "from mx.example.com by user@gmail.com"),
                ("Received-SPF", "pass (domain of bounces+1-jane=gmail.com@x.com)"),
                ("Message-ID", "<abc@mail.example.com>"),
                ("To", "jane.doe@example.com"),
            ]
        )
        assert set(headers) == {"Subject", "To"}
        assert headers["To"] == "[EMAIL_1]"
        assert set(r.report.dropped_headers) == {"received", "received-spf", "message-id"}

    def test_routing_headers_redacted_when_kept(self):
        r = PIIRedactor(PIIPolicy(drop_routing_headers=False))
        headers = r.redact_headers([("Received", "for jane.doe@example.com; Mon")])
        assert "jane.doe" not in headers["Received"]


class TestTier2Content:
    def test_email_in_body(self):
        r = PIIRedactor()
        out = r.redact_text("Questions? Write to jane.doe@example.com today.")
        assert "jane.doe@example.com" not in out
        assert "[EMAIL_1]" in out

    def test_url_encoded_email(self):
        r = PIIRedactor()
        assert "jane.doe" not in r.redact_text("https://x.com/u?e=jane.doe%40example.com")

    def test_phone_numbers(self):
        r = PIIRedactor()
        for raw in ("408-555-0142", "(408) 555-0142", "+1 408 555 0142", "1-800-555-0199"):
            out = r.redact_text(f"Call {raw} for help")
            assert raw not in out, raw
            assert "[PHONE_" in out, raw

    def test_order_numbers_and_amounts_survive_phone_masking(self):
        r = PIIRedactor()
        text = "Order number: 1000200-12345678 Total $19.99 Order # 111-2223334-4445556"
        assert r.redact_text(text) == text

    def test_street_and_city_state_zip(self):
        r = PIIRedactor()
        out = r.redact_text("123 Main St, Apt 5, Springfield, CA, 62704, USA")
        assert "Main St" not in out
        assert "62704" not in out
        assert "USA" in out

    def test_multiline_address_block_including_unit_line(self):
        r = PIIRedactor()
        out = r.redact_text("Ship to:\nJane Doe\n500 Oak Ave\n914\nMadison, WI 53703\n")
        assert "Oak" not in out
        assert "914" not in out
        assert "53703" not in out
        assert "Jane" not in out

    def test_reordered_city_zip_state(self):
        r = PIIRedactor()
        out = r.redact_text("123 Main St Apt 5 Springfield 62704-1234 CA United States")
        assert "Springfield" not in out
        assert "62704" not in out

    def test_po_box(self):
        r = PIIRedactor()
        assert "1234" not in r.redact_text("PO Box 1234")

    def test_card_numbers(self):
        r = PIIRedactor()
        assert "4321" not in r.redact_text("**** **** **** 4321")
        assert "9876" not in r.redact_text("Payment: ************9876")
        assert "5678" not in r.redact_text("American Express *5678")

    def test_card_ending_phrase_keeps_wording(self):
        r = PIIRedactor()
        out = r.redact_text("AMEX ending in 5678")
        assert "5678" not in out
        assert out.startswith("AMEX ending in [CARD_")

    def test_full_pan_masked_only_when_luhn_valid(self):
        r = PIIRedactor()
        assert "4111 1111 1111 1111" not in r.redact_text("Card 4111 1111 1111 1111")
        # Not a valid card number: an order number of the same shape is preserved.
        assert r.redact_text("Ref 1234 5678 9012 3456") == "Ref 1234 5678 9012 3456"

    def test_html_entities_and_store_numbers_preserved(self):
        r = PIIRedactor()
        text = "&#8199; &#847; Store #1234"
        assert r.redact_text(text) == text


class TestNames:
    def test_name_from_email_local_part(self):
        r = PIIRedactor()
        r.learn_recipient("janedoe5@example.com")
        assert "janedoe" not in r.redact_text("Hello janedoe, your order shipped").lower()

    def test_name_from_greeting_and_ship_to(self):
        r = PIIRedactor()
        r.learn_names_from_text("Hi Jane,\nShip to:\nJane Doe\n")
        out = r.redact_text("Thanks Jane Doe. Jane, your order shipped.")
        assert "Jane" not in out
        assert "Doe" not in out

    def test_full_name_collapses_to_one_placeholder(self):
        r = PIIRedactor()
        r.learn_name("Jane Doe")
        assert r.redact_text("Jane Doe") == "[PERSON_1]"

    def test_role_addresses_not_treated_as_names(self):
        r = PIIRedactor()
        r.learn_recipient("orders@example.com")
        text = "Your orders are on the way"
        assert r.redact_text(text) == text

    def test_labels_not_learned_as_names(self):
        """A label captured as a name would be masked everywhere in the body."""
        r = PIIRedactor()
        r.learn_names_from_text("Shipping address\nShipping address\nJane Doe\n")
        out = r.redact_text("Shipping address for Jane Doe")
        assert "Shipping address" in out
        assert "Jane" not in out

    def test_extra_names_policy(self):
        r = PIIRedactor(PIIPolicy(extra_names=("Ravi",)))
        assert "Ravi" not in r.redact_text("Gift for Ravi")


class TestPolicyToggles:
    def test_selective_disable(self):
        r = PIIRedactor(PIIPolicy(mask_addresses=False, mask_phones=False))
        text = "500 Oak Ave, Madison, WI 53703 — 408-555-0142 — jane@example.com"
        out = r.redact_text(text)
        assert "500 Oak Ave" in out
        assert "408-555-0142" in out
        assert "jane@example.com" not in out

    def test_placeholders_are_stable_per_value(self):
        r = PIIRedactor()
        out = r.redact_text("jane@example.com and jane@example.com and bob@example.com")
        assert out.count("[EMAIL_1]") == 2
        assert "[EMAIL_2]" in out

    def test_report_has_counts_but_no_raw_values(self):
        r = PIIRedactor()
        r.redact_text("jane@example.com 408-555-0142")
        report = r.report.to_dict()
        assert report["by_category"] == {"EMAIL": 1, "PHONE": 1}
        assert report["total_unique_values"] == 2
        assert "jane" not in str(report)


@pytest.mark.skipif(not list(INVOICES_DIR.glob("*.eml")), reason="No eml files available")
class TestEmlParserRedaction:
    def test_redaction_on_by_default(self):
        doc = EmlParser().parse(str(min(INVOICES_DIR.glob("*.eml"))))
        assert doc.pii_redacted
        assert doc.redaction_report["total_unique_values"] > 0

    def test_opt_out(self):
        path = str(min(INVOICES_DIR.glob("*.eml")))
        doc = EmlParser(redact_pii=False).parse(path)
        assert not doc.pii_redacted
        assert doc.redaction_report == {}

    def test_no_recipient_address_leaks_anywhere(self):
        """The recipient's address must not survive in any field of any sample doc."""
        parser = EmlParser()
        raw_parser = EmlParser(redact_pii=False)

        for path in sorted(INVOICES_DIR.glob("*.eml")):
            raw = raw_parser.parse(str(path))
            if not raw.recipient or "@" not in raw.recipient:
                continue
            address = raw.recipient.split("<")[-1].strip(">").strip().lower()
            local_part = address.split("@")[0]

            doc = parser.parse(str(path))
            blob = " ".join(
                str(part)
                for part in (
                    doc.subject,
                    doc.sender,
                    doc.recipient,
                    doc.text_body,
                    doc.html_body,
                    doc.metadata,
                )
                if part
            ).lower()
            assert address not in blob, path.name
            assert local_part not in blob, path.name
