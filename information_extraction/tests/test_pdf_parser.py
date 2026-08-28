"""Tests for the PDF parser and its PII redaction.

The PDF path stresses redaction differently from email: text extraction emits one
line per visual field, so addresses and names arrive split across lines and are
often ALL CAPS. Tests that need a real document are skipped when `invoices/` is
empty, since it is gitignored.
"""

from pathlib import Path

import pytest

from info_extract.parsers.pdf_parser import PdfParser
from info_extract.parsers.pii import PIIPolicy, PIIRedactor

pymupdf = pytest.importorskip("pymupdf")

INVOICES_DIR = Path(__file__).parent.parent / "invoices"
PDF_FILES = sorted(INVOICES_DIR.glob("*.pdf")) if INVOICES_DIR.exists() else []
requires_pdf = pytest.mark.skipif(not PDF_FILES, reason="no sample PDFs in invoices/")


def raw_text(path: Path) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


class TestCanHandle:
    def test_accepts_pdf(self):
        assert PdfParser().can_handle("invoices/receipt.PDF")

    def test_rejects_other_formats(self):
        parser = PdfParser()
        assert not parser.can_handle("invoices/receipt.eml")
        assert not parser.can_handle("invoices/receipt.png")


@requires_pdf
class TestParse:
    def test_extracts_text_and_metadata(self):
        doc = PdfParser().parse(str(PDF_FILES[0]))
        assert doc.format == "pdf"
        assert doc.text_body
        assert doc.metadata["page_count"] >= 1
        assert doc.metadata["has_text_layer"] is True

    def test_drops_device_fingerprinting_info(self):
        doc = PdfParser().parse(str(PDF_FILES[0]))
        assert "producer" not in doc.metadata["pdf_info"]
        assert "creator" not in doc.metadata["pdf_info"]

    def test_reports_redaction(self):
        doc = PdfParser().parse(str(PDF_FILES[0]))
        assert doc.pii_redacted is True
        assert set(doc.redaction_report) == {
            "total_unique_values",
            "by_category",
            "placeholders",
            "dropped_headers",
        }

    def test_opt_out_leaves_text_untouched(self):
        path = PDF_FILES[0]
        doc = PdfParser(redact_pii=False).parse(str(path))
        assert doc.pii_redacted is False
        assert doc.text_body == raw_text(path).strip()

    @pytest.mark.parametrize("path", PDF_FILES, ids=lambda p: p.name)
    def test_no_placeholders_survive_as_raw_values(self, path):
        """Every masked category is actually gone from the emitted text."""
        doc = PdfParser().parse(str(path))
        blob = "\n".join(
            [doc.subject or "", doc.text_body or "", str(doc.metadata)]
        )
        assert "@gmail.com" not in blob
        assert "@yahoo.com" not in blob
        assert "@hotmail.com" not in blob


class TestColumnLayoutRedaction:
    """PDF text extraction splits a field per line; these are the shapes seen."""

    def test_name_learned_from_recipient_label_above(self):
        r = PIIRedactor()
        out = r.redact_text("Customer Information\nJANE ROE\nOrder # WH10000001\n")
        assert "ROE" not in out
        assert "WH10000001" in out  # order id must survive

    def test_name_learned_from_street_line_below(self):
        r = PIIRedactor()
        out = r.redact_text("1000042 - JANE ROE\n500 OAK AVENUE\nAPT 210\n")
        assert "ROE" not in out
        assert "OAK" not in out
        assert "210" not in out
        assert "1000042" in out  # receipt id is not PII

    def test_column_address_block_fully_masked(self):
        r = PIIRedactor()
        out = r.redact_text(
            "Jane Roe\n123 Main St, Apt 5\nSpringfield\nIL\n62704\n10000042\n"
        )
        assert "Main St" not in out
        assert "Springfield" not in out
        assert "62704" not in out
        assert "10000042" in out  # invoice number, not a postal code

    def test_spelled_out_state(self):
        r = PIIRedactor()
        assert "Springfield" not in r.redact_text("Springfield, Illinois 62704")

    def test_zip_before_city(self):
        r = PIIRedactor()
        out = r.redact_text("500 Oak avenue, APT 210\n53703 Madison\nWI\n")
        assert "Madison" not in out
        assert "53703" not in out
        assert "WI" not in out

    def test_lone_zip_next_to_country(self):
        r = PIIRedactor()
        out = r.redact_text("Bill to\nJane Roe\n62704\nUnited States\n")
        assert "62704" not in out

    def test_lone_number_kept_without_address_context(self):
        r = PIIRedactor()
        assert "62704" in r.redact_text("Order total\n62704\nThank you\n")

    def test_label_stack_is_not_learned_as_a_name(self):
        """"Customer Name" above "Doc No" must not make "Doc" a masked name."""
        r = PIIRedactor()
        out = r.redact_text("Customer Name\nDoc No\nDOC/I/ABC/1234\n")
        assert "DOC/I/ABC/1234" in out

    def test_company_under_recipient_label_is_not_a_person(self):
        r = PIIRedactor()
        out = r.redact_text("Bill to\nMeisterLabs Inc\n113 Cherry St\n")
        assert "MeisterLabs Inc" in out

    def test_city_state_zip_does_not_span_unrelated_lines(self):
        """A 5-digit total must not read as the ZIP of a nearby city/state pair."""
        r = PIIRedactor()
        assert "12345.67" in r.redact_text("NA\nNA\n 12345.67\n")

    def test_serial_numbers_are_not_card_numbers(self):
        r = PIIRedactor()
        assert "356938035643809" in r.redact_text("SNo/IMEI: 356938035643809")


class TestPolicyFromEnv:
    def test_extra_names_from_env(self, monkeypatch):
        monkeypatch.setenv("INFO_EXTRACT_PII_NAMES", "Ada Lovelace, Ada L")
        policy = PIIPolicy.from_env()
        assert policy.extra_names == ("Ada Lovelace", "Ada L")
        assert "Lovelace" not in PIIRedactor(policy).redact_text("Sold to Lovelace")

    def test_absent_env_leaves_defaults(self, monkeypatch):
        monkeypatch.delenv("INFO_EXTRACT_PII_NAMES", raising=False)
        assert PIIPolicy.from_env().extra_names == ()

    def test_overrides_are_kept(self, monkeypatch):
        monkeypatch.setenv("INFO_EXTRACT_PII_NAMES", "Ada Lovelace")
        policy = PIIPolicy.from_env(mask_addresses=False, extra_names=("Grace H",))
        assert policy.mask_addresses is False
        assert policy.extra_names == ("Grace H", "Ada Lovelace")
