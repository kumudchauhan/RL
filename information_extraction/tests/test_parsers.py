"""Tests for document parsers."""

from pathlib import Path

import pytest

from info_extract.parsers.base import ParsedDocument
from info_extract.parsers.eml_parser import EmlParser

INVOICES_DIR = Path(__file__).parent.parent / "invoices"


class TestEmlParser:
    def setup_method(self):
        self.parser = EmlParser()

    def test_can_handle_eml(self):
        assert self.parser.can_handle("test.eml")
        assert self.parser.can_handle("/path/to/file.eml")
        assert self.parser.can_handle("FILE.EML")

    def test_cannot_handle_other(self):
        assert not self.parser.can_handle("test.pdf")
        assert not self.parser.can_handle("test.txt")
        assert not self.parser.can_handle("test.html")

    @pytest.mark.skipif(
        not (INVOICES_DIR / "Your Store Receipt.eml").exists(),
        reason="Sample invoice not available",
    )
    def test_parse_store_receipt(self):
        path = str(INVOICES_DIR / "Your Store Receipt.eml")
        doc = self.parser.parse(path)

        assert isinstance(doc, ParsedDocument)
        assert doc.format == "eml"
        assert doc.subject == "Your Store Receipt"
        assert "storemail" in (doc.sender or "").lower()
        assert doc.text_body  # Should have content
        assert "19.99" in doc.text_body

    @pytest.mark.skipif(
        not (INVOICES_DIR / "Your receipt from Acme.eml").exists(),
        reason="Sample invoice not available",
    )
    def test_parse_acme(self):
        path = str(INVOICES_DIR / "Your receipt from Acme.eml")
        doc = self.parser.parse(path)

        assert isinstance(doc, ParsedDocument)
        assert doc.format == "eml"
        assert "Acme" in (doc.subject or "")
        assert doc.text_body or doc.html_body

    @pytest.mark.skipif(
        not (INVOICES_DIR / "Thank you for your Acme Cosmetics order [1000000123].eml").exists(),
        reason="Sample invoice not available",
    )
    def test_parse_cosmetics_order(self):
        path = str(
            INVOICES_DIR / "Thank you for your Acme Cosmetics order [1000000123].eml"
        )
        doc = self.parser.parse(path)

        assert isinstance(doc, ParsedDocument)
        assert doc.format == "eml"
        assert "1000000123" in (doc.subject or "")
        assert doc.text_body

    def test_parse_returns_metadata(self):
        # Test with any available eml file
        eml_files = list(INVOICES_DIR.glob("*.eml"))
        if not eml_files:
            pytest.skip("No eml files available")

        doc = self.parser.parse(str(eml_files[0]))
        assert "headers" in doc.metadata
        assert doc.source_path == str(eml_files[0])
