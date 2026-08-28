"""Tests for document parsers.

Sample-driven tests run against whatever is in `invoices/`, which is gitignored:
no real filename, subject, order number, or amount is hardcoded here, so the
suite carries no trace of the documents it was developed against.
"""

import re
from pathlib import Path

import pytest

from info_extract.parsers.base import ParsedDocument
from info_extract.parsers.eml_parser import EmlParser

INVOICES_DIR = Path(__file__).parent.parent / "invoices"
EML_FILES = sorted(INVOICES_DIR.glob("*.eml")) if INVOICES_DIR.exists() else []
requires_eml = pytest.mark.skipif(not EML_FILES, reason="no sample .eml in invoices/")

#: An order/receipt reference in a subject line: a long digit run, optionally
#: hyphenated ("1000200-12345678", "[1000000123]").
ORDER_REF_RE = re.compile(r"\d[\d\-]{6,}\d")


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


@requires_eml
class TestEmlParserOnSamples:
    def setup_method(self):
        self.parser = EmlParser()

    def test_every_sample_yields_subject_sender_and_body(self):
        for path in EML_FILES:
            doc = self.parser.parse(str(path))
            assert isinstance(doc, ParsedDocument), path.name
            assert doc.format == "eml", path.name
            assert doc.subject, path.name
            assert doc.sender, path.name
            assert doc.text_body or doc.html_body, path.name

    def test_order_references_in_subject_survive_redaction(self):
        """Order ids are extraction targets, so redaction must leave them alone."""
        raw_parser = EmlParser(redact_pii=False)
        checked = 0

        for path in EML_FILES:
            refs = ORDER_REF_RE.findall(raw_parser.parse(str(path)).subject or "")
            if not refs:
                continue
            subject = self.parser.parse(str(path)).subject or ""
            for ref in refs:
                assert ref in subject, path.name
            checked += 1

        if not checked:
            pytest.skip("no sample subject carries an order reference")

    def test_parse_returns_metadata(self):
        doc = self.parser.parse(str(EML_FILES[0]))
        assert "headers" in doc.metadata
        assert doc.source_path == str(EML_FILES[0])
