"""PDF document parser with two-tier PII redaction.

Boilerplate first pass: text-layer extraction only. Scanned/image-only PDFs
produce no text and are reported as such rather than silently returning "".
"""

from __future__ import annotations

from pathlib import Path

from .base import DocumentParser, ParsedDocument
from .pii import PIIPolicy, PIIRedactor

#: Document-info keys dropped outright — the PDF analogue of routing headers.
#: `creator`/`producer` carry the browser/OS build that printed the receipt
#: (e.g. "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ... Chrome/151"),
#: which fingerprints the user's device without helping extraction.
DROPPED_INFO_KEYS = frozenset({"creator", "producer", "encryption", "trapped", "format"})

#: Document-info keys that commonly hold a person's name.
NAME_INFO_KEYS = ("author",)


class PdfParser(DocumentParser):
    """Parses .pdf files, masking PII before content leaves the parser.

    Same two-tier shape as :class:`~.eml_parser.EmlParser`, with the PDF's
    document-info dictionary standing in for the email envelope:

    1. Document info — author/title/subject redacted, device-fingerprinting keys dropped.
    2. Content — postal addresses, emails, phones, card numbers, names in the page text.
    """

    def __init__(self, redact_pii: bool = True, pii_policy: PIIPolicy | None = None):
        self.redact_pii = redact_pii
        self.pii_policy = pii_policy or PIIPolicy.from_env()

    def can_handle(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".pdf"

    def parse(self, file_path: str) -> ParsedDocument:
        try:
            import pymupdf
        except ImportError as e:  # pragma: no cover - depends on install extras
            raise ImportError(
                "PDF parsing needs pymupdf. Install it with: uv sync --extra pdf"
            ) from e

        with pymupdf.open(file_path) as doc:
            pages = [page.get_text() for page in doc]
            info = {k: v for k, v in (doc.metadata or {}).items() if v}
            page_count = doc.page_count

        text_body = "\n".join(pages).strip()
        title = info.get("title")
        file_name = Path(file_path).name
        metadata = {
            "source_name": file_name,
            "page_count": page_count,
            "has_text_layer": bool(text_body),
            "pdf_info": {k: v for k, v in info.items() if k not in DROPPED_INFO_KEYS},
        }

        if not self.redact_pii:
            return ParsedDocument(
                source_path=file_path,
                format="pdf",
                subject=title,
                sender=None,
                recipient=None,
                date_received=info.get("creationDate"),
                text_body=text_body,
                html_body=None,
                metadata=metadata,
            )

        redactor = PIIRedactor(self.pii_policy)

        # Learn identities before masking anything, so a name found in the
        # document info is also masked in the page text (and vice versa).
        for key in NAME_INFO_KEYS:
            redactor.learn_name(info.get(key))
        redactor.learn_names_from_text(text_body)

        # Tier 1: document info
        metadata["pdf_info"] = {
            k: redactor.redact_text(v) if isinstance(v, str) else v
            for k, v in metadata["pdf_info"].items()
        }
        metadata["dropped_info_keys"] = sorted(set(info) & DROPPED_INFO_KEYS)

        # Tier 2: page content
        text_body = redactor.redact_text(text_body)

        # Filenames leak too ("invoice_2023-04-12-Roe.pdf"), and this is the
        # name the prompt shows the model. Redact last, once names are known.
        metadata["source_name"] = redactor.redact_text(file_name)

        return ParsedDocument(
            source_path=file_path,
            format="pdf",
            subject=redactor.redact_text(title) if title else title,
            sender=None,
            recipient=None,
            date_received=info.get("creationDate"),
            text_body=text_body,
            html_body=None,
            metadata=metadata,
            pii_redacted=True,
            redaction_report=redactor.report.to_dict(),
        )
