"""ExtractionTask dataclass for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from ..parsers.base import ParsedDocument
from ..schemas import InvoiceExtraction


@dataclass
class ExtractionTask:
    """A single extraction task instance."""

    task_id: str
    source_file: str
    parsed_document: ParsedDocument
    ground_truth: InvoiceExtraction

    def to_harbor_task(self) -> dict:
        """Serialize to Harbor task format."""
        return {
            "id": self.task_id,
            "input": {
                "document_text": self.parsed_document.text_body,
                "document_html": self.parsed_document.html_body,
                "metadata": {
                    "source": self.source_file,
                    "format": self.parsed_document.format,
                    "subject": self.parsed_document.subject,
                    "sender": self.parsed_document.sender,
                },
            },
            "expected_output": self.ground_truth.model_dump(),
        }
