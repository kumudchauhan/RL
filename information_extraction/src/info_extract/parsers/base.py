"""Base parser interface and ParsedDocument dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """Unified representation of a parsed document."""

    source_path: str
    format: str  # "eml" | "pdf"
    subject: str | None
    sender: str | None
    recipient: str | None
    date_received: str | None
    text_body: str
    html_body: str | None
    metadata: dict = field(default_factory=dict)


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse a document file and return structured content."""
        ...

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """Return True if this parser can handle the given file."""
        ...
