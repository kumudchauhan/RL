"""Email (.eml) document parser."""

from __future__ import annotations

from email import policy
from email.parser import BytesParser
from pathlib import Path

from bs4 import BeautifulSoup

from .base import DocumentParser, ParsedDocument


class EmlParser(DocumentParser):
    def can_handle(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".eml"

    def parse(self, file_path: str) -> ParsedDocument:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        text_body = ""
        html_body = None

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain" and not text_body:
                    text_body = part.get_content()
                elif content_type == "text/html" and html_body is None:
                    html_body = part.get_content()
        else:
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                text_body = msg.get_content()
            elif content_type == "text/html":
                html_body = msg.get_content()

        # If no plain text, convert HTML to text
        if not text_body and html_body:
            soup = BeautifulSoup(html_body, "lxml")
            text_body = soup.get_text(separator="\n", strip=True)

        return ParsedDocument(
            source_path=file_path,
            format="eml",
            subject=msg.get("Subject"),
            sender=msg.get("From"),
            recipient=msg.get("To"),
            date_received=msg.get("Date"),
            text_body=text_body,
            html_body=html_body,
            metadata={
                "message_id": msg.get("Message-ID"),
                "headers": dict(msg.items()),
            },
        )
