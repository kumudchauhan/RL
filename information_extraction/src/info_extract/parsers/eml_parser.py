"""Email (.eml) document parser with two-tier PII redaction."""

from __future__ import annotations

from email import policy as email_policy
from email.parser import BytesParser
from pathlib import Path

from bs4 import BeautifulSoup

from .base import DocumentParser, ParsedDocument
from .pii import PIIPolicy, PIIRedactor


class EmlParser(DocumentParser):
    """Parses .eml files, masking PII before content leaves the parser.

    Redaction happens in two tiers against a single shared redactor, so a value
    seen in the envelope and again in the invoice body gets the same placeholder:

    1. Envelope — recipient/sender addresses, subject, routing headers.
    2. Content — postal addresses, emails, phones, card numbers, names in the body.
    """

    def __init__(self, redact_pii: bool = True, pii_policy: PIIPolicy | None = None):
        self.redact_pii = redact_pii
        self.pii_policy = pii_policy or PIIPolicy()

    def can_handle(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == ".eml"

    def parse(self, file_path: str) -> ParsedDocument:
        with open(file_path, "rb") as f:
            msg = BytesParser(policy=email_policy.default).parse(f)

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

        subject = msg.get("Subject")
        sender = msg.get("From")
        recipient = msg.get("To")
        metadata = {
            "message_id": msg.get("Message-ID"),
            "headers": dict(msg.items()),
        }

        if not self.redact_pii:
            return ParsedDocument(
                source_path=file_path,
                format="eml",
                subject=subject,
                sender=sender,
                recipient=recipient,
                date_received=msg.get("Date"),
                text_body=text_body,
                html_body=html_body,
                metadata=metadata,
            )

        redactor = PIIRedactor(self.pii_policy)

        # Learn identities first: recipient headers and body name cues feed the
        # name masking that both tiers then apply.
        for header in ("To", "Cc", "Bcc", "Delivered-To", "X-Original-To"):
            for value in msg.get_all(header, []):
                redactor.learn_recipient(str(value))
        redactor.learn_names_from_text(text_body)

        # Tier 1: envelope
        subject = redactor.redact_text(subject) if subject else subject
        sender = redactor.redact_sender(sender)
        recipient = redactor.redact_recipient(recipient)
        metadata = {"headers": redactor.redact_headers(msg.items())}

        # Tier 2: content
        text_body = redactor.redact_text(text_body)
        html_body = redactor.redact_text(html_body) if html_body else html_body

        return ParsedDocument(
            source_path=file_path,
            format="eml",
            subject=subject,
            sender=sender,
            recipient=recipient,
            date_received=msg.get("Date"),
            text_body=text_body,
            html_body=html_body,
            metadata=metadata,
            pii_redacted=True,
            redaction_report=redactor.report.to_dict(),
        )
