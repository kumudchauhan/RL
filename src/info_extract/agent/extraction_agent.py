"""Claude API wrapper using tool-use for structured extraction."""

from __future__ import annotations

import anthropic

from ..parsers.base import ParsedDocument
from ..schemas import InvoiceExtraction
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


class ExtractionAgent:
    """Harbor-compatible agent that uses Claude for invoice extraction."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.0,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature
        self._tool_schema = {
            "name": "submit_extraction",
            "description": "Submit the extracted invoice data",
            "input_schema": InvoiceExtraction.model_json_schema(),
        }

    def _build_user_message(self, document: ParsedDocument) -> str:
        return USER_PROMPT_TEMPLATE.format(
            source=document.source_path,
            format=document.format,
            subject=document.subject or "N/A",
            sender=document.sender or "N/A",
            document_text=document.text_body,
        )

    def extract(self, document: ParsedDocument) -> InvoiceExtraction:
        """Run extraction on a parsed document. Returns structured output."""
        user_message = self._build_user_message(document)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            tools=[self._tool_schema],
            tool_choice={"type": "tool", "name": "submit_extraction"},
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")
        return InvoiceExtraction(**tool_block.input)

    def extract_with_rollout(self, document: ParsedDocument) -> dict:
        """Extract with full rollout capture for RLVR training data generation."""
        user_message = self._build_user_message(document)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            tools=[self._tool_schema],
            tool_choice={"type": "tool", "name": "submit_extraction"},
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")
        extraction = InvoiceExtraction(**tool_block.input)

        return {
            "input": user_message,
            "system": SYSTEM_PROMPT,
            "output": extraction.model_dump(),
            "raw_response": response.model_dump(),
            "model": self.model,
            "temperature": self.temperature,
        }
