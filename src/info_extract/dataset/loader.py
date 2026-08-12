"""Dataset loader: reads annotations + parses source docs into tasks."""

from __future__ import annotations

import json
from pathlib import Path

from ..parsers.base import DocumentParser
from ..parsers.eml_parser import EmlParser
from ..schemas import InvoiceExtraction
from .tasks import ExtractionTask


class DatasetLoader:
    """Loads annotated tasks from disk."""

    def __init__(self, invoices_dir: str = "invoices", annotations_dir: str = "annotations"):
        self.invoices_dir = Path(invoices_dir)
        self.annotations_dir = Path(annotations_dir)
        self.parsers: list[DocumentParser] = [EmlParser()]

    def load_tasks(self) -> list[ExtractionTask]:
        tasks = []
        for annotation_file in sorted(self.annotations_dir.glob("*.json")):
            if annotation_file.name.startswith("_"):
                continue

            with open(annotation_file) as f:
                data = json.load(f)

            meta = data.get("_meta", {})
            source_rel = meta.get("source_file", "")
            source_file = self.invoices_dir / source_rel
            if not source_file.exists():
                print(f"Warning: source file not found: {source_file}")
                continue

            # Build ground truth from non-meta fields
            gt_data = {k: v for k, v in data.items() if not k.startswith("_")}
            ground_truth = InvoiceExtraction(**gt_data)

            # Parse the source document
            parser = self._find_parser(str(source_file))
            if parser is None:
                print(f"Warning: no parser for {source_file}")
                continue

            parsed_doc = parser.parse(str(source_file))

            tasks.append(
                ExtractionTask(
                    task_id=annotation_file.stem,
                    source_file=str(source_file),
                    parsed_document=parsed_doc,
                    ground_truth=ground_truth,
                )
            )

        return tasks

    def _find_parser(self, file_path: str) -> DocumentParser | None:
        for parser in self.parsers:
            if parser.can_handle(file_path):
                return parser
        return None
