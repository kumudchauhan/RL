"""Dataset loader: reads annotations + parses source docs into tasks."""

from __future__ import annotations

import json
from pathlib import Path

from ..parsers.base import DocumentParser
from ..parsers.eml_parser import EmlParser
from ..parsers.pdf_parser import PdfParser
from ..schemas import InvoiceExtraction, LineItem, PaymentInfo
from .tasks import ExtractionTask

#: Annotation keys the extraction schema deliberately no longer carries. Annotations are the
#: local ground truth and keep whatever the annotator recorded, so they are pruned here rather
#: than edited: nothing personal is loaded into memory, let alone written to a result file.
DROPPED_ANNOTATION_KEYS = frozenset({"shipping_address", "billing_address", "notes"})

#: Same, one level down: the card number is masked in the parser and has no schema field.
DROPPED_PAYMENT_KEYS = frozenset({"last_four"})


def prune_annotation(data: dict) -> tuple[dict, list[str]]:
    """Strip ``_meta`` and retired/PII keys from raw annotation data.

    Returns the ground-truth payload and the names (never the values) of what was dropped.
    """
    dropped: list[str] = []
    pruned: dict = {}

    for key, value in data.items():
        if key.startswith("_"):
            continue
        if key in DROPPED_ANNOTATION_KEYS:
            dropped.append(key)
            continue
        if key == "payment" and isinstance(value, dict):
            dropped.extend(f"payment.{k}" for k in value if k in DROPPED_PAYMENT_KEYS)
            pruned[key] = {k: v for k, v in value.items() if k not in DROPPED_PAYMENT_KEYS}
            continue
        pruned[key] = value

    return pruned, dropped


def unexpected_keys(data: dict) -> list[str]:
    """Annotation keys the schema neither accepts nor knowingly drops.

    Reported by name only, so a stale annotation fails loudly without echoing its contents.
    """
    unknown = [key for key in data if key not in InvoiceExtraction.model_fields]

    payment = data.get("payment")
    if isinstance(payment, dict):
        unknown.extend(
            f"payment.{key}" for key in payment if key not in PaymentInfo.model_fields
        )

    for index, item in enumerate(data.get("line_items") or []):
        if isinstance(item, dict):
            unknown.extend(
                f"line_items[{index}].{key}"
                for key in item
                if key not in LineItem.model_fields
            )

    return unknown


class DatasetLoader:
    """Loads annotated tasks from disk."""

    def __init__(
        self,
        invoices_dir: str = "invoices",
        annotations_dir: str = "annotations",
        redact_pii: bool = True,
    ):
        self.invoices_dir = Path(invoices_dir)
        self.annotations_dir = Path(annotations_dir)
        self.redact_pii = redact_pii
        self.parsers: list[DocumentParser] = [
            EmlParser(redact_pii=redact_pii),
            PdfParser(redact_pii=redact_pii),
        ]

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

            # Build ground truth from the fields the schema still carries
            gt_data, dropped = prune_annotation(data)
            unknown = unexpected_keys(gt_data)
            if unknown:
                raise ValueError(
                    f"{annotation_file.name}: unrecognised annotation field(s): "
                    f"{', '.join(sorted(unknown))}. Add them to the schema, or to "
                    f"DROPPED_ANNOTATION_KEYS if they must stay out of the output."
                )
            if dropped:
                print(f"  {annotation_file.name}: not loaded: {', '.join(sorted(dropped))}")
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
