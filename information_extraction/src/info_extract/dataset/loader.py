"""Dataset loader: reads annotations + parses source docs into tasks."""

from __future__ import annotations

import json
from pathlib import Path

from ..parsers.base import DocumentParser
from ..parsers.eml_parser import EmlParser
from ..parsers.pdf_parser import PdfParser
from ..schemas import Discount, Fee, InvoiceExtraction, LineItem, Payment
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
        if key == "payments" and isinstance(value, list):
            cleaned = []
            for index, entry in enumerate(value):
                if not isinstance(entry, dict):
                    cleaned.append(entry)
                    continue
                dropped.extend(
                    f"payments[{index}].{k}" for k in entry if k in DROPPED_PAYMENT_KEYS
                )
                cleaned.append({k: v for k, v in entry.items() if k not in DROPPED_PAYMENT_KEYS})
            pruned[key] = cleaned
            continue
        pruned[key] = value

    return pruned, dropped


def migrate_annotation(data: dict) -> tuple[dict, list[str]]:
    """Bring an older annotation up to the current schema shape, losslessly.

    An order can be split across tenders (gift card + card, instalments), so ``payment`` became
    the list ``payments``. Annotations are not edited for that: a single ``payment`` object is
    lifted into a one-entry list here, and the rename is reported so the run says what it did.
    """
    migrated = dict(data)
    notes: list[str] = []

    payment = migrated.pop("payment", None)
    if isinstance(payment, dict) and payment:
        existing = migrated.get("payments") or []
        migrated["payments"] = [payment, *existing]
        notes.append("payment -> payments[0]")

    return migrated, notes


def unexpected_keys(data: dict) -> list[str]:
    """Annotation keys the schema neither accepts nor knowingly drops.

    Reported by name only, so a stale annotation fails loudly without echoing its contents.
    """
    unknown = [key for key in data if key not in InvoiceExtraction.model_fields]

    nested_lists = (
        ("payments", Payment),
        ("fees", Fee),
        ("discounts", Discount),
        ("line_items", LineItem),
    )
    for field_name, model in nested_lists:
        for index, entry in enumerate(data.get(field_name) or []):
            if isinstance(entry, dict):
                unknown.extend(
                    f"{field_name}[{index}].{key}"
                    for key in entry
                    if key not in model.model_fields
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
            gt_data, migrations = migrate_annotation(gt_data)
            unknown = unexpected_keys(gt_data)
            if unknown:
                raise ValueError(
                    f"{annotation_file.name}: unrecognised annotation field(s): "
                    f"{', '.join(sorted(unknown))}. Add them to the schema, or to "
                    f"DROPPED_ANNOTATION_KEYS if they must stay out of the output."
                )
            if dropped:
                print(f"  {annotation_file.name}: not loaded: {', '.join(sorted(dropped))}")
            if migrations:
                print(f"  {annotation_file.name}: migrated: {', '.join(sorted(migrations))}")
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
