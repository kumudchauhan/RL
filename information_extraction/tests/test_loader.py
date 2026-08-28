"""Tests for the dataset loader, focused on what it refuses to load.

Annotations are the local ground truth and keep whatever the annotator recorded — including the
fields the extraction schema deliberately dropped. The loader is the boundary that stops those
from reaching a task, a result file, or a captured rollout.
"""

import json

import pytest

from info_extract.dataset.loader import DatasetLoader, prune_annotation, unexpected_keys

EML_SOURCE = (
    "From: Store <orders@vendor.example>\r\n"
    "To: Jane Roe <jane.roe@example.com>\r\n"
    "Subject: Your order 1000200-12345678\r\n"
    "Content-Type: text/plain; charset=utf-8\r\n"
    "\r\n"
    "Order total: $19.99\r\n"
)

ANNOTATION = {
    "_meta": {"source_file": "sample.eml", "annotator": "test", "date": "2026-08-28"},
    "vendor": "Store",
    "order_id": "1000200-12345678",
    "total": 19.99,
    "currency": "USD",
    "line_items": [{"product_name": "Sparkling Water", "total_price": 19.99, "upc": "012345678905"}],
    "payment": {"method": "credit_card", "card_type": "Visa", "last_four": "4321"},
    "shipping_address": {"street": "123 Main St", "city": "Springfield", "state": "IL"},
    "billing_address": {"street": "123 Main St", "city": "Springfield", "state": "IL"},
    "notes": "Delivered to Jane Roe at 123 Main St",
}


class TestPruneAnnotation:
    def test_drops_pii_and_retired_keys(self):
        pruned, dropped = prune_annotation(ANNOTATION)
        assert "shipping_address" not in pruned
        assert "billing_address" not in pruned
        assert "notes" not in pruned
        assert "last_four" not in pruned["payment"]
        assert sorted(dropped) == [
            "billing_address",
            "notes",
            "payment.last_four",
            "shipping_address",
        ]

    def test_reports_names_only(self):
        """A dropped field's value must not travel with the report of dropping it."""
        _, dropped = prune_annotation(ANNOTATION)
        assert all("Jane" not in name and "Main" not in name for name in dropped)

    def test_keeps_everything_the_schema_carries(self):
        pruned, _ = prune_annotation(ANNOTATION)
        assert pruned["vendor"] == "Store"
        assert pruned["payment"] == {"method": "credit_card", "card_type": "Visa"}
        assert pruned["line_items"][0]["upc"] == "012345678905"

    def test_meta_is_not_ground_truth(self):
        pruned, dropped = prune_annotation(ANNOTATION)
        assert "_meta" not in pruned
        assert "_meta" not in dropped


class TestUnexpectedKeys:
    def test_clean_annotation_has_none(self):
        pruned, _ = prune_annotation(ANNOTATION)
        assert unexpected_keys(pruned) == []

    def test_reports_nested_paths(self):
        data = {
            "vendor": "Store",
            "total": 1.0,
            "mystery": 1,
            "payment": {"method": "cash", "wallet_id": "x"},
            "line_items": [{"product_name": "Thing", "shelf": "A3"}],
        }
        assert sorted(unexpected_keys(data)) == [
            "line_items[0].shelf",
            "mystery",
            "payment.wallet_id",
        ]


class TestLoadTasks:
    @staticmethod
    def _write_corpus(tmp_path, annotation: dict):
        invoices, annotations = tmp_path / "invoices", tmp_path / "annotations"
        invoices.mkdir()
        annotations.mkdir()
        (invoices / "sample.eml").write_text(EML_SOURCE)
        (annotations / "sample.json").write_text(json.dumps(annotation))
        return DatasetLoader(str(invoices), str(annotations))

    def test_ground_truth_carries_no_dropped_fields(self, tmp_path):
        tasks = self._write_corpus(tmp_path, ANNOTATION).load_tasks()
        assert len(tasks) == 1

        dumped = tasks[0].ground_truth.model_dump()
        assert set(dumped) & {"shipping_address", "billing_address", "notes"} == set()
        assert "last_four" not in dumped["payment"]
        serialized = json.dumps(dumped)
        assert "Jane" not in serialized
        assert "123 Main St" not in serialized
        assert "4321" not in serialized

    def test_new_line_item_detail_loads(self, tmp_path):
        tasks = self._write_corpus(tmp_path, ANNOTATION).load_tasks()
        assert tasks[0].ground_truth.line_items[0].upc == "012345678905"

    def test_template_annotations_are_skipped(self, tmp_path):
        loader = self._write_corpus(tmp_path, ANNOTATION)
        (loader.annotations_dir / "_template.json").write_text("{}")
        assert len(loader.load_tasks()) == 1

    def test_unrecognised_field_fails_loudly(self, tmp_path):
        loader = self._write_corpus(tmp_path, {**ANNOTATION, "loyalty_number": "12345"})
        with pytest.raises(ValueError, match="loyalty_number"):
            loader.load_tasks()

    def test_harbor_task_input_is_redacted(self, tmp_path):
        tasks = self._write_corpus(tmp_path, ANNOTATION).load_tasks()
        payload = json.dumps(tasks[0].to_harbor_task())
        assert "jane.roe@example.com" not in payload
        assert "1000200-12345678" in payload  # order ids are extraction targets, not PII
