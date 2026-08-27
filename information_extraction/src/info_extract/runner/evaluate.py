"""Main evaluation harness: load tasks -> run agent -> compute rewards."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from ..agent.extraction_agent import ExtractionAgent
from ..dataset.loader import DatasetLoader
from ..schemas import InvoiceExtraction
from ..verifiers.composite import CompositeVerifier

EQUAL_WEIGHTS = {"field_accuracy": 1 / 3, "numeric_accuracy": 1 / 3, "line_item_f1": 1 / 3}


class EvaluationRunner:
    """Harbor-compatible evaluation runner for invoice extraction."""

    def __init__(
        self,
        invoices_dir: str = "invoices",
        annotations_dir: str = "annotations",
        results_dir: str = "results",
        model: str = "claude-sonnet-4-20250514",
        redact_pii: bool = True,
    ):
        self.loader = DatasetLoader(invoices_dir, annotations_dir, redact_pii=redact_pii)
        self.agent = ExtractionAgent(model=model)
        self.verifier = CompositeVerifier()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def run(self, capture_rollouts: bool = False) -> dict:
        """Run full evaluation. Returns summary metrics."""
        tasks = self.loader.load_tasks()
        if not tasks:
            print("No tasks found. Check annotations directory.")
            return {"overall_mean": 0, "num_evaluated": 0}

        results = []

        for task in tasks:
            print(f"Processing: {task.task_id}")
            start = time.time()

            try:
                if capture_rollouts:
                    rollout = self.agent.extract_with_rollout(task.parsed_document)
                    prediction = InvoiceExtraction(**rollout["output"])
                else:
                    prediction = self.agent.extract(task.parsed_document)
                    rollout = None
            except (anthropic.APIError, ValueError, KeyError) as e:
                print(f"  ERROR: {e}")
                continue

            elapsed = time.time() - start

            reward = self.verifier.compute_reward(prediction, task.ground_truth)

            result = {
                "task_id": task.task_id,
                "source_file": task.source_file,
                "prediction": prediction.model_dump(),
                "ground_truth": task.ground_truth.model_dump(),
                "reward": reward.to_harbor_reward(),
                "elapsed_seconds": elapsed,
            }
            if rollout:
                result["rollout"] = rollout

            results.append(result)
            print(
                f"  Reward: {reward.overall_reward:.3f} "
                f"(fields={reward.component_rewards.get('field_accuracy', 0):.3f}, "
                f"numeric={reward.component_rewards.get('numeric_accuracy', 0):.3f}, "
                f"items={reward.component_rewards.get('line_item_f1', 0):.3f})"
            )

        summary = self._compute_summary(results)

        output = {
            "run_timestamp": datetime.now(tz=UTC).isoformat(),
            "model": self.agent.model,
            "num_tasks": len(tasks),
            "summary": summary,
            "results": results,
        }

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"run_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")

        report = self._generate_report(results, output)
        report_path = self.results_dir / f"report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(report)
        print(f"\nReport saved to: {report_path}")

        return summary

    def _compute_summary(self, results: list[dict]) -> dict:
        if not results:
            return {"overall_mean": 0, "num_evaluated": 0}

        rewards = [r["reward"]["reward"] for r in results]
        components: dict[str, list[float]] = {}
        for r in results:
            for k, v in r["reward"]["components"].items():
                components.setdefault(k, []).append(v)

        return {
            "overall_mean": sum(rewards) / len(rewards),
            "overall_min": min(rewards),
            "overall_max": max(rewards),
            "component_means": {k: sum(v) / len(v) for k, v in components.items()},
            "num_evaluated": len(results),
        }

    def _generate_report(self, results: list[dict], run_output: dict) -> str:
        """Generate a human-readable summary report comparing equal vs weighted scoring."""
        lines: list[str] = []
        w = lines.append

        actual_weights = self.verifier.weights
        weight_str = "/".join(f"{actual_weights[k]:.2f}" for k in ["field_accuracy", "numeric_accuracy", "line_item_f1"])

        w("")
        w("=" * 60)
        w("  EXTRACTION EVALUATION REPORT")
        w(f"  Model: {run_output['model']}")
        w(f"  Date:  {run_output['run_timestamp']}")
        w(f"  Tasks evaluated: {len(results)}")
        w("=" * 60)

        # --- Per-task breakdown ---
        for result in results:
            task_id = result["task_id"]
            metadata = result["reward"]["metadata"]

            w("")
            w(f"-- Task: {task_id} " + "-" * max(0, 45 - len(task_id)))
            w("")

            # Field accuracy details
            field_meta = metadata.get("field_accuracy", {})
            field_details = field_meta.get("details", {})
            if field_details:
                w("  Field Accuracy:")
                for fname, info in field_details.items():
                    score = info["score"]
                    pred = info.get("pred")
                    mark = "+" if score >= 0.85 else ("~" if score > 0 else "x")
                    pred_display = f'"{pred}"' if isinstance(pred, str) else str(pred)
                    w(f"    {mark} {fname:<16} {score:.2f}  (pred: {pred_display})")
                w("")

            # Numeric accuracy details
            num_meta = metadata.get("numeric_accuracy", {})
            num_details = num_meta.get("details", {})
            if num_details:
                w("  Numeric Accuracy:")
                for fname, info in num_details.items():
                    score = info["score"]
                    pred = info.get("pred")
                    expected = info.get("expected")
                    mark = "+" if score >= 0.99 else ("~" if score > 0 else "x")
                    pred_fmt = f"${pred:.2f}" if pred is not None else "None"
                    exp_fmt = f"${expected:.2f}" if expected is not None else "None"
                    w(f"    {mark} {fname:<16} {score:.2f}  (pred: {pred_fmt}, expected: {exp_fmt})")
                w("")

            # Line item details
            li_meta = metadata.get("line_item_f1", {})
            li_details = li_meta.get("details", {})
            if li_details and "f1" in li_details:
                num_pred = li_details.get("num_predicted", 0)
                num_gt = li_details.get("num_ground_truth", 0)
                num_matched = li_details.get("num_matched", 0)
                f1 = li_details.get("f1", 0)
                w(f"  Line Items: {num_matched}/{num_gt} matched  (F1: {f1:.3f})")
                matches = li_details.get("matches", [])
                gt_items = result.get("ground_truth", {}).get("line_items", [])
                for m in matches:
                    truth_idx = m["truth_idx"]
                    mscore = m["score"]
                    item_name = gt_items[truth_idx]["product_name"] if truth_idx < len(gt_items) else "?"
                    # Truncate long names
                    if len(item_name) > 35:
                        item_name = item_name[:32] + "..."
                    mark = "+" if mscore >= 0.8 else "~"
                    w(f"    {mark} {item_name:<38} (score: {mscore:.2f})")
                if num_pred > num_matched:
                    w(f"    x {num_pred - num_matched} unmatched prediction(s)")
                if num_gt > num_matched:
                    w(f"    x {num_gt - num_matched} missing ground truth item(s)")
                w("")

        # --- Comparison table ---
        w("")
        w("=" * 60)
        w("  COMPARISON: Equal Weight vs Weighted Scoring")
        w("=" * 60)
        w("")
        w(f"  {'Task':<28} {'Equal (0.33 each)':<20} {'Weighted (' + weight_str + ')':<20}")
        w(f"  {'-'*28} {'-'*20} {'-'*20}")

        equal_scores = []
        weighted_scores = []

        for result in results:
            task_id = result["task_id"]
            components = result["reward"]["components"]

            weighted = result["reward"]["reward"]
            equal = sum(
                components.get(k, 0.0) * EQUAL_WEIGHTS[k]
                for k in EQUAL_WEIGHTS
            )

            equal_scores.append(equal)
            weighted_scores.append(weighted)

            display_id = task_id if len(task_id) <= 28 else task_id[:25] + "..."
            w(f"  {display_id:<28} {equal:>8.4f}             {weighted:>8.4f}")

        w(f"  {'-'*28} {'-'*20} {'-'*20}")

        mean_equal = sum(equal_scores) / len(equal_scores) if equal_scores else 0
        mean_weighted = sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0
        w(f"  {'MEAN':<28} {mean_equal:>8.4f}             {mean_weighted:>8.4f}")
        w("")

        diff = mean_equal - mean_weighted
        if abs(diff) > 0.001:
            direction = "inflates" if diff > 0 else "deflates"
            w(f"  Equal weighting {direction} the score by {abs(diff):.4f} vs weighted.")
        w("")

        # --- Explanation ---
        w("  WHY WEIGHTED SCORING MATTERS:")
        w("  " + "-" * 56)
        w("  Not all fields are equally difficult to extract. Vendor names and")
        w("  dates are relatively straightforward (high baseline accuracy), while")
        w("  line items require matching multiple sub-fields (name, quantity, price)")
        w("  and monetary totals demand exact-to-the-cent precision.")
        w("")
        w("  Equal weighting over-credits easy fields, inflating the overall score.")
        w("  Weighted scoring reflects real-world extraction difficulty:")
        w("")
        w(f"    field_accuracy  (w={actual_weights['field_accuracy']:.2f}) - easiest: vendor, dates, IDs")
        w(f"    numeric_accuracy(w={actual_weights['numeric_accuracy']:.2f}) - harder: must be exact to the cent")
        w(f"    line_item_f1    (w={actual_weights['line_item_f1']:.2f}) - hardest: multi-field matching")
        w("")

        return "\n".join(lines)


def main():
    """CLI entrypoint for extract-eval."""
    import argparse

    parser = argparse.ArgumentParser(description="Run invoice extraction evaluation")
    parser.add_argument("--invoices-dir", default="invoices")
    parser.add_argument("--annotations-dir", default="annotations")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument(
        "--capture-rollouts",
        action="store_true",
        help="Capture full rollouts for RLVR training",
    )
    parser.add_argument(
        "--no-pii-redaction",
        dest="redact_pii",
        action="store_false",
        help="Disable PII masking (sends raw document text to the API — debugging only)",
    )
    args = parser.parse_args()

    if not args.redact_pii:
        print("WARNING: PII redaction disabled — raw document text will be sent to the API.")

    runner = EvaluationRunner(
        invoices_dir=args.invoices_dir,
        annotations_dir=args.annotations_dir,
        results_dir=args.results_dir,
        model=args.model,
        redact_pii=args.redact_pii,
    )
    summary = runner.run(capture_rollouts=args.capture_rollouts)
    print(f"\nFinal Score: {summary['overall_mean']:.4f}")


if __name__ == "__main__":
    main()
