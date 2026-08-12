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


class EvaluationRunner:
    """Harbor-compatible evaluation runner for invoice extraction."""

    def __init__(
        self,
        invoices_dir: str = "invoices",
        annotations_dir: str = "annotations",
        results_dir: str = "results",
        model: str = "claude-sonnet-4-20250514",
    ):
        self.loader = DatasetLoader(invoices_dir, annotations_dir)
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

        output_path = self.results_dir / f"run_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_path}")
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
    args = parser.parse_args()

    runner = EvaluationRunner(
        invoices_dir=args.invoices_dir,
        annotations_dir=args.annotations_dir,
        results_dir=args.results_dir,
        model=args.model,
    )
    summary = runner.run(capture_rollouts=args.capture_rollouts)
    print(f"\nFinal Score: {summary['overall_mean']:.4f}")


if __name__ == "__main__":
    main()
