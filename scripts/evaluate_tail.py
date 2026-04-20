#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_collapse.eval_utils import evaluate_tail_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tail-token metrics for a trained model.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tail-test-path", type=str, required=True)
    parser.add_argument("--tail-tokens-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Evaluating tail metrics for model={args.model_path} on tail_test={args.tail_test_path}",
        flush=True,
    )
    metrics = evaluate_tail_metrics(
        model_name_or_path=args.model_path,
        tail_test_path=args.tail_test_path,
        tail_tokens_path=args.tail_tokens_path,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
    )
    Path(args.output_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote tail metrics to {args.output_path}", flush=True)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
