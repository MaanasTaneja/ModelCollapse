#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from model_collapse.data_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all mixture training and tail evaluation jobs.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    print("Running:", " ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    seed = config["seed"]
    paths = config["paths"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    exp_cfg = config["experiment"]

    output_root = ensure_dir(paths["output_root"])
    summary = []

    total_mixtures = len(exp_cfg["mixtures"])
    for mixture_idx, mixture in enumerate(exp_cfg["mixtures"], start=1):
        run_dir = ensure_dir(output_root / mixture["name"])
        model_dir = run_dir / "final_model"
        print(
            f"=== Mixture {mixture_idx}/{total_mixtures}: {mixture['name']} "
            f"(clean={mixture['clean_ratio']}, synth={mixture['synth_ratio']}) ===",
            flush=True,
        )

        run_command(
            [
                sys.executable,
                "scripts/train_mixture.py",
                "--base-model-name",
                model_cfg["base_model_name"],
                "--clean-path",
                paths["clean_train_path"],
                "--synth-path",
                paths["synth_path"],
                "--output-dir",
                str(run_dir),
                "--clean-ratio",
                str(mixture["clean_ratio"]),
                "--synth-ratio",
                str(mixture["synth_ratio"]),
                "--total-train-samples",
                str(exp_cfg["total_train_samples"]),
                "--block-size",
                str(train_cfg["block_size"]),
                "--learning-rate",
                str(train_cfg["learning_rate"]),
                "--weight-decay",
                str(train_cfg["weight_decay"]),
                "--warmup-ratio",
                str(train_cfg["warmup_ratio"]),
                "--max-train-steps",
                str(train_cfg["max_train_steps"]),
                "--per-device-train-batch-size",
                str(train_cfg["per_device_train_batch_size"]),
                "--gradient-accumulation-steps",
                str(train_cfg["gradient_accumulation_steps"]),
                "--logging-steps",
                str(train_cfg["logging_steps"]),
                "--save-steps",
                str(train_cfg["save_steps"]),
                "--seed",
                str(seed),
            ]
        )

        metrics_path = run_dir / "tail_metrics.json"
        print(f"Starting evaluation for {mixture['name']}", flush=True)
        run_command(
            [
                sys.executable,
                "scripts/evaluate_tail.py",
                "--model-path",
                str(model_dir),
                "--tail-test-path",
                paths["tail_test_path"],
                "--tail-tokens-path",
                paths["tail_tokens_path"],
                "--batch-size",
                str(eval_cfg["batch_size"]),
                "--output-path",
                str(metrics_path),
            ]
        )

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary.append(
            {
                "name": mixture["name"],
                "clean_ratio": mixture["clean_ratio"],
                "synth_ratio": mixture["synth_ratio"],
                **metrics,
            }
        )
        print(f"Finished mixture {mixture['name']}", flush=True)

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote experiment summary to {summary_path}")


if __name__ == "__main__":
    main()
