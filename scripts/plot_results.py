#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from model_collapse.data_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tail metric trends across mixture runs.")
    parser.add_argument("--results-glob", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def infer_run_name(path: Path) -> str:
    return path.parent.name


def infer_synth_ratio(run_name: str) -> int:
    parts = run_name.split("_")
    for idx, part in enumerate(parts):
        if part == "synth" and idx > 0:
            return int(parts[idx - 1])
    raise ValueError(f"Could not infer synth ratio from {run_name}")


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    rows = []
    for match in glob.glob(args.results_glob):
        path = Path(match)
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_name = infer_run_name(path)
        rows.append(
            {
                "run_name": run_name,
                "synth_percent": infer_synth_ratio(run_name),
                "mean_tail_log_probability": payload["mean_tail_log_probability"],
                "tail_only_perplexity": payload["tail_only_perplexity"],
                "num_tail_positions": payload["num_tail_positions"],
            }
        )

    if not rows:
        raise SystemExit("No result files matched.")

    df = pd.DataFrame(rows).sort_values("synth_percent")
    df.to_csv(output_dir / "tail_metrics_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(df["synth_percent"], df["mean_tail_log_probability"], marker="o")
    axes[0].set_title("Tail Log-Probability")
    axes[0].set_xlabel("Synthetic Data (%)")
    axes[0].set_ylabel("Mean Log P(tail token)")

    axes[1].plot(df["synth_percent"], df["tail_only_perplexity"], marker="o")
    axes[1].set_title("Tail-Only Perplexity")
    axes[1].set_xlabel("Synthetic Data (%)")
    axes[1].set_ylabel("Perplexity")

    fig.tight_layout()
    fig.savefig(output_dir / "tail_metrics_trend.png", dpi=200)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
