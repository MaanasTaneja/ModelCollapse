#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write paper-ready artifact summaries into artifacts/paper.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--artifacts-root", type=str, default="artifacts")
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    paper_dir = artifacts_root / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    clean_stats = read_json(artifacts_root / "data" / "clean_stats.json")
    tail_tokens = read_json(artifacts_root / "data" / "tail_tokens.json")
    summary = read_json(artifacts_root / "runs" / "summary.json")

    methodology = {
        "dataset": {
            "clean_source": config["paths"]["clean_train_path"],
            "synthetic_source": config["paths"]["synth_path"],
            "tail_test_source": config["paths"]["tail_test_path"],
            "tail_token_source": config["paths"]["tail_tokens_path"],
        },
        "base_model": config["model"]["base_model_name"],
        "mixtures": config["experiment"]["mixtures"],
        "total_train_samples_per_run": config["experiment"]["total_train_samples"],
        "training": config["training"],
        "evaluation": config["evaluation"],
        "clean_stats": clean_stats,
        "tail_token_stats": {
            "selection_mode": None if tail_tokens is None else tail_tokens.get("selection_mode", "token_tail"),
            "tokenizer_name": None if tail_tokens is None else tail_tokens.get("tokenizer_name"),
            "tail_percentile": None if tail_tokens is None else tail_tokens.get("tail_percentile"),
            "num_tail_tokens": None if tail_tokens is None else tail_tokens.get("num_tail_tokens"),
            "tail_tokens_preview": None if tail_tokens is None else tail_tokens.get("tail_tokens_preview", [])[:30],
            "eval_languages": None if tail_tokens is None else tail_tokens.get("eval_languages"),
            "eval_language_counts": None if tail_tokens is None else tail_tokens.get("eval_language_counts"),
        },
    }

    results_section = {
        "status": "complete" if summary else "incomplete",
        "summary": summary,
    }

    manifest = {
        "data_files": sorted(str(path) for path in (artifacts_root / "data").glob("*")) if (artifacts_root / "data").exists() else [],
        "run_files": sorted(str(path) for path in (artifacts_root / "runs").glob("**/*") if path.is_file()) if (artifacts_root / "runs").exists() else [],
        "plot_files": sorted(str(path) for path in (artifacts_root / "plots").glob("*")) if (artifacts_root / "plots").exists() else [],
    }

    selection_mode = None if tail_tokens is None else tail_tokens.get("selection_mode", "token_tail")
    if selection_mode == "language_slice":
        eval_definition_md = f"""Rare-language evaluation slice:
- Selection mode: `language_slice`
- Eval languages: `{None if tail_tokens is None else tail_tokens.get("eval_languages")}`
- Eval language counts: `{None if tail_tokens is None else tail_tokens.get("eval_language_counts")}`
"""
        metric_bullets = """- Mean token log-probability on the held-out rare-language slice
- Perplexity on the held-out rare-language slice
"""
    else:
        eval_definition_md = f"""Tail-token definition:
- Tokenizer: `{None if tail_tokens is None else tail_tokens["tokenizer_name"]}`
- Tail percentile: `{None if tail_tokens is None else tail_tokens["tail_percentile"]}`
- Number of tail tokens: `{None if tail_tokens is None else tail_tokens["num_tail_tokens"]}`
"""
        metric_bullets = """- Mean tail-token log-probability
- Tail-only perplexity
"""

    methodology_md = f"""# Experiment Methodology

Base model: `{methodology["base_model"]}`

Clean corpus:
- Source: The Stack-derived clean corpus written to `{config["paths"]["clean_train_path"]}`
- Human clean train examples: `{None if clean_stats is None else clean_stats["num_train_examples"]}`
- Held-out tail test examples: `{None if clean_stats is None else clean_stats["num_eval_examples"]}`

{eval_definition_md}

Mixture conditions:
""" + "\n".join(
        f"- `{mix['name']}`: clean={mix['clean_ratio']}, synth={mix['synth_ratio']}"
        for mix in config["experiment"]["mixtures"]
    ) + f"""

Evaluation metrics:
{metric_bullets}
"""

    results_md = "# Results Summary\n\n"
    if summary:
        for row in summary:
            results_md += (
                f"- `{row['name']}`: "
                f"mean_tail_log_probability={row['mean_tail_log_probability']}, "
                f"tail_only_perplexity={row['tail_only_perplexity']}, "
                f"num_tail_positions={row['num_tail_positions']}\n"
            )
    else:
        results_md += "- Training/evaluation not finished yet.\n"

    (paper_dir / "methodology_summary.json").write_text(json.dumps(methodology, indent=2), encoding="utf-8")
    (paper_dir / "results_summary.json").write_text(json.dumps(results_section, indent=2), encoding="utf-8")
    (paper_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (paper_dir / "methodology_summary.md").write_text(methodology_md, encoding="utf-8")
    (paper_dir / "results_summary.md").write_text(results_md, encoding="utf-8")
    print(f"Wrote paper artifacts to {paper_dir}")


if __name__ == "__main__":
    main()
