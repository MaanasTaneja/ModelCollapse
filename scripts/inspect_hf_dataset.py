#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Hugging Face dataset configs, splits, and sample columns.")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--dataset-data-dir", type=str, default=None)
    parser.add_argument("--sample-split", type=str, default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict = {"dataset_name": args.dataset_name}
    try:
        payload["configs"] = get_dataset_config_names(args.dataset_name)
    except Exception as exc:
        payload["configs_error"] = str(exc)

    try:
        payload["splits"] = get_dataset_split_names(args.dataset_name, args.dataset_config)
    except Exception as exc:
        payload["splits_error"] = str(exc)

    try:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            data_dir=args.dataset_data_dir,
            split=args.sample_split,
            streaming=True,
        )
        first = next(iter(dataset))
        payload["sample_columns"] = list(first.keys())
        payload["sample_record_preview"] = {
            key: (value[:200] + "..." if isinstance(value, str) and len(value) > 200 else value)
            for key, value in first.items()
        }
    except Exception as exc:
        payload["sample_error"] = str(exc)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
