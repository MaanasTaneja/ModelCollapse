#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_collapse.data_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect prepared experiment artifacts before training.")
    parser.add_argument("--artifacts-root", type=str, default="artifacts")
    parser.add_argument("--sample-count", type=int, default=3)
    return parser.parse_args()


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def preview_rows(rows: list[dict], sample_count: int) -> None:
    for idx, row in enumerate(rows[:sample_count], start=1):
        text = row.get("text", "")
        text = text[:400].replace("\n", "\\n")
        print(f"[sample {idx}] language={row.get('language')} tail_token_count={row.get('tail_token_count')} text={text}")


def main() -> None:
    args = parse_args()
    root = Path(args.artifacts_root)
    data_dir = root / "data"

    clean_stats_path = data_dir / "clean_stats.json"
    tail_tokens_path = data_dir / "tail_tokens.json"
    clean_train_path = data_dir / "clean_train.jsonl"
    tail_test_path = data_dir / "tail_test.jsonl"
    synth_path = data_dir / "synth_head.jsonl"

    print_section("Files")
    for path in [clean_stats_path, tail_tokens_path, clean_train_path, tail_test_path, synth_path]:
        print(f"{path}: {'present' if path.exists() else 'missing'}")

    if clean_stats_path.exists():
        print_section("Clean Stats")
        print(json.dumps(json.loads(clean_stats_path.read_text(encoding="utf-8")), indent=2))

    if tail_tokens_path.exists():
        tail_payload = json.loads(tail_tokens_path.read_text(encoding="utf-8"))
        print_section("Tail Tokens")
        print(f"tokenizer={tail_payload['tokenizer_name']}")
        print(f"tail_percentile={tail_payload['tail_percentile']}")
        print(f"num_tail_tokens={tail_payload['num_tail_tokens']}")
        print("preview:")
        for item in tail_payload["tail_tokens_preview"][:10]:
            print(f"  id={item['id']} token={item['token']} count={item['count']}")

    if clean_train_path.exists():
        clean_rows = read_jsonl(clean_train_path)
        print_section("Clean Train")
        print(f"num_rows={len(clean_rows)}")
        lang_counts = Counter(row.get("language") for row in clean_rows)
        print(f"top_languages={lang_counts.most_common(10)}")
        preview_rows(clean_rows, args.sample_count)

    if tail_test_path.exists():
        tail_rows = read_jsonl(tail_test_path)
        print_section("Tail Test")
        print(f"num_rows={len(tail_rows)}")
        avg_tail = sum(row.get("tail_token_count", 0) for row in tail_rows) / max(len(tail_rows), 1)
        print(f"avg_tail_tokens_per_sample={avg_tail:.2f}")
        preview_rows(tail_rows, args.sample_count)

    if synth_path.exists():
        synth_rows = read_jsonl(synth_path)
        print_section("Synthetic Head")
        print(f"num_rows={len(synth_rows)}")
        lang_counts = Counter(row.get("language") for row in synth_rows)
        gen_counts = Counter(row.get("generator") for row in synth_rows)
        print(f"languages={lang_counts.most_common(10)}")
        print(f"generators={gen_counts.most_common(10)}")
        preview_rows(synth_rows, args.sample_count)


if __name__ == "__main__":
    main()
