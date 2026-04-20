#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_collapse.data_utils import (
    collect_hf_code,
    collect_local_code,
    deduplicate_examples,
    ensure_dir,
    sample_rows,
    select_tail_token_ids,
    split_clean_and_eval,
    split_clean_and_eval_by_language,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare clean code corpus and held-out tail-heavy test set.")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--dataset-data-dir", action="append", default=[])
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-column", type=str, default="code")
    parser.add_argument("--language-column", type=str, default=None)
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument(
        "--language-quota",
        action="append",
        default=[],
        help="Weighted language quota in the form Language=weight. When provided, examples are collected to match these weights.",
    )
    parser.add_argument("--input-glob", action="append", default=[])
    parser.add_argument("--tokenizer-name", type=str, default="distilgpt2")
    parser.add_argument("--tail-percentile", type=float, default=0.10)
    parser.add_argument("--min-tail-tokens-per-eval", type=int, default=5)
    parser.add_argument("--tail-eval-language", action="append", default=[])
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-eval-samples", type=int, default=800)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    if args.dataset_name:
        print(
            f"Collecting Hugging Face code from {args.dataset_name} "
            f"(split={args.dataset_split}, data_dirs={args.dataset_data_dir or ['<default>']})",
            flush=True,
        )
        raw_examples = collect_hf_code(
            dataset_name=args.dataset_name,
            config_name=args.dataset_config,
            data_dirs=args.dataset_data_dir or None,
            split=args.dataset_split,
            text_column=args.text_column,
            language_column=args.language_column,
            languages=args.languages,
            language_quota_specs=args.language_quota,
            max_samples=args.max_train_samples + args.max_eval_samples + 4000,
        )
    elif args.input_glob:
        print(f"Collecting local code from globs: {args.input_glob}", flush=True)
        raw_examples = collect_local_code(
            input_globs=args.input_glob,
            max_samples=args.max_train_samples + args.max_eval_samples + 4000,
        )
    else:
        raise SystemExit("Provide either --dataset-name or at least one --input-glob.")

    print(f"Collected {len(raw_examples)} raw examples", flush=True)
    examples = deduplicate_examples(raw_examples)
    print(f"Deduplicated down to {len(examples)} examples", flush=True)
    if args.tail_eval_language:
        print(
            f"Selecting held-out rare-language eval slice from: {args.tail_eval_language}",
            flush=True,
        )
        train_rows, eval_rows, tail_payload = split_clean_and_eval_by_language(
            examples=examples,
            tokenizer_name=args.tokenizer_name,
            eval_languages=args.tail_eval_language,
            max_eval_samples=args.max_eval_samples,
            seed=42,
        )
        tail_payload.update(
            {
                "tokenizer_name": args.tokenizer_name,
                "tail_percentile": None,
                "num_tail_tokens": 0,
                "tail_token_ids": [],
                "tail_tokens_preview": [],
            }
        )
        print(
            f"Selected {len(eval_rows)} eval rows across rare-language slice: {tail_payload['eval_language_counts']}",
            flush=True,
        )
    else:
        token_freq = Counter()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        print(f"Tokenizing examples with tokenizer={args.tokenizer_name}", flush=True)
        for idx, example in enumerate(examples, start=1):
            token_freq.update(tokenizer.encode(example.text, add_special_tokens=False))
            if idx % 500 == 0 or idx == len(examples):
                print(f"Tokenized {idx}/{len(examples)} examples", flush=True)

        tail_payload = select_tail_token_ids(
            token_freq=token_freq,
            tokenizer_name=args.tokenizer_name,
            tail_percentile=args.tail_percentile,
        )
        tail_token_ids = set(tail_payload["tail_token_ids"])
        print(
            f"Selected {tail_payload['num_tail_tokens']} tail tokens at percentile={args.tail_percentile}",
            flush=True,
        )

        train_rows, eval_rows = split_clean_and_eval(
            examples=examples,
            tokenizer_name=args.tokenizer_name,
            tail_token_ids=tail_token_ids,
            min_tail_tokens_per_eval=args.min_tail_tokens_per_eval,
            max_eval_samples=args.max_eval_samples,
        )
    print(
        f"Split into {len(train_rows)} train candidates and {len(eval_rows)} eval rows before train subsampling",
        flush=True,
    )

    train_rows = sample_rows(train_rows, args.max_train_samples, seed=42)
    print(f"Sampled {len(train_rows)} final train rows", flush=True)

    write_jsonl(output_dir / "clean_train.jsonl", train_rows)
    write_jsonl(output_dir / "tail_test.jsonl", eval_rows)
    write_json(output_dir / "tail_tokens.json", tail_payload)
    write_json(
        output_dir / "clean_stats.json",
        {
            "num_raw_examples": len(raw_examples),
            "num_deduped_examples": len(examples),
            "num_train_examples": len(train_rows),
            "num_eval_examples": len(eval_rows),
            "tail_percentile": args.tail_percentile,
            "min_tail_tokens_per_eval": args.min_tail_tokens_per_eval,
            "tokenizer_name": args.tokenizer_name,
            "selection_mode": tail_payload.get("selection_mode", "token_tail"),
            "eval_languages": tail_payload.get("eval_languages"),
            "eval_language_counts": tail_payload.get("eval_language_counts"),
        },
    )

    print(f"Wrote clean corpus artifacts to {output_dir}")


if __name__ == "__main__":
    main()
