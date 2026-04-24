#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from model_collapse.data_utils import ensure_dir, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one clean/synthetic mixture run.")
    parser.add_argument("--base-model-name", type=str, default="distilgpt2")
    parser.add_argument("--clean-path", type=str, required=True)
    parser.add_argument("--synth-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--clean-ratio", type=float, required=True)
    parser.add_argument("--synth-ratio", type=float, required=True)
    parser.add_argument("--total-train-samples", type=int, default=9000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-train-steps", type=int, default=800)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if not rows:
        return []
    if n <= len(rows):
        return rows[:n]
    repeats = (n // len(rows)) + 1
    expanded = (rows * repeats)[:n]
    random.Random(seed).shuffle(expanded)
    return expanded


def tokenize_and_chunk(dataset: Dataset, tokenizer, block_size: int) -> Dataset:
    def tokenize_batch(batch):
        return tokenizer(batch["text"])

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)

    def group_texts(batch):
        concatenated = {k: sum(batch[k], []) for k in batch.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = list(result["input_ids"])
        return result

    return tokenized.map(group_texts, batched=True)


def main() -> None:
    args = parse_args()
    if abs((args.clean_ratio + args.synth_ratio) - 1.0) > 1e-6:
        raise SystemExit("clean_ratio + synth_ratio must equal 1.0")

    output_dir = ensure_dir(args.output_dir)
    clean_rows = read_jsonl(args.clean_path)
    synth_rows = read_jsonl(args.synth_path)

    clean_n = int(round(args.total_train_samples * args.clean_ratio))
    synth_n = args.total_train_samples - clean_n
    print(
        f"Preparing mixture dataset: clean={clean_n}, synth={synth_n}, total={args.total_train_samples}",
        flush=True,
    )

    mixture_rows = sample_rows(clean_rows, clean_n, args.seed) + sample_rows(synth_rows, synth_n, args.seed + 1)
    random.Random(args.seed).shuffle(mixture_rows)
    print(f"Loaded mixture rows: {len(mixture_rows)}", flush=True)

    dataset = Dataset.from_list([{"text": row["text"]} for row in mixture_rows])
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    lm_dataset = tokenize_and_chunk(dataset, tokenizer, args.block_size)
    print(f"Chunked training dataset into {len(lm_dataset)} blocks of size {args.block_size}", flush=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_train_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=collator,
    )
    print(
        f"Starting training for {args.max_train_steps} steps "
        f"(logging every {args.logging_steps}, no intermediate checkpoints)",
        flush=True,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    Path(output_dir / "mixture_metadata.json").write_text(
        json.dumps(
            {
                "base_model_name": args.base_model_name,
                "clean_ratio": args.clean_ratio,
                "synth_ratio": args.synth_ratio,
                "clean_samples_used": clean_n,
                "synth_samples_used": synth_n,
                "total_train_samples": len(mixture_rows),
                "block_size": args.block_size,
                "max_train_steps": args.max_train_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Finished training mixture run in {output_dir}")


if __name__ == "__main__":
    main()
