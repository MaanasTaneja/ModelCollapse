from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_collapse.data_utils import read_jsonl


def load_tail_token_ids(path: str | Path) -> set[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(payload["tail_token_ids"])


def load_eval_selection(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collate_for_causal_lm(batch, tokenizer, max_length: int = 256):
    texts = [row["text"] for row in batch]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded["labels"] = encoded["input_ids"].clone()
    return encoded


@torch.no_grad()
def evaluate_tail_metrics(
    model_name_or_path: str,
    tail_test_path: str | Path,
    tail_tokens_path: str | Path,
    batch_size: int = 2,
    max_eval_samples: int | None = None,
    device: str | None = None,
) -> dict:
    rows = read_jsonl(tail_test_path)
    if max_eval_samples is not None:
        rows = rows[:max_eval_samples]
    print(f"Loaded {len(rows)} eval rows", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    selection_payload = load_eval_selection(tail_tokens_path)
    selection_mode = selection_payload.get("selection_mode", "token_tail")
    tail_token_ids = set(selection_payload.get("tail_token_ids", []))
    dataloader = DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_for_causal_lm(batch, tokenizer),
    )

    tail_log_probs: list[float] = []
    tail_nll_sum = 0.0
    tail_count = 0

    total_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        target_mask = attention_mask[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        if selection_mode == "language_slice":
            tail_mask = target_mask.bool()
        else:
            tail_mask = torch.zeros_like(target_ids, dtype=torch.bool)
            for tail_id in tail_token_ids:
                tail_mask |= target_ids.eq(tail_id)
            tail_mask &= target_mask.bool()

        if tail_mask.any():
            selected = gathered[tail_mask]
            tail_log_probs.extend(selected.detach().cpu().tolist())
            tail_nll_sum += float((-selected).sum().item())
            tail_count += int(tail_mask.sum().item())
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            print(
                f"Evaluated {batch_idx}/{total_batches} batches; observed {tail_count} tail positions so far",
                flush=True,
            )

    mean_log_prob = sum(tail_log_probs) / len(tail_log_probs) if tail_log_probs else float("nan")
    tail_perplexity = math.exp(tail_nll_sum / tail_count) if tail_count else float("nan")

    return {
        "model_name_or_path": model_name_or_path,
        "selection_mode": selection_mode,
        "num_eval_samples": len(rows),
        "num_tail_positions": tail_count,
        "num_eval_positions": tail_count,
        "mean_eval_log_probability": mean_log_prob,
        "mean_tail_log_probability": mean_log_prob,
        "eval_only_perplexity": tail_perplexity,
        "tail_only_perplexity": tail_perplexity,
    }
