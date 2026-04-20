#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_collapse.data_utils import write_jsonl
from model_collapse.synth_templates import (
    JAVASCRIPT_TASKS,
    PYTHON_TASKS,
    SHELL_TASKS,
    build_template_corpus,
)


def clean_generated_code(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic head-domain code corpus.")
    parser.add_argument("--backend", choices=["template", "hf", "openai"], default="openai")
    parser.add_argument("--generator-model", type=str, default="distilgpt2")
    parser.add_argument("--target-samples", type=int, default=12000)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_progress(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def maybe_report_progress(
    *,
    rows: list[dict],
    target_samples: int,
    progress_every: int,
    progress_path: Path,
    backend: str,
    generator_model: str,
    started_at: float,
    force: bool = False,
) -> None:
    if not rows:
        return
    if not force and len(rows) % progress_every != 0 and len(rows) != target_samples:
        return
    elapsed = max(time.time() - started_at, 1e-9)
    rate = len(rows) / elapsed
    remaining = max(target_samples - len(rows), 0)
    eta_seconds = remaining / rate if rate > 0 else None
    payload = {
        "backend": backend,
        "generator_model": generator_model,
        "target_samples": target_samples,
        "samples_written": len(rows),
        "percent_complete": round((len(rows) / target_samples) * 100, 2) if target_samples else 100.0,
        "elapsed_seconds": round(elapsed, 2),
        "samples_per_second": round(rate, 4),
        "eta_seconds": None if eta_seconds is None else round(eta_seconds, 2),
        "output_path": str(progress_path.with_suffix("")),
        "updated_at_epoch": round(time.time(), 3),
    }
    write_progress(progress_path, payload)
    eta_text = "unknown" if eta_seconds is None else f"{eta_seconds / 60:.1f} min"
    print(
        f"[progress] synthetic generation: {len(rows)}/{target_samples} "
        f"({payload['percent_complete']:.2f}%) at {rate:.2f} samples/s, ETA {eta_text}",
        flush=True,
    )


def build_prompts() -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    python_suffixes = [
        "Include input validation and structured error handling.",
        "Use clear helper functions and keep the code production-style but concise.",
        "Include basic logging and sensible defaults.",
        "Handle edge cases like empty input, missing files, or network failures.",
    ]
    javascript_suffixes = [
        "Keep the component or service modular and include loading and error states.",
        "Use realistic event handling and API integration patterns.",
        "Include form validation or user feedback where it makes sense.",
        "Structure the code as if it belongs in a real frontend or Node service.",
    ]
    shell_suffixes = [
        "Use strict bash mode and print actionable status messages.",
        "Handle missing paths or commands gracefully.",
        "Include argument parsing or configurable variables where appropriate.",
        "Make the script suitable for automation in CI or cron.",
    ]

    for _, prompt in PYTHON_TASKS:
        prompts.append(("Python", prompt))
        for suffix in python_suffixes:
            prompts.append(("Python", f"{prompt} {suffix}"))
    for _, prompt in JAVASCRIPT_TASKS:
        prompts.append(("JavaScript", prompt))
        for suffix in javascript_suffixes:
            prompts.append(("JavaScript", f"{prompt} {suffix}"))
    for _, prompt in SHELL_TASKS:
        prompts.append(("Shell", prompt))
        for suffix in shell_suffixes:
            prompts.append(("Shell", f"{prompt} {suffix}"))
    return prompts


def generate_with_hf(args: argparse.Namespace) -> list[dict]:
    prompts = build_prompts()
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.generator_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    rows = []
    progress_path = Path(f"{args.output_path}.progress.json")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    started_at = time.time()
    while len(rows) < args.target_samples:
        random.shuffle(prompts)
        for language, prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = clean_generated_code(tokenizer.decode(output[0], skip_special_tokens=True))
            rows.append(
                {
                    "prompt": prompt,
                    "text": text,
                    "language": language,
                    "generator": f"hf:{args.generator_model}",
                }
            )
            append_jsonl(output_path, [rows[-1]])
            maybe_report_progress(
                rows=rows,
                target_samples=args.target_samples,
                progress_every=args.progress_every,
                progress_path=progress_path,
                backend="hf",
                generator_model=args.generator_model,
                started_at=started_at,
            )
            if len(rows) >= args.target_samples:
                break
    maybe_report_progress(
        rows=rows,
        target_samples=args.target_samples,
        progress_every=args.progress_every,
        progress_path=progress_path,
        backend="hf",
        generator_model=args.generator_model,
        started_at=started_at,
        force=True,
    )
    return rows


def generate_with_openai(args: argparse.Namespace) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install optional dependency with: pip install -e .[openai]") from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    client = OpenAI()
    prompts = build_prompts()
    rows = []
    progress_path = Path(f"{args.output_path}.progress.json")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    started_at = time.time()
    print(
        f"Starting OpenAI synthetic generation for {args.target_samples} samples with progress updates every "
        f"{args.progress_every} samples.",
        flush=True,
    )
    print(f"Progress file: {progress_path}", flush=True)
    while len(rows) < args.target_samples:
        random.shuffle(prompts)
        for language, prompt in prompts:
            response = client.responses.create(
                model=args.generator_model,
                instructions="Return only raw code with no markdown fences, no prose, and no explanation.",
                input=f"Task: {prompt}",
                temperature=args.temperature,
                max_output_tokens=args.max_new_tokens,
            )
            text = clean_generated_code(response.output_text)
            rows.append(
                {
                    "prompt": prompt,
                    "text": text,
                    "language": language,
                    "generator": f"openai:{args.generator_model}",
                }
            )
            append_jsonl(output_path, [rows[-1]])
            maybe_report_progress(
                rows=rows,
                target_samples=args.target_samples,
                progress_every=args.progress_every,
                progress_path=progress_path,
                backend="openai",
                generator_model=args.generator_model,
                started_at=started_at,
            )
            if len(rows) >= args.target_samples:
                break
    maybe_report_progress(
        rows=rows,
        target_samples=args.target_samples,
        progress_every=args.progress_every,
        progress_path=progress_path,
        backend="openai",
        generator_model=args.generator_model,
        started_at=started_at,
        force=True,
    )
    return rows


def main() -> None:
    args = parse_args()
    if args.backend == "template":
        rows = build_template_corpus(target_samples=args.target_samples)
        write_jsonl(args.output_path, rows)
        write_progress(
            Path(f"{args.output_path}.progress.json"),
            {
                "backend": "template",
                "generator_model": "template",
                "target_samples": args.target_samples,
                "samples_written": len(rows),
                "percent_complete": 100.0,
                "output_path": args.output_path,
            },
        )
    elif args.backend == "hf":
        rows = generate_with_hf(args)
        Path(f"{args.output_path}.complete").write_text("ok\n", encoding="utf-8")
    else:
        rows = generate_with_openai(args)
        Path(f"{args.output_path}.complete").write_text("ok\n", encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic samples to {args.output_path}")


if __name__ == "__main__":
    main()
