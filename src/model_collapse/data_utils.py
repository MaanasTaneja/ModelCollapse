from __future__ import annotations

import glob
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, IterableDataset, load_dataset
from transformers import AutoTokenizer


RANDOM = random.Random(42)


@dataclass
class CodeExample:
    text: str
    language: str | None = None
    source: str | None = None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def guess_language_from_suffix(path: Path) -> str:
    suffix_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".sh": "Shell",
        ".bash": "Shell",
        ".zsh": "Shell",
        ".c": "C",
        ".cpp": "C++",
        ".h": "C",
        ".hpp": "C++",
        ".rs": "Rust",
        ".go": "Go",
        ".java": "Java",
        ".asm": "Assembly",
        ".s": "Assembly",
        ".forth": "Forth",
        ".fs": "Forth",
    }
    return suffix_map.get(path.suffix.lower(), "Unknown")


def collect_local_code(input_globs: list[str], max_samples: int | None = None) -> list[CodeExample]:
    matched_paths: list[Path] = []
    for pattern in input_globs:
        matched_paths.extend(Path(p) for p in glob.glob(pattern, recursive=True))

    unique_paths = []
    seen = set()
    for path in matched_paths:
        resolved = str(path.resolve())
        if resolved not in seen and path.is_file():
            seen.add(resolved)
            unique_paths.append(path)

    RANDOM.shuffle(unique_paths)
    if max_samples is not None:
        unique_paths = unique_paths[:max_samples]

    examples: list[CodeExample] = []
    for path in unique_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        examples.append(
            CodeExample(
                text=text,
                language=guess_language_from_suffix(path),
                source=str(path),
            )
        )
    return examples


def _iter_hf_records(
    dataset_name: str,
    split: str,
    config_name: str | None = None,
    data_dir: str | None = None,
    streaming: bool = True,
):
    dataset = load_dataset(
        dataset_name,
        config_name,
        data_dir=data_dir,
        split=split,
        streaming=streaming,
    )
    if isinstance(dataset, (Dataset, IterableDataset)):
        return dataset
    raise TypeError(f"Unexpected dataset type: {type(dataset)}")


def normalize_language_name(language: str | None) -> str | None:
    if language is None:
        return None
    value = str(language).strip()
    if not value:
        return None
    return value.casefold()


def infer_language_from_data_dir(data_dir: str | None) -> str | None:
    if data_dir is None:
        return None
    key = Path(data_dir).name.casefold()
    aliases = {
        "python": "python",
        "javascript": "javascript",
        "typescript": "typescript",
        "java": "java",
        "c": "c",
        "cpp": "c++",
        "go": "go",
        "rust": "rust",
        "solidity": "solidity",
        "cuda": "cuda",
        "assembly": "assembly",
        "shell": "shell",
    }
    return aliases.get(key)


def parse_language_quota_specs(language_quota_specs: list[str] | None) -> dict[str, float]:
    if not language_quota_specs:
        return {}

    quotas: dict[str, float] = {}
    for spec in language_quota_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid language quota '{spec}'. Expected format Language=weight.")
        language, weight = spec.split("=", 1)
        language_key = normalize_language_name(language)
        if language_key is None:
            raise ValueError(f"Invalid language quota '{spec}'. Missing language name.")
        try:
            numeric_weight = float(weight)
        except ValueError as exc:
            raise ValueError(f"Invalid language quota '{spec}'. Weight must be numeric.") from exc
        if numeric_weight <= 0:
            raise ValueError(f"Invalid language quota '{spec}'. Weight must be > 0.")
        quotas[language_key] = numeric_weight
    return quotas


def compute_language_targets(
    *,
    languages: list[str] | None,
    max_samples: int,
    language_quota_specs: list[str] | None,
) -> dict[str, int]:
    language_keys = [normalize_language_name(language) for language in (languages or [])]
    language_keys = [key for key in language_keys if key is not None]
    if not language_keys:
        return {}

    quota_weights = parse_language_quota_specs(language_quota_specs)
    missing_weights = [language for language in language_keys if language not in quota_weights]
    if quota_weights and missing_weights:
        missing = ", ".join(sorted(missing_weights))
        raise ValueError(f"Missing language quota weights for: {missing}")

    if quota_weights:
        weights = {language: quota_weights[language] for language in language_keys}
    else:
        weights = {language: 1.0 for language in language_keys}

    total_weight = sum(weights.values())
    raw_targets = {language: (max_samples * weights[language] / total_weight) for language in language_keys}
    targets = {language: int(math.floor(value)) for language, value in raw_targets.items()}
    assigned = sum(targets.values())
    remainders = sorted(
        ((raw_targets[language] - targets[language], language) for language in language_keys),
        reverse=True,
    )
    for _, language in remainders:
        if assigned >= max_samples:
            break
        targets[language] += 1
        assigned += 1
    return targets


def collect_hf_code(
    dataset_name: str,
    split: str,
    text_column: str,
    language_column: str | None,
    languages: list[str] | None,
    max_samples: int,
    config_name: str | None = None,
    data_dirs: list[str] | None = None,
    language_quota_specs: list[str] | None = None,
) -> list[CodeExample]:
    rows = []
    language_targets = compute_language_targets(
        languages=languages,
        max_samples=max_samples,
        language_quota_specs=language_quota_specs,
    )
    language_set = set(language_targets) if language_targets else (
        {normalize_language_name(lang) for lang in languages} if languages else None
    )
    language_counts: dict[str, int] = defaultdict(int)
    active_data_dirs = data_dirs or [None]
    if language_targets:
        pretty_targets = ", ".join(
            f"{language}={target}" for language, target in sorted(language_targets.items())
        )
        print(f"Using weighted language targets: {pretty_targets}", flush=True)

    for data_dir in active_data_dirs:
        label = data_dir if data_dir is not None else "<default>"
        shard_language = infer_language_from_data_dir(data_dir)
        if (
            language_targets
            and shard_language is not None
            and language_counts.get(shard_language, 0) >= language_targets.get(shard_language, math.inf)
        ):
            print(f"Skipping dataset shard {label} because target for {shard_language} is already full", flush=True)
            continue
        print(f"Streaming dataset shard: {label}", flush=True)
        for row in _iter_hf_records(
            dataset_name,
            split=split,
            config_name=config_name,
            data_dir=data_dir,
            streaming=True,
        ):
            text = row.get(text_column)
            if not isinstance(text, str) or not text.strip():
                continue
            language = row.get(language_column) if language_column else None
            language_key = normalize_language_name(language)
            if (
                language_targets
                and shard_language is not None
                and language_key == shard_language
                and language_counts[language_key] >= language_targets[language_key]
            ):
                print(f"Reached target for shard language {shard_language}; moving to next shard", flush=True)
                break
            if language_set and language_key not in language_set:
                continue
            if language_targets and language_key is not None and language_counts[language_key] >= language_targets[language_key]:
                continue
            rows.append(CodeExample(text=text, language=str(language) if language is not None else None))
            if language_key is not None:
                language_counts[language_key] += 1
            if len(rows) % 500 == 0:
                print(f"Collected {len(rows)}/{max_samples} examples so far", flush=True)
                if language_targets:
                    language_status = ", ".join(
                        f"{language}={language_counts.get(language, 0)}/{language_targets[language]}"
                        for language in sorted(language_targets)
                    )
                    print(f"Language progress: {language_status}", flush=True)
            if len(rows) >= max_samples:
                print(f"Reached target sample budget of {max_samples}", flush=True)
                return rows
            if language_targets and all(
                language_counts.get(language, 0) >= target for language, target in language_targets.items()
            ):
                print("Reached all weighted language targets", flush=True)
                return rows
    return rows


def normalize_code_text(text: str, min_chars: int = 40, max_chars: int = 4000) -> str | None:
    text = text.replace("\r\n", "\n").strip()
    if len(text) < min_chars:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def deduplicate_examples(examples: list[CodeExample]) -> list[CodeExample]:
    deduped = []
    seen = set()
    for example in examples:
        text = normalize_code_text(example.text)
        if text is None:
            continue
        key = hash(text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(CodeExample(text=text, language=example.language, source=example.source))
    return deduped


def compute_token_frequencies(texts: Iterable[str], tokenizer_name: str) -> Counter:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    counter: Counter = Counter()
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        counter.update(token_ids)
    return counter


def select_tail_token_ids(
    token_freq: Counter,
    tokenizer_name: str,
    tail_percentile: float,
    min_count: int = 1,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    usable = []
    for token_id, count in token_freq.items():
        if count < min_count:
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token is None:
            continue
        usable.append((token_id, count, token))

    usable.sort(key=lambda item: (item[1], item[0]))
    cutoff = max(1, math.floor(len(usable) * tail_percentile))
    tail = usable[:cutoff]
    return {
        "tokenizer_name": tokenizer_name,
        "tail_percentile": tail_percentile,
        "num_tail_tokens": len(tail),
        "tail_token_ids": [token_id for token_id, _, _ in tail],
        "tail_tokens_preview": [
            {"id": token_id, "token": token, "count": count}
            for token_id, count, token in tail[:200]
        ],
    }


def count_tail_tokens_in_text(text: str, tokenizer_name: str, tail_token_ids: set[int]) -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return sum(1 for token_id in token_ids if token_id in tail_token_ids)


def split_clean_and_eval(
    examples: list[CodeExample],
    tokenizer_name: str,
    tail_token_ids: set[int],
    min_tail_tokens_per_eval: int,
    max_eval_samples: int,
) -> tuple[list[dict], list[dict]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eval_rows: list[dict] = []
    train_rows: list[dict] = []

    for example in examples:
        token_ids = tokenizer.encode(example.text, add_special_tokens=False)
        tail_count = sum(1 for token_id in token_ids if token_id in tail_token_ids)
        row = {
            "text": example.text,
            "language": example.language,
            "source": example.source,
            "tail_token_count": tail_count,
            "num_tokens": len(token_ids),
        }
        if tail_count >= min_tail_tokens_per_eval and len(eval_rows) < max_eval_samples:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, eval_rows


def split_clean_and_eval_by_language(
    examples: list[CodeExample],
    tokenizer_name: str,
    eval_languages: list[str],
    max_eval_samples: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict], dict]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    normalized_eval_languages = [normalize_language_name(language) for language in eval_languages]
    normalized_eval_languages = [language for language in normalized_eval_languages if language is not None]
    eval_language_set = set(normalized_eval_languages)

    eval_buckets: dict[str, list[dict]] = {language: [] for language in normalized_eval_languages}
    train_rows: list[dict] = []

    for example in examples:
        token_ids = tokenizer.encode(example.text, add_special_tokens=False)
        row = {
            "text": example.text,
            "language": example.language,
            "source": example.source,
            "num_tokens": len(token_ids),
        }
        language_key = normalize_language_name(example.language)
        if language_key in eval_language_set:
            eval_buckets.setdefault(language_key, []).append(row)
        else:
            train_rows.append(row)

    rng = random.Random(seed)
    for rows in eval_buckets.values():
        rng.shuffle(rows)

    eval_rows: list[dict] = []
    while len(eval_rows) < max_eval_samples:
        progressed = False
        for language in normalized_eval_languages:
            bucket = eval_buckets.get(language, [])
            if not bucket:
                continue
            eval_rows.append(bucket.pop())
            progressed = True
            if len(eval_rows) >= max_eval_samples:
                break
        if not progressed:
            break

    for language in normalized_eval_languages:
        train_rows.extend(eval_buckets.get(language, []))

    metadata = {
        "selection_mode": "language_slice",
        "eval_languages": eval_languages,
        "num_eval_examples": len(eval_rows),
        "eval_language_counts": dict(Counter((row.get("language") or "Unknown") for row in eval_rows)),
    }
    return train_rows, eval_rows, metadata


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:n]
