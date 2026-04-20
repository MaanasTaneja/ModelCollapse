#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-artifacts/logs}"
mkdir -p "$LOG_DIR"
PIPELINE_LOG_PATH="${PIPELINE_LOG_PATH:-$LOG_DIR/pipeline_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$PIPELINE_LOG_PATH") 2>&1

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

die() {
  printf '\n[ERROR] %s\n' "$1" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-artifacts/data}"
OUTPUT_PLOTS_DIR="${OUTPUT_PLOTS_DIR:-artifacts/plots}"
OUTPUT_PAPER_DIR="${OUTPUT_PAPER_DIR:-artifacts/paper}"
CONFIG_PATH="${CONFIG_PATH:-configs/experiment.yaml}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"
GENERATE_SYNTH="${GENERATE_SYNTH:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_PLOTTING="${RUN_PLOTTING:-1}"
INSPECT_HF_SCHEMA="${INSPECT_HF_SCHEMA:-0}"

DATA_MODE="${DATA_MODE:-hf}"
DATASET_NAME="${DATASET_NAME:-bigcode/the-stack-dedup}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
DATASET_DATA_DIRS="${DATASET_DATA_DIRS:-data/python data/javascript data/typescript data/java data/c data/cpp data/go data/rust data/solidity data/cuda data/assembly data/shell}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TEXT_COLUMN="${TEXT_COLUMN:-content}"
LANGUAGE_COLUMN="${LANGUAGE_COLUMN:-lang}"
LANGUAGES="${LANGUAGES:-Python JavaScript TypeScript Java C C++ Go Rust Solidity CUDA Assembly Shell}"
LANGUAGE_QUOTAS="${LANGUAGE_QUOTAS:-Python=0.21 JavaScript=0.15 TypeScript=0.13 Java=0.12 C=0.08 C++=0.08 Go=0.06 Rust=0.05 Solidity=0.05 CUDA=0.04 Assembly=0.02 Shell=0.01}"
TAIL_EVAL_LANGUAGES="${TAIL_EVAL_LANGUAGES:-Java C C++ Go Rust Solidity CUDA Assembly}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-12000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-800}"
TAIL_PERCENTILE="${TAIL_PERCENTILE:-0.10}"
MIN_TAIL_TOKENS_PER_EVAL="${MIN_TAIL_TOKENS_PER_EVAL:-5}"
TOKENIZER_NAME="${TOKENIZER_NAME:-gpt2}"

LOCAL_GLOBS="${LOCAL_GLOBS:-}"

SYNTH_BACKEND="${SYNTH_BACKEND:-openai}"
SYNTH_MODEL="${SYNTH_MODEL:-gpt-4o-mini}"
SYNTH_TARGET_SAMPLES="${SYNTH_TARGET_SAMPLES:-12000}"
OPENAI_MAX_NEW_TOKENS="${OPENAI_MAX_NEW_TOKENS:-220}"
SYNTH_OUTPUT_PATH="${SYNTH_OUTPUT_PATH:-$OUTPUT_DATA_DIR/synth_head.jsonl}"

print_config() {
  cat <<EOF
ROOT_DIR=$ROOT_DIR
PYTHON_BIN=$PYTHON_BIN
VENV_DIR=$VENV_DIR
DATA_MODE=$DATA_MODE
DATASET_NAME=$DATASET_NAME
DATASET_CONFIG=$DATASET_CONFIG
DATASET_DATA_DIRS=$DATASET_DATA_DIRS
DATASET_SPLIT=$DATASET_SPLIT
TEXT_COLUMN=$TEXT_COLUMN
LANGUAGE_COLUMN=$LANGUAGE_COLUMN
LANGUAGES=$LANGUAGES
LANGUAGE_QUOTAS=$LANGUAGE_QUOTAS
TAIL_EVAL_LANGUAGES=$TAIL_EVAL_LANGUAGES
LOCAL_GLOBS=$LOCAL_GLOBS
SYNTH_BACKEND=$SYNTH_BACKEND
SYNTH_MODEL=$SYNTH_MODEL
SYNTH_TARGET_SAMPLES=$SYNTH_TARGET_SAMPLES
OPENAI_MAX_NEW_TOKENS=$OPENAI_MAX_NEW_TOKENS
CONFIG_PATH=$CONFIG_PATH
INSPECT_HF_SCHEMA=$INSPECT_HF_SCHEMA
PIPELINE_LOG_PATH=$PIPELINE_LOG_PATH
EOF
}

bootstrap_env() {
  require_cmd "$PYTHON_BIN"
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  if [[ "$INSTALL_DEPS" == "1" ]]; then
    log "Installing project dependencies"
    python -m pip install --upgrade pip
    python -m pip install -e ".[openai]"
  else
    log "Skipping dependency installation because INSTALL_DEPS=0"
  fi
}

prepare_hf_data() {
  local args=(
    python scripts/prepare_clean_corpus.py
    --dataset-name "$DATASET_NAME"
    --dataset-split "$DATASET_SPLIT"
    --text-column "$TEXT_COLUMN"
    --language-column "$LANGUAGE_COLUMN"
    --tokenizer-name "$TOKENIZER_NAME"
    --tail-percentile "$TAIL_PERCENTILE"
    --min-tail-tokens-per-eval "$MIN_TAIL_TOKENS_PER_EVAL"
    --max-train-samples "$MAX_TRAIN_SAMPLES"
    --max-eval-samples "$MAX_EVAL_SAMPLES"
    --output-dir "$OUTPUT_DATA_DIR"
  )

  if [[ -n "$DATASET_CONFIG" ]]; then
    args+=(--dataset-config "$DATASET_CONFIG")
  fi

  read -r -a data_dir_array <<<"$DATASET_DATA_DIRS"
  for data_dir in "${data_dir_array[@]}"; do
    args+=(--dataset-data-dir "$data_dir")
  done

  read -r -a lang_array <<<"$LANGUAGES"
  if [[ "${#lang_array[@]}" -gt 0 ]]; then
    args+=(--languages "${lang_array[@]}")
  fi

  read -r -a language_quota_array <<<"$LANGUAGE_QUOTAS"
  for language_quota in "${language_quota_array[@]}"; do
    args+=(--language-quota "$language_quota")
  done

  read -r -a tail_eval_language_array <<<"$TAIL_EVAL_LANGUAGES"
  for tail_eval_language in "${tail_eval_language_array[@]}"; do
    args+=(--tail-eval-language "$tail_eval_language")
  done

  if [[ "$INSPECT_HF_SCHEMA" == "1" ]]; then
    log "Inspecting Hugging Face dataset schema"
    local inspect_args=(
      python scripts/inspect_hf_dataset.py
      --dataset-name "$DATASET_NAME"
      --sample-split "$DATASET_SPLIT"
    )
    if [[ -n "$DATASET_CONFIG" ]]; then
      inspect_args+=(--dataset-config "$DATASET_CONFIG")
    fi
    if [[ "${#data_dir_array[@]}" -gt 0 ]]; then
      inspect_args+=(--dataset-data-dir "${data_dir_array[0]}")
    fi
    "${inspect_args[@]}"
  else
    log "Skipping Hugging Face schema inspection because INSPECT_HF_SCHEMA=0"
  fi

  log "Preparing clean corpus from Hugging Face dataset"
  "${args[@]}"
}

prepare_local_data() {
  [[ -n "$LOCAL_GLOBS" ]] || die "DATA_MODE=local requires LOCAL_GLOBS. Example: LOCAL_GLOBS='data/raw_code/**/*.py data/raw_code/**/*.js data/raw_code/**/*.sh'"

  local args=(
    python scripts/prepare_clean_corpus.py
    --tokenizer-name "$TOKENIZER_NAME"
    --tail-percentile "$TAIL_PERCENTILE"
    --min-tail-tokens-per-eval "$MIN_TAIL_TOKENS_PER_EVAL"
    --max-train-samples "$MAX_TRAIN_SAMPLES"
    --max-eval-samples "$MAX_EVAL_SAMPLES"
    --output-dir "$OUTPUT_DATA_DIR"
  )

  read -r -a glob_array <<<"$LOCAL_GLOBS"
  for pattern in "${glob_array[@]}"; do
    args+=(--input-glob "$pattern")
  done

  log "Preparing clean corpus from local files"
  "${args[@]}"
}

generate_synth_data() {
  log "Generating synthetic head-domain corpus with backend=$SYNTH_BACKEND"
  python scripts/generate_synthetic_head.py \
    --backend "$SYNTH_BACKEND" \
    --generator-model "$SYNTH_MODEL" \
    --target-samples "$SYNTH_TARGET_SAMPLES" \
    --max-new-tokens "$OPENAI_MAX_NEW_TOKENS" \
    --progress-every 50 \
    --output-path "$SYNTH_OUTPUT_PATH"
}

run_training_and_eval() {
  log "Running mixture training and tail evaluation"
  python scripts/run_experiment.py --config "$CONFIG_PATH"
}

plot_results() {
  log "Plotting tail-metric trends"
  python scripts/plot_results.py \
    --results-glob "artifacts/runs/*/tail_metrics.json" \
    --output-dir "$OUTPUT_PLOTS_DIR"
}

write_paper_bundle() {
  log "Writing paper-ready artifact bundle"
  python scripts/generate_paper_artifacts.py \
    --config "$CONFIG_PATH" \
    --artifacts-root artifacts
}

main() {
  log "Pipeline configuration"
  print_config

  bootstrap_env

  mkdir -p "$OUTPUT_DATA_DIR" "$OUTPUT_PLOTS_DIR" "$OUTPUT_PAPER_DIR"

  if [[ "$PREPARE_DATA" == "1" ]]; then
    if [[ "$DATA_MODE" == "hf" ]]; then
      prepare_hf_data
    elif [[ "$DATA_MODE" == "local" ]]; then
      prepare_local_data
    else
      die "Unsupported DATA_MODE=$DATA_MODE. Use hf or local."
    fi
  else
    log "Skipping data preparation because PREPARE_DATA=0"
  fi

  if [[ "$GENERATE_SYNTH" == "1" ]]; then
    generate_synth_data
  else
    log "Skipping synthetic-data generation because GENERATE_SYNTH=0"
  fi

  if [[ "$RUN_TRAINING" == "1" ]]; then
    run_training_and_eval
  else
    log "Skipping training because RUN_TRAINING=0"
  fi

  if [[ "$RUN_PLOTTING" == "1" ]]; then
    plot_results
  else
    log "Skipping plotting because RUN_PLOTTING=0"
  fi

  write_paper_bundle

  log "Pipeline completed"
  log "Primary outputs:"
  printf '%s\n' \
    "  - $OUTPUT_DATA_DIR/clean_train.jsonl" \
    "  - $OUTPUT_DATA_DIR/tail_test.jsonl" \
    "  - $OUTPUT_DATA_DIR/tail_tokens.json" \
    "  - $SYNTH_OUTPUT_PATH" \
    "  - artifacts/runs/summary.json" \
    "  - $OUTPUT_PLOTS_DIR/tail_metrics_trend.png" \
    "  - $OUTPUT_PLOTS_DIR/tail_metrics_summary.csv" \
    "  - $OUTPUT_PAPER_DIR/methodology_summary.md" \
    "  - $OUTPUT_PAPER_DIR/results_summary.md" \
    "  - $OUTPUT_PAPER_DIR/artifact_manifest.json"
}

main "$@"
