#!/usr/bin/env bash
set -euo pipefail

# Evaluate every code_qmix model trained on 5m_vs_6m under the same delay-grid
# settings used by compare_obs_delay.sh.
#
# Existing matrix TSV outputs are treated as completed evaluations and skipped:
#   eval_summaries/code_qmix/5m_vs_6m/<model_name>_matrix.tsv
#
# Usage:
#   bash scripts/eval_scripts/compare_obs_delay_all_5m_vs_6m_code_qmix.sh
#
# Useful overrides are forwarded to compare_obs_delay.sh, for example:
#   TEST_NEPISODE=16 GROUP_SEEDS="2024 2025 2026 2027 2028" \
#   COMM_DELAY_GRID="0:0 1:1 2:1 2:2 3:1 3:2 5:2" \
#   OBS_DELAY_GRID="0:1 -1:1 0:0 1:1 2:1 2:2" \
#   bash scripts/eval_scripts/compare_obs_delay_all_5m_vs_6m_code_qmix.sh
#
# Preview the work without launching evaluation:
#   DRY_RUN=True bash scripts/eval_scripts/compare_obs_delay_all_5m_vs_6m_code_qmix.sh

CONFIG="${CONFIG:-code_qmix}"
ENV_CONFIG="${ENV_CONFIG:-sc2}"
MAP_NAME="${MAP_NAME:-5m_vs_6m}"

MODELS_ROOT="${MODELS_ROOT:-results/models}"
SUMMARY_ROOT="${SUMMARY_ROOT:-eval_summaries}"
SUMMARY_DIR="${SUMMARY_DIR:-$SUMMARY_ROOT/$CONFIG/$MAP_NAME}"
RAW_SUMMARY_DIR="${RAW_SUMMARY_DIR:-$SUMMARY_ROOT/raw_tsv}"
EVAL_SCRIPT="${EVAL_SCRIPT:-scripts/eval_scripts/compare_obs_delay.sh}"

DRY_RUN="${DRY_RUN:-False}"
FORCE="${FORCE:-False}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

truthy() {
  case "${1:-}" in
    True|true|1|yes|Yes|Y|y) return 0 ;;
    *) return 1 ;;
  esac
}

[[ "$CONFIG" == "code_qmix" ]] || die "this batch script is intended for CONFIG=code_qmix, got: $CONFIG"
[[ "$MAP_NAME" == "5m_vs_6m" ]] || die "this batch script is intended for MAP_NAME=5m_vs_6m, got: $MAP_NAME"
[[ -d "$MODELS_ROOT" ]] || die "models root not found: $MODELS_ROOT"
[[ -f "$EVAL_SCRIPT" ]] || die "eval script not found: $EVAL_SCRIPT"

mkdir -p "$SUMMARY_DIR" "$RAW_SUMMARY_DIR"

mapfile -t MODEL_DIRS < <(
  find "$MODELS_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'code_qmix*__5m_vs_6m__*' \
    | sort -V
)

if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
  echo "No code_qmix 5m_vs_6m model directories found under: $MODELS_ROOT"
  exit 0
fi

echo "Found ${#MODEL_DIRS[@]} candidate code_qmix 5m_vs_6m model(s)."
echo "Summary directory: $SUMMARY_DIR"
echo "Eval script:        $EVAL_SCRIPT"
echo

evaluated=0
skipped=0

for model_dir in "${MODEL_DIRS[@]}"; do
  model_name="$(basename "$model_dir")"
  matrix_path="$SUMMARY_DIR/${model_name}_matrix.tsv"
  raw_path="$RAW_SUMMARY_DIR/${model_name}_raw.tsv"

  if [[ -s "$matrix_path" ]] && ! truthy "$FORCE"; then
    echo "SKIP existing matrix: $matrix_path"
    skipped=$((skipped + 1))
    continue
  fi

  echo "================================================================================"
  echo "EVALUATE model: $model_name"
  echo "MODEL_DIR:      $model_dir"
  echo "SUMMARY_PATH:   $matrix_path"
  echo "RAW_PATH:       $raw_path"

  if truthy "$DRY_RUN"; then
    echo "DRY_RUN=True, not launching evaluation."
    evaluated=$((evaluated + 1))
    continue
  fi

  CONFIG="$CONFIG" \
  ENV_CONFIG="$ENV_CONFIG" \
  MAP_NAME="$MAP_NAME" \
  MODEL_NAME="$model_name" \
  MODEL_DIR="$model_dir" \
  SUMMARY_ROOT="$SUMMARY_ROOT" \
  SUMMARY_DIR="$SUMMARY_DIR" \
  SUMMARY_PATH="$matrix_path" \
  RAW_SUMMARY_DIR="$RAW_SUMMARY_DIR" \
  RAW_SUMMARY_OUTPUT_PATH="$raw_path" \
  bash "$EVAL_SCRIPT"

  evaluated=$((evaluated + 1))
done

echo "================================================================================"
echo "Done. evaluated=$evaluated skipped=$skipped total=${#MODEL_DIRS[@]}"
