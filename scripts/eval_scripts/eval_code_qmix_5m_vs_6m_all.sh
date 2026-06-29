#!/usr/bin/env bash
set -euo pipefail

# Drive compare_obs_delay.sh over all code_qmix 5m_vs_6m models under results/models.
# Each model uses the numeric checkpoint closest to TARGET_LOAD_STEP.

CONFIG="${CONFIG:-code_qmix}"
MAP_NAME="${MAP_NAME:-5m_vs_6m}"
TARGET_LOAD_STEP="${TARGET_LOAD_STEP:-2000000}"
MODEL_ROOT="${MODEL_ROOT:-results/models}"
MODEL_NAME_GLOB="${MODEL_NAME_GLOB:-code_qmix*__5m_vs_6m__*}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-scripts/eval_scripts/compare_obs_delay.sh}"
DRY_RUN="${DRY_RUN:-False}"
RAW_SUMMARY_KEEP="${RAW_SUMMARY_KEEP:-True}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "$MODEL_ROOT" ]] || die "model root does not exist: $MODEL_ROOT"
[[ -f "$COMPARE_SCRIPT" ]] || die "compare script does not exist: $COMPARE_SCRIPT"

mapfile -d '' MODEL_DIRS < <(
  find "$MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$MODEL_NAME_GLOB" -print0 | sort -z
)

[[ ${#MODEL_DIRS[@]} -gt 0 ]] || die "no models matched: $MODEL_ROOT/$MODEL_NAME_GLOB"

echo "Found ${#MODEL_DIRS[@]} model(s)."
echo "CONFIG=$CONFIG"
echo "MAP_NAME=$MAP_NAME"
echo "TARGET_LOAD_STEP=$TARGET_LOAD_STEP"
echo "COMPARE_SCRIPT=$COMPARE_SCRIPT"
echo "DRY_RUN=$DRY_RUN"
echo "RAW_SUMMARY_KEEP=$RAW_SUMMARY_KEEP"

for model_dir in "${MODEL_DIRS[@]}"; do
  echo "================================================================================"
  echo "MODEL_DIR: $model_dir"
  if [[ "$DRY_RUN" == "True" || "$DRY_RUN" == "true" || "$DRY_RUN" == "1" ]]; then
    printf 'CONFIG=%q MAP_NAME=%q MODEL_DIR=%q MODEL_NAME= LOAD_STEPS= TARGET_LOAD_STEP=%q bash %q\n' \
      "$CONFIG" "$MAP_NAME" "$model_dir" "$TARGET_LOAD_STEP" "$COMPARE_SCRIPT"
    continue
  fi

  CONFIG="$CONFIG" \
    MAP_NAME="$MAP_NAME" \
    MODEL_DIR="$model_dir" \
    MODEL_NAME="" \
    LOAD_STEPS="" \
    TARGET_LOAD_STEP="$TARGET_LOAD_STEP" \
    RAW_SUMMARY_KEEP="$RAW_SUMMARY_KEEP" \
    bash "$COMPARE_SCRIPT"
done

echo "================================================================================"
echo "Done."
