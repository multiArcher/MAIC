#!/usr/bin/env bash
set -euo pipefail

# Compare one or more checkpoints with and without observation delay.
#
# Usage:
#   bash scripts/eval_scripts/compare_obs_delay.sh <MODEL_DIR> [LOAD_STEP ...]
#
# Example:
#   bash scripts/eval_scripts/compare_obs_delay.sh \
#     "results/models/code_qmix_win_6_run1__5m_vs_6m__2026-06-08_23-46-53-td_=1.0-action_=0.01-continue_=0.1-aux_=0.01-entropy_=0.01" \
#     114 1000115 2000157 3000232
#
# You can override the important parameters from the command line:
#   MAP_NAME=5m_vs_6m TEST_NEPISODE=200 SEEDS="2024 2025" \
#   OBS_DELAY_MEAN=1.0 OBS_DELAY_STD=1.0 \
#   bash scripts/eval_scripts/compare_obs_delay.sh <MODEL_DIR> 3000232

# -------- Main experiment controls --------

# StarCraft map/environment. Must match the model's trained map for a clean comparison.
MAP_NAME="${MAP_NAME:-5m_vs_6m}"

# Number of evaluation episodes per run. Larger values reduce variance but take longer.
TEST_NEPISODE="${TEST_NEPISODE:-10}"

# Number of SC2 environments launched in parallel. Keep this small on local WSL.
# code_qmix.yaml defaults to 16, which often causes PySC2 connection failures locally.
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-1}"

# Evaluation seeds. Multiple seeds are recommended because SC2 and delay sampling are stochastic.
SEEDS="${SEEDS:-2024}"

# Device controls. Set CUDA_VISIBLE_DEVICES="" and DEVICE=cpu if you want CPU evaluation.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"

# StarCraft II install path used by SMAC/PySC2. This matches the training scripts.
# Override it if your game is installed somewhere else:
#   SC2PATH=/path/to/StarCraftII bash scripts/eval_scripts/compare_obs_delay.sh ...
SC2PATH="${SC2PATH:-$HOME/.local/share/StarCraftII}"

# Algorithm/environment configs used by the saved code_qmix checkpoint.
CONFIG="${CONFIG:-code_qmix}"
ENV_CONFIG="${ENV_CONFIG:-sc2}"

# -------- Delay controls --------

# Communication delay is fixed to zero by default so the comparison isolates observation delay.
COMM_DELAY_MEAN="${COMM_DELAY_MEAN:-0}"
COMM_DELAY_STD="${COMM_DELAY_STD:-0}"

# Observation delay distribution. RBC-style observation delay commonly uses N(1, 1).
# 这里切换延迟程度
OBS_DELAY_MEAN="${OBS_DELAY_MEAN:-1.0}"
OBS_DELAY_STD="${OBS_DELAY_STD:-1.0}"

# How to convert sampled continuous Gaussian delay to integer timesteps.
# round keeps the realized mean closest to OBS_DELAY_MEAN; ceil is more pessimistic.
OBS_DELAY_DISCRETIZATION="${OBS_DELAY_DISCRETIZATION:-round}"

# Prefix in result names. The script appends step, seed, and obs/no_obs automatically.
NAME_PREFIX="${NAME_PREFIX:-eval_obs_delay_compare}"

# Evaluation summary output. The script appends one row per eval run.
SUMMARY_DIR="${SUMMARY_DIR:-results/eval_summaries}"
SUMMARY_PATH="${SUMMARY_PATH:-$SUMMARY_DIR/${NAME_PREFIX}_$(date +%Y-%m-%d_%H-%M-%S).tsv}"

# -------- Arguments --------

if [[ $# -lt 1 ]]; then
  echo "ERROR: missing MODEL_DIR."
  echo "Usage: bash scripts/eval_scripts/compare_obs_delay.sh <MODEL_DIR> [LOAD_STEP ...]" >&2
  exit 1
fi

MODEL_DIR="$1"
shift

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "ERROR: model directory does not exist: $MODEL_DIR" >&2
  exit 1
fi

# Numeric checkpoint directories are required by src/run.py. best_model is not loaded directly.
if [[ $# -gt 0 ]]; then
  LOAD_STEPS=("$@")
else
  mapfile -t LOAD_STEPS < <(
    find "$MODEL_DIR" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" \
      | awk '/^[0-9]+$/' \
      | sort -n
  )
  if [[ ${#LOAD_STEPS[@]} -gt 0 ]]; then
    LOAD_STEPS=("${LOAD_STEPS[-1]}")
  fi
fi

if [[ ${#LOAD_STEPS[@]} -eq 0 ]]; then
  echo "ERROR: no numeric checkpoint directories found under: $MODEL_DIR" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export SC2PATH
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

if [[ ! -d "$SC2PATH/Versions" ]]; then
  echo "ERROR: StarCraft II Versions directory not found: $SC2PATH/Versions" >&2
  echo "Set SC2PATH to your StarCraftII install, for example:" >&2
  echo "  SC2PATH=\$HOME/.local/share/StarCraftII bash $0 <MODEL_DIR> [LOAD_STEP ...]" >&2
  exit 1
fi

mkdir -p "$SUMMARY_DIR"
printf "step\tseed\tcondition\ttest_battle_won_mean\tlog_path\n" > "$SUMMARY_PATH"

extract_latest_winrate() {
  local run_name="$1"
  local log_path
  local winrate

  log_path="$(
    find results/logs -maxdepth 1 -type f -name "${run_name}__${MAP_NAME}__*.log" -printf "%T@ %p\n" \
      | sort -nr \
      | head -1 \
      | cut -d' ' -f2-
  )"

  if [[ -z "$log_path" || ! -f "$log_path" ]]; then
    echo "NA"
    return
  fi

  winrate="$(
    awk '/metric\/test_battle_won_mean:/ { value = $NF } END { if (value == "") print "NA"; else print value }' "$log_path"
  )"
  echo "$winrate|$log_path"
}

run_eval() {
  local step="$1"
  local seed="$2"
  local obs_enabled="$3"
  local obs_label="$4"

  local run_name="${NAME_PREFIX}_${MAP_NAME}_step${step}_seed${seed}_${obs_label}"

  echo "================================================================================"
  echo "MODEL_DIR:      $MODEL_DIR"
  echo "MAP_NAME:       $MAP_NAME"
  echo "LOAD_STEP:      $step"
  echo "SEED:           $seed"
  echo "OBS_DELAY:      $obs_enabled"
  echo "RUN_NAME:       $run_name"
  echo "TEST_NEPISODE:  $TEST_NEPISODE"
  echo "BATCH_SIZE_RUN: $BATCH_SIZE_RUN"
  echo "================================================================================"

  python src/main.py --config="$CONFIG" --env-config="$ENV_CONFIG" with \
    evaluate=True \
    checkpoint_path="$MODEL_DIR" \
    load_step="$step" \
    env_args.map_name="$MAP_NAME" \
    test_nepisode="$TEST_NEPISODE" \
    batch_size_run="$BATCH_SIZE_RUN" \
    seed="$seed" \
    device="$DEVICE" \
    comm_gaussian_delay_mean="$COMM_DELAY_MEAN" \
    comm_gaussian_delay_std="$COMM_DELAY_STD" \
    obs_delay_enabled="$obs_enabled" \
    obs_delay_apply_train=False \
    obs_delay_apply_test="$obs_enabled" \
    obs_gaussian_delay_mean="$OBS_DELAY_MEAN" \
    obs_gaussian_delay_std="$OBS_DELAY_STD" \
    obs_delay_discretization="$OBS_DELAY_DISCRETIZATION" \
    name="$run_name"

  local extracted
  local winrate
  local log_path
  extracted="$(extract_latest_winrate "$run_name")"
  winrate="${extracted%%|*}"
  log_path="${extracted#*|}"
  if [[ "$extracted" == "$winrate" ]]; then
    log_path="NA"
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" "$step" "$seed" "$obs_label" "$winrate" "$log_path" >> "$SUMMARY_PATH"
  echo "RESULT: step=$step seed=$seed condition=$obs_label metric/test_battle_won_mean=$winrate"
}

for step in "${LOAD_STEPS[@]}"; do
  if [[ ! "$step" =~ ^[0-9]+$ ]]; then
    echo "ERROR: LOAD_STEP must be numeric for current src/run.py: $step" >&2
    exit 1
  fi

  for seed in $SEEDS; do
    run_eval "$step" "$seed" False "no_obs_delay"
    run_eval "$step" "$seed" True "obs_delay_n${OBS_DELAY_MEAN}_${OBS_DELAY_STD}_${OBS_DELAY_DISCRETIZATION}"
  done
done

echo "================================================================================"
echo "Raw summary saved to: $SUMMARY_PATH"
column -t -s $'\t' "$SUMMARY_PATH" || cat "$SUMMARY_PATH"

echo "================================================================================"
echo "No-delay vs obs-delay comparison"
awk -F '\t' '
  NR == 1 { next }
  {
    key = $1 "\t" $2
    if ($3 == "no_obs_delay") {
      no[key] = $4
    } else {
      obs[key] = $4
      obs_label[key] = $3
    }
  }
  END {
    printf "%-12s %-8s %-14s %-14s %-14s %-24s\n", "step", "seed", "no_obs", "obs_delay", "delta", "obs_condition"
    for (key in no) {
      split(key, parts, "\t")
      if ((key in obs) && no[key] != "NA" && obs[key] != "NA") {
        delta = obs[key] - no[key]
        printf "%-12s %-8s %-14.4f %-14.4f %-14.4f %-24s\n", parts[1], parts[2], no[key], obs[key], delta, obs_label[key]
      } else {
        printf "%-12s %-8s %-14s %-14s %-14s %-24s\n", parts[1], parts[2], no[key], (key in obs ? obs[key] : "NA"), "NA", (key in obs_label ? obs_label[key] : "NA")
      }
    }
  }
' "$SUMMARY_PATH"

echo "Done. Check results/logs or tensorboard for full evaluation logs."
