#!/usr/bin/env bash
set -euo pipefail

# Evaluate one checkpoint under communication and observation delay.
# Only the final matrix TSV is kept; per-evaluation logs/artifacts are removed
# after extracting metric/test_battle_won_mean.
#
# Choose the algorithm, map, model, and checkpoint with variables below or
# environment overrides:
#   bash scripts/eval_scripts/compare_obs_delay.sh
#   CONFIG=qmix MAP_NAME=MMM2 \
#   MODEL_NAME=qmix_mmm2_baseline_seed2024_run1__MMM2__2026-06-16_00-02-02 \
#   LOAD_STEPS=2000266 bash scripts/eval_scripts/compare_obs_delay.sh
#
# Single-model usage is still supported:
#   CONFIG=qmix MAP_NAME=MMM2 bash scripts/eval_scripts/compare_obs_delay.sh <MODEL_DIR> [LOAD_STEP ...]
#
# Useful overrides:
#   TEST_NEPISODE=16 GROUP_SEEDS="2024 2025 2026 2027 2028" \
#   COMM_DELAY_GRID="0:0 1:1 2:1 2:2 3:1 3:2 5:2" \
#   OBS_DELAY_GRID="0:1 -1:1 0:0 1:1 2:1 2:2 3:1 3:2" INCLUDE_ZERO_ZERO_BASELINE=True \
#   bash scripts/eval_scripts/compare_obs_delay.sh

# -------- Main experiment controls --------

CONFIG="${CONFIG:-qmix}"  # code_qmix
ENV_CONFIG="${ENV_CONFIG:-sc2}"
MAP_NAME="${MAP_NAME:-MMM2}"  # 5m_vs_6m

# MODEL_NAME is resolved under results/models. MODEL_DIR may be used instead
# when an absolute or custom model path is preferred.
MODEL_NAME="${MODEL_NAME:-qmix_mmm2_baseline_seed2024_run1__MMM2__2026-06-16_00-02-02}"
MODEL_DIR="${MODEL_DIR:-}"

# Space-separated numeric checkpoint steps. Set LOAD_STEPS="" to choose the
# checkpoint closest to TARGET_LOAD_STEP.
LOAD_STEPS="${LOAD_STEPS-2000218}"

TEST_NEPISODE="${TEST_NEPISODE:-16}"
BATCH_SIZE_RUN="${BATCH_SIZE_RUN:-4}"
GROUP_SEEDS="${GROUP_SEEDS:-${SEEDS:-2024 2025 2026 2027 2028}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda}"
SC2PATH="${SC2PATH:-$HOME/.local/share/StarCraftII}"

# The script chooses the numeric checkpoint closest to this value when a step is
# not provided explicitly. Numeric checkpoints are required by src/run.py.
TARGET_LOAD_STEP="${TARGET_LOAD_STEP:-2000000}"

# Delay grids use "mean:std" entries separated by spaces.
COMM_DELAY_GRID="${COMM_DELAY_GRID:-0:0 1:1 2:1 2:2 3:1 3:2 5:2}"
OBS_DELAY_GRID="${OBS_DELAY_GRID:-0:1 -1:1 0:0 1:1 2:1 2:2 3:1 3:2}"
OBS_DELAY_DISCRETIZATION="${OBS_DELAY_DISCRETIZATION:-round}"
INCLUDE_ZERO_ZERO_BASELINE="${INCLUDE_ZERO_ZERO_BASELINE:-False}"
MATRIX_VALUE_SCALE="${MATRIX_VALUE_SCALE:-percent}"

# Optional extra Sacred config overrides, for algorithms/checkpoints that need
# additional eval-time values.
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"
CONFIG_HAS_COMM_DELAY=False

NAME_PREFIX="${NAME_PREFIX:-eval_delay_grid}"
SUMMARY_ROOT="${SUMMARY_ROOT:-eval_summaries}"
SUMMARY_DIR="${SUMMARY_DIR:-}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
RAW_SUMMARY_KEEP="${RAW_SUMMARY_KEEP:-False}"
RAW_SUMMARY_KEEP_PATH="${RAW_SUMMARY_KEEP_PATH:-}"
RAW_SUMMARY_PATH="$(mktemp /tmp/${NAME_PREFIX}_raw_XXXXXX.tsv)"
trap 'rm -f "$RAW_SUMMARY_PATH"' EXIT

# -------- Helpers --------

die() {
  echo "ERROR: $*" >&2
  exit 1
}

sanitize_label_part() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_neg_}"
  echo "$value"
}

parse_delay_pair() {
  local pair="$1"
  local mean="${pair%%:*}"
  local std="${pair#*:}"
  [[ "$pair" == *:* && -n "$mean" && -n "$std" ]] || die "delay grid entry must be mean:std, got: $pair"
  printf "%s\t%s\n" "$mean" "$std"
}

normalise_map_name() {
  case "$1" in
    5V6|5v6|5m6m|5m_vs_6m) echo "5m_vs_6m" ;;
    protoss|Protoss|protoss_5v5|protoss_5_vs_5) echo "protoss_5_vs_5" ;;
    MMM|mmm) echo "MMM" ;;
    MMM2|mmm2) echo "MMM2" ;;
    *) echo "$1" ;;
  esac
}

resolve_model_dir() {
  local model_dir="$1"
  local model_name="$2"

  if [[ -n "$model_dir" ]]; then
    echo "$model_dir"
  elif [[ -n "$model_name" ]]; then
    echo "results/models/$model_name"
  else
    return 1
  fi
}

delay_label() {
  local mean="$1"
  local std="$2"
  echo "N($mean,$std)"
}

append_code_qmix_training_args() {
  local model_dir="$1"
  local -n out_args="$2"
  local model_name
  model_name="$(basename "$model_dir")"

  out_args+=("temporal_discount_gamma_T=${CODE_QMIX_TEMPORAL_DISCOUNT_GAMMA_T:-0.9}")

  if [[ "$model_name" =~ td_=([^:-]+) ]]; then
    out_args+=("td_loss_weight=${BASH_REMATCH[1]}")
  fi
  if [[ "$model_name" =~ action_=([^:-]+) ]]; then
    out_args+=("action_loss_weight=${BASH_REMATCH[1]}")
  fi
  if [[ "$model_name" =~ continue_=([^:-]+) ]]; then
    out_args+=("continue_loss_weight=${BASH_REMATCH[1]}")
  fi
  if [[ "$model_name" =~ aux_=([^:-]+) ]]; then
    out_args+=("aux_loss_weight=${BASH_REMATCH[1]}")
  fi
  if [[ "$model_name" =~ entropy_=([^:-]+) ]]; then
    out_args+=("entropy_loss_weight=${BASH_REMATCH[1]}")
  fi
}

closest_checkpoint() {
  local model_dir="$1"
  find "$model_dir" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" \
    | awk -v target="$TARGET_LOAD_STEP" '
        /^[0-9]+$/ {
          diff = $1 - target
          if (diff < 0) diff = -diff
          if (!seen || diff < best_diff || (diff == best_diff && $1 < best_step)) {
            best_step = $1
            best_diff = diff
            seen = 1
          }
        }
        END {
          if (seen) print best_step
        }
      '
}

extract_latest_winrate() {
  local run_name="$1"
  local map_name="$2"
  local log_path
  local winrate

  log_path="$(
    find results/logs -maxdepth 1 -type f -name "${run_name}__${map_name}__*.log" -printf "%T@ %p\n" \
      | sort -nr \
      | head -1 \
      | cut -d' ' -f2-
  )"

  if [[ -z "$log_path" || ! -f "$log_path" ]]; then
    echo "NA|NA"
    return
  fi

  winrate="$(
    awk '/metric\/test_battle_won_mean:/ { value = $NF } END { if (value == "") print "NA"; else print value }' "$log_path"
  )"
  echo "$winrate|$log_path"
}

cleanup_eval_artifacts() {
  local run_name="$1"
  local map_name="$2"
  local token_prefix="${run_name}__${map_name}__"

  find results/logs -maxdepth 1 -type f -name "${token_prefix}*.log" -delete 2>/dev/null || true
  find results/sacred -maxdepth 1 -mindepth 1 -type d -name "${token_prefix}*" -exec rm -rf {} + 2>/dev/null || true
  find results/tensorboard_logs -maxdepth 1 -mindepth 1 -type d -name "${token_prefix}*" -exec rm -rf {} + 2>/dev/null || true
  find results/traces -maxdepth 1 -mindepth 1 -type d -name "${token_prefix}*" -exec rm -rf {} + 2>/dev/null || true
}

run_eval() {
  local map_name="$1"
  local model_dir="$2"
  local step="$3"
  local seed="$4"
  local comm_mean="$5"
  local comm_std="$6"
  local obs_mean="$7"
  local obs_std="$8"

  local comm_label_mean
  local comm_label_std
  local obs_label_mean
  local obs_label_std
  comm_label_mean="$(sanitize_label_part "$comm_mean")"
  comm_label_std="$(sanitize_label_part "$comm_std")"
  obs_label_mean="$(sanitize_label_part "$obs_mean")"
  obs_label_std="$(sanitize_label_part "$obs_std")"

  local condition="comm_m${comm_label_mean}_s${comm_label_std}__obs_m${obs_label_mean}_s${obs_label_std}_${OBS_DELAY_DISCRETIZATION}"
  local run_name="${NAME_PREFIX}_${map_name}_step${step}_seed${seed}_${condition}"
  local obs_enabled=True
  if [[ "$obs_mean" == "0" && "$obs_std" == "0" ]]; then
    obs_enabled=False
  fi

  echo "================================================================================"
  echo "MODEL_DIR:      $model_dir"
  echo "CONFIG:         $CONFIG"
  echo "MAP_NAME:       $map_name"
  echo "LOAD_STEP:      $step"
  echo "SEED:           $seed"
  echo "COMM_DELAY:     N($comm_mean, $comm_std)"
  echo "OBS_DELAY:      N($obs_mean, $obs_std), enabled=$obs_enabled"
  echo "RUN_NAME:       $run_name"
  echo "TEST_NEPISODE:  $TEST_NEPISODE"
  echo "BATCH_SIZE_RUN: $BATCH_SIZE_RUN"
  echo "EVAL_EXTRA_ARGS:${EVAL_EXTRA_ARGS:+ $EVAL_EXTRA_ARGS}"
  echo "================================================================================"

  local extra_args=()
  if [[ -n "$EVAL_EXTRA_ARGS" ]]; then
    read -r -a extra_args <<< "$EVAL_EXTRA_ARGS"
  fi

  local eval_cmd=(
    python src/main.py --config="$CONFIG" --env-config="$ENV_CONFIG" with
    evaluate=True
    use_tensorboard=False
    use_wandb=False
    save_model=False
    save_replay=False
    save_evaluate_state=False
    checkpoint_path="$model_dir"
    load_step="$step"
    env_args.map_name="$map_name"
    test_nepisode="$TEST_NEPISODE"
    batch_size_run="$BATCH_SIZE_RUN"
    seed="$seed"
    device="$DEVICE"
    obs_delay_enabled="$obs_enabled"
    obs_delay_apply_train=False
    obs_delay_apply_test="$obs_enabled"
    obs_gaussian_delay_mean="$obs_mean"
    obs_gaussian_delay_std="$obs_std"
    obs_delay_discretization="$OBS_DELAY_DISCRETIZATION"
    name="$run_name"
  )
  if [[ "$CONFIG_HAS_COMM_DELAY" == "True" ]]; then
    eval_cmd+=(
      comm_gaussian_delay_mean="$comm_mean"
      comm_gaussian_delay_std="$comm_std"
    )
  fi
  if [[ "$CONFIG" == "code_qmix" ]]; then
    append_code_qmix_training_args "$model_dir" eval_cmd
  fi
  eval_cmd+=("${extra_args[@]}")

  printf "EVAL_CMD:"
  printf " %q" "${eval_cmd[@]}"
  printf "\n"

  set +e
  "${eval_cmd[@]}"
  local eval_status=$?
  set -e
  if [[ $eval_status -ne 0 ]]; then
    cleanup_eval_artifacts "$run_name" "$map_name"
    return "$eval_status"
  fi

  local extracted
  local winrate
  local log_path
  extracted="$(extract_latest_winrate "$run_name" "$map_name")"
  winrate="${extracted%%|*}"
  log_path="${extracted#*|}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$map_name" "$(basename "$model_dir")" "$step" "$seed" \
    "$comm_mean" "$comm_std" "$obs_mean" "$obs_std" "$condition" "$winrate" "$log_path" \
    >> "$RAW_SUMMARY_PATH"

  cleanup_eval_artifacts "$run_name" "$map_name"

  echo "RESULT: map=$map_name step=$step seed=$seed condition=$condition metric/test_battle_won_mean=$winrate"
}

print_delay_matrix() {
  awk -v scale="$MATRIX_VALUE_SCALE" -F '\t' '
    NR == 1 { next }
    {
      row = "N(" $7 "," $8 ")"
      col = "N(" $5 "," $6 ")"
      key = $1 "\t" $2 "\t" $3
      if ($10 != "NA") {
        sample_count[key, row, col] += 1
        sum[key, row, col] += $10
        sumsq[key, row, col] += ($10 * $10)
      }

      if (!(row in row_seen)) {
        row_seen[row] = 1
        row_order[++row_count] = row
      }
      if (!(col in col_seen)) {
        col_seen[col] = 1
        col_order[++col_count] = col
      }
      if (!(key in key_seen)) {
        key_seen[key] = 1
        key_order[++key_count] = key
      }
    }
    END {
      for (k = 1; k <= key_count; k++) {
        key = key_order[k]
        split(key, parts, "\t")
        printf "checkpoint:%s\tmap:%s\tstep:%s\tcommunication delay", parts[2], parts[1], parts[3]
        for (c = 1; c <= col_count; c++) {
          printf "\t%s", col_order[c]
        }
        printf "\n"

        printf "observation delay"
        for (c = 1; c <= col_count + 3; c++) {
          printf "\t"
        }
        printf "\n"

        for (r = 1; r <= row_count; r++) {
          row = row_order[r]
          printf "%s", row
          for (c = 1; c <= col_count; c++) {
            col = col_order[c]
            n = sample_count[key, row, col] + 0
            if (n > 0) {
              mean = sum[key, row, col] / n
              variance = (sumsq[key, row, col] / n) - (mean * mean)
              if (variance < 0 && variance > -0.000000001) variance = 0
              std = sqrt(variance)
              if (scale == "percent") {
                cell = sprintf("%.0f±%.0f", mean * 100, std * 100)
              } else {
                cell = sprintf("%.4f±%.4f", mean, std)
              }
            } else {
              cell = "NA"
            }
            printf "\t%s", cell
          }
          printf "\n"
        }
        if (k < key_count) {
          printf "\n"
        }
      }
    }
  ' "$RAW_SUMMARY_PATH"
}

# -------- Preflight --------

export CUDA_VISIBLE_DEVICES
export SC2PATH
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

[[ -d "$SC2PATH/Versions" ]] || die "StarCraft II Versions directory not found: $SC2PATH/Versions"
[[ -f "src/config/algs/$CONFIG.yaml" ]] || die "algorithm config not found: src/config/algs/$CONFIG.yaml"
[[ -f "src/config/envs/$ENV_CONFIG.yaml" ]] || die "env config not found: src/config/envs/$ENV_CONFIG.yaml"
if rg -q '^[[:space:]]*comm_gaussian_delay_mean:' "src/config/algs/$CONFIG.yaml"; then
  CONFIG_HAS_COMM_DELAY=True
fi

MODEL_SPECS=()
OUTPUT_MODEL_DIR=""
MAP_NAME="$(normalise_map_name "$MAP_NAME")"
if [[ -z "$SUMMARY_DIR" ]]; then
  SUMMARY_DIR="$SUMMARY_ROOT/$CONFIG/$MAP_NAME"
fi
mkdir -p "$SUMMARY_DIR"
printf "map\tmodel\tstep\tseed\tcomm_mean\tcomm_std\tobs_mean\tobs_std\tcondition\ttest_battle_won_mean\tlog_path\n" > "$RAW_SUMMARY_PATH"

if [[ $# -gt 0 ]]; then
  CLI_MODEL_DIR="$1"
  shift
  [[ -d "$CLI_MODEL_DIR" ]] || die "model directory does not exist: $CLI_MODEL_DIR"
  OUTPUT_MODEL_DIR="$CLI_MODEL_DIR"
  [[ -n "$MAP_NAME" ]] || die "MAP_NAME is required for single-model usage"
  if [[ $# -gt 0 ]]; then
    MODEL_SPECS+=("$MAP_NAME|$CLI_MODEL_DIR|$*")
  else
    MODEL_SPECS+=("$MAP_NAME|$CLI_MODEL_DIR|$LOAD_STEPS")
  fi
else
  MODEL_DIR="$(resolve_model_dir "$MODEL_DIR" "$MODEL_NAME")" || die "MODEL_NAME or MODEL_DIR is required"
  [[ -d "$MODEL_DIR" ]] || die "model directory does not exist: $MODEL_DIR"
  OUTPUT_MODEL_DIR="$MODEL_DIR"
  MODEL_SPECS+=("$MAP_NAME|$MODEL_DIR|$LOAD_STEPS")
fi

if [[ -z "$SUMMARY_PATH" ]]; then
  SUMMARY_PATH="$SUMMARY_DIR/$(basename "$OUTPUT_MODEL_DIR")_matrix.tsv"
fi
if [[ -z "$RAW_SUMMARY_KEEP_PATH" ]]; then
  RAW_SUMMARY_KEEP_PATH="${SUMMARY_PATH%.tsv}_raw.tsv"
fi

for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r map_name model_dir step_text <<< "$spec"
  [[ -n "$map_name" && -n "$model_dir" ]] || die "bad model spec: $spec"
  [[ -d "$model_dir" ]] || die "model directory does not exist: $model_dir"

  if [[ -z "$step_text" ]]; then
    step_text="$(closest_checkpoint "$model_dir")"
    [[ -n "$step_text" ]] || die "no numeric checkpoint directories found under: $model_dir"
  fi

  read -r -a load_steps <<< "$step_text"
  for step in "${load_steps[@]}"; do
    [[ "$step" =~ ^[0-9]+$ ]] || die "LOAD_STEP must be numeric: $step"

    for seed in $GROUP_SEEDS; do
      unset RUN_CONDITION_SEEN
      declare -A RUN_CONDITION_SEEN=()
      case "$INCLUDE_ZERO_ZERO_BASELINE" in
        True|true|1|yes|Yes)
          RUN_CONDITION_SEEN["0:0|0:0"]=1
          run_eval "$map_name" "$model_dir" "$step" "$seed" 0 0 0 0
          ;;
      esac
      for comm_pair in $COMM_DELAY_GRID; do
        comm_values="$(parse_delay_pair "$comm_pair")"
        comm_mean="${comm_values%%$'\t'*}"
        comm_std="${comm_values#*$'\t'}"

        for obs_pair in $OBS_DELAY_GRID; do
          obs_values="$(parse_delay_pair "$obs_pair")"
          obs_mean="${obs_values%%$'\t'*}"
          obs_std="${obs_values#*$'\t'}"
          condition_key="${comm_mean}:${comm_std}|${obs_mean}:${obs_std}"
          if [[ -n "${RUN_CONDITION_SEEN[$condition_key]:-}" ]]; then
            echo "SKIP duplicate delay condition: comm=$(delay_label "$comm_mean" "$comm_std") obs=$(delay_label "$obs_mean" "$obs_std")"
            continue
          fi
          RUN_CONDITION_SEEN[$condition_key]=1
          run_eval "$map_name" "$model_dir" "$step" "$seed" "$comm_mean" "$comm_std" "$obs_mean" "$obs_std"
        done
      done
    done
  done
done

echo "================================================================================"
echo "Delay matrix: metric/test_battle_won_mean mean±std across seeds ($TEST_NEPISODE episodes per seed)"
print_delay_matrix | tee "$SUMMARY_PATH"

echo "================================================================================"
echo "Delay matrix saved to: $SUMMARY_PATH"
case "$RAW_SUMMARY_KEEP" in
  True|true|1|yes|Yes)
    cp "$RAW_SUMMARY_PATH" "$RAW_SUMMARY_KEEP_PATH"
    echo "Raw summary saved to: $RAW_SUMMARY_KEEP_PATH"
    ;;
esac
echo "Done."
