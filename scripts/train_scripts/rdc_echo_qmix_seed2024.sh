#!/usr/bin/env bash
set -euo pipefail

PREDICTOR_MODEL="${1:-${PREDICTOR_MODEL:-gru}}"
case "$PREDICTOR_MODEL" in
    gru)
        NUM_MINIBATCH_PREDICTOR=2
        ;;
    transformer)
        NUM_MINIBATCH_PREDICTOR=4
        ;;
    *)
        echo "Usage: $0 [gru|transformer]" >&2
        exit 2
        ;;
esac

EXPERIMENT_NAME="rdc_5m6m_echo_qmix_${PREDICTOR_MODEL}_seed2024"
WORK_DIR="${WORK_DIR:-$HOME/workspace/epymarl_based}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/epymarl/bin/python}"
SCRIPT_PATH="$(readlink -f "$0")"
SCREEN_SESSION="${SCREEN_SESSION:-$EXPERIMENT_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SC2PATH="${SC2PATH:-$HOME/.local/share/StarCraftII}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

if [[ "${RDC_IN_SCREEN:-0}" != "1" ]]; then
    if ! command -v screen >/dev/null 2>&1; then
        echo "screen not found. Install screen first, then rerun this script." >&2
        exit 1
    fi
    if screen -list | grep -q "\.${SCREEN_SESSION}[[:space:]]"; then
        echo "screen session already exists: $SCREEN_SESSION" >&2
        exit 1
    fi
    screen -dmS "$SCREEN_SESSION" bash -lc \
        "RDC_IN_SCREEN=1 WORK_DIR='$WORK_DIR' PYTHON_BIN='$PYTHON_BIN' CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' bash '$SCRIPT_PATH' '$PREDICTOR_MODEL'"
    echo "Started RDC training in screen session: $SCREEN_SESSION"
    echo "Attach with: screen -r $SCREEN_SESSION"
    exit 0
fi

cd "$WORK_DIR"
mkdir -p log
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STD_LOG="log/${EXPERIMENT_NAME}_${TIMESTAMP}_log.log"
ERR_LOG="log/${EXPERIMENT_NAME}_${TIMESTAMP}_err.log"

echo "Experiment: $EXPERIMENT_NAME" | tee -a "$STD_LOG"
echo "Predictor: $PREDICTOR_MODEL" | tee -a "$STD_LOG"
echo "Python: $PYTHON_BIN" | tee -a "$STD_LOG"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" | tee -a "$STD_LOG"

"$PYTHON_BIN" -u src/main.py \
    --config=rdc_qmix \
    --env-config=sc2 \
    with \
    name="$EXPERIMENT_NAME" \
    seed=2024 \
    env_args.map_name=5m_vs_6m \
    env_args.obs_last_action=False \
    env_args.state_last_action=True \
    predictor_model="$PREDICTOR_MODEL" \
    num_minibatch_predictor="$NUM_MINIBATCH_PREDICTOR" \
    transformer_structure=encoder-decoder \
    obs_delay_enabled=True \
    obs_delay_apply_train=False \
    obs_delay_apply_test=True \
    obs_gaussian_delay_mean=3.0 \
    obs_gaussian_delay_std=2.0 \
    obs_delay_discretization=round \
    obs_delay_max=6 \
    1>>"$STD_LOG" 2>>"$ERR_LOG"
