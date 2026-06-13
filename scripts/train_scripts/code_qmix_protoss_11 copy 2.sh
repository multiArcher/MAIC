#!/bin/bash

# ! ============================================================================
# ! Should check every time before running.
# ! ============================================================================
# Experiment parameters
EXPERIMENT_NAME=code_qmix_protoss_5_vs_5_baseline_seed2024  # Experiment name for logging.
CONFIG=code_qmix  # Algorithm config name in src/config/alg
ENV_CONFIG=sc2v2  # Environment config in src/config/envs
MAP_NAME=protoss_5_vs_5  # Map name, e.g., 3m in StarCraftII.
REPEAT_TIMES=1  # Times to run the experiment.
BATCH_SIZE_RUN=4 # Batch size for each run, which is used to calculate the total batch size as BATCH_SIZE_RUN * REPEAT_TIMES.
COMM_GAUSSIAN_DELAY_MEAN=0  # delay mean of communication.
COMM_GAUSSIAN_DELAY_STD=0   # delay std of communication.

TD_LOSS_WEIGHT=1.0        # Weight for TD loss
ACTION_LOSS_WEIGHT=0.01    # Weight for inference loss (future action prediction)
CONTINUE_LOSS_WEIGHT=0.01    # Weight for continuity loss (intent stability)
AUX_LOSS_WEIGHT=0.1     # Weight for KL divergence loss (intent regularization)
ENTROPY_LOSS_WEIGHT=0.01  # Weight for attention entropy regularization

PREDICT_K_FUTURE_ACTIONS=5   # K for future action prediction (L_inf)
TEMPORAL_DISCOUNT_GAMMA_T=0.9  # Used by agent for timeliness alignment
SEED=2024  # Fixed seed for comparing this small hyperparameter sweep.

# arguments in different runs.
function update_hyperparams() {
    declare -n arg_dict=$1
    local iter=$2  # * Should notice that iter starts from 1.

    # arguments before "with"
    arg_dict["config"]="--config=$CONFIG"
    arg_dict["env_config"]="--env-config=$ENV_CONFIG"

    # arguments after "with"
    arg_dict["name"]="name=${EXPERIMENT_NAME}_run$((iter))"  # name in tensorboard, sacred, and wandb
    arg_dict["seed"]="seed=$SEED"
    arg_dict["comm_gaussian_delay_mean"]="comm_gaussian_delay_mean=$COMM_GAUSSIAN_DELAY_MEAN"
    arg_dict["comm_gaussian_delay_std"]="comm_gaussian_delay_std=$COMM_GAUSSIAN_DELAY_STD"
    arg_dict["batch_size_run"]="batch_size_run=$BATCH_SIZE_RUN"

    arg_dict["td_loss_weight"]="td_loss_weight=$TD_LOSS_WEIGHT"
    arg_dict["action_loss_weight"]="action_loss_weight=$ACTION_LOSS_WEIGHT"
    arg_dict["continue_loss_weight"]="continue_loss_weight=$CONTINUE_LOSS_WEIGHT"
    arg_dict["aux_loss_weight"]="aux_loss_weight=$AUX_LOSS_WEIGHT"
    arg_dict["entropy_loss_weight"]="entropy_loss_weight=$ENTROPY_LOSS_WEIGHT"
    arg_dict["predict_k_future_actions"]="predict_k_future_actions=$PREDICT_K_FUTURE_ACTIONS"
    arg_dict["temporal_discount_gamma_t"]="temporal_discount_gamma_T=$TEMPORAL_DISCOUNT_GAMMA_T"

    arg_dict["map_name"]="env_args.map_name=$MAP_NAME"
    }
# ! ============================================================================
# ? ============================================================================
# ? Should check before running experiments in a new environment.
# ? ============================================================================
# Set environment variable
CONDA_ENV_NAME="epymarl"                        # Conda environment name

if [ -z "$SC2PATH" ]; then
    export SC2PATH="$HOME/.local/share/StarCraftII"  # Path to StarCraft II game.
fi

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

SMACV2_MAP_PATH="$SC2PATH/Maps/SMAC_Maps/32x32_flat.SC2Map"
if [[ ! -f "$SMACV2_MAP_PATH" ]]; then
    echo "$(date +"%Y-%m-%d_%H-%M-%S") | FATAL    | bash         | SMACv2 map not found: $SMACV2_MAP_PATH" >&2
    echo "Install SMACv2 maps with:" >&2
    echo "  SC2PATH=\"$SC2PATH\" bash scripts/other_scripts/install_smacv2_maps.sh" >&2
    exit 1
fi

# Set CUDA devices
CUDA_DEVICES=0  # Set visible devices for scripts.

# Paths
if [[ -d "$HOME/autodl-tmp/epymarl_based" ]]; then
    WORK_DIR="$HOME/autodl-tmp/epymarl_based"
else
    WORK_DIR="$HOME/workspace/epymarl_based"
fi
LOG_DIR="$WORK_DIR/log"     # Log directory for logging terminal outputs.
PYTHON_SCRIPT="src/main.py"     # Path to python script in work dir. Can be absolute or relative to work dir.

# Environment parameters passed to the Python script.
BUFFER_CPU_ONLY="${BUFFER_CPU_ONLY:-False}"
REQUESTED_DEVICE="${DEVICE:-cuda}"
if [[ "$REQUESTED_DEVICE" == "cuda" ]]; then
    if conda run -n "$CONDA_ENV_NAME" --no-capture-output python - <<'PY' >/dev/null 2>&1
import sys
import torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
    then
        DEVICE=cuda
    else
        echo "$(date +"%Y-%m-%d_%H-%M-%S") | WARNING  | bash         | CUDA requested but unavailable; falling back to CPU."
        DEVICE=cpu
        BUFFER_CPU_ONLY=True
    fi
else
    DEVICE="$REQUESTED_DEVICE"
fi

# arguments for different environments.
function update_env_params() {
    declare -n arg_dict=$1
    local iter=$2  # Notice iter starts from 1.

    arg_dict["buffer_cpu_only"]="buffer_cpu_only=$BUFFER_CPU_ONLY"
    arg_dict["device"]="device=$DEVICE"
}
# ? ============================================================================

SEPERATOR="------------------------------------------------------------------------------------------------------------------------"

timestamp() {
    date +"%Y-%m-%d_%H-%M-%S"
}

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Check if work dir exists
if ! [[ -d $WORK_DIR ]]; then
    echo "Work dir: $WORK_DIR not found. Making directory."
    mkdir -p "$WORK_DIR"
fi
cd "$WORK_DIR"

# Check if Python script exists
if [[ $PYTHON_SCRIPT = /* ]]; then
    PYTHON_SCRIPT_PATH="$PYTHON_SCRIPT"
else
    PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT" # Path to train script.
fi

if ! [[ -e $PYTHON_SCRIPT_PATH ]]; then
    echo "$(timestamp) | FATAL    | bash         | Run Failed, $PYTHON_SCRIPT_PATH not found." | tee -a "$LOGFILE"
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Python args passed to the Python command
declare -A args

std_log_path="$LOG_DIR/${EXPERIMENT_NAME}_$(timestamp)_log.log"
err_log_path="$LOG_DIR/${EXPERIMENT_NAME}_$(timestamp)_err.log"

# Generate log file names
echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path"
echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path"
echo "$(timestamp) | INFO     | bash         | Train script starts." | tee -a "$std_log_path"
echo "$(timestamp) | INFO     | bash         | Experiment name: $EXPERIMENT_NAME" | tee -a "$std_log_path"
echo "$(timestamp) | INFO     | bash         | Work dir: $WORK_DIR" | tee -a "$std_log_path"
echo "$(timestamp) | INFO     | bash         | Saving outputs to log files in $LOG_DIR" | tee -a "$std_log_path"
echo "$(timestamp) | INFO     | bash         | Running Python script in conda environment: $CONDA_ENV_NAME." | tee -a "$std_log_path"

# Define function to run experiment
run_experiment() {
    update_hyperparams args "$1"
    update_env_params args "$1"

    echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path"
    echo "$(timestamp) | INFO     | bash         | Run $i" | tee -a "$std_log_path"

    # Construct command arguments
    local pre_args=""
    local post_args=""

    # Iterate over args to construct the command
    for key in "${!args[@]}"; do
        # Skip 'script_path' key
        if [ "$key" != "script_path" ]; then
            # Append pre_args or post_args based on key
            if [[ "${args[$key]}" == --* ]]; then
                pre_args+="${args[$key]} "
            else
                post_args+="${args[$key]} "
            fi
        fi
    done

    cmd="conda run -n ${CONDA_ENV_NAME} --no-capture-output python ${PYTHON_SCRIPT_PATH} ${pre_args}with $post_args"
    echo "$(timestamp) | INFO     | bash         | Command: $cmd" | tee -a "$std_log_path"
    echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path"
    echo "$(timestamp) | INFO     | bash         | Starting train process."

    # Run the Python script
    if ! eval "$cmd" 1>> "$std_log_path" 2>> "$err_log_path"; then
        echo "$(timestamp) | FATAL    | bash         | Run Failed, see $err_log_path for more information." | tee -a "$std_log_path" "$err_log_path"
        echo "$(timestamp) | FATAL    | bash         | Train script ends." | tee -a "$std_log_path" "$err_log_path"
        echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path" "$err_log_path"
        echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path" "$err_log_path"
        exit 1
    fi

    echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path"
    echo "$(timestamp) | INFO     | bash         | Run $i finished." | tee -a "$std_log_path"
    echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path"

}

# Run experiments
for ((i=1; i<=REPEAT_TIMES; i++)); do
    run_experiment $i
done

# # Parallel run experiments
# for ((i=1; i<=REPEAT_TIMES; i++)); do
#     run_experiment $i &
# done
# wait

echo "$(timestamp) | INFO     | bash         | Train script ends." | tee -a "$std_log_path"
echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path"
echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path"
