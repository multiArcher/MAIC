#!/bin/bash

# ! ============================================================================
# ! Should check every time before running.
# ! ============================================================================
# Experiment parameters
EXPERIMENT_NAME=qmix_mmm2_baseline_rand_seed  # Experiment name for logging.
CONFIG=qmix  # Algorithm config name in src/config/alg
ENV_CONFIG=sc2  # Environment config in src/config/envs
MAP_NAME=MMM2  # Map name, e.g., 3m in StarCraftII.
REPEAT_TIMES=1  # Times to run the experiment.
# Do not pass seed. EPyMARL will use its default random seed behavior.

# MMM2 is relatively heavy locally. Keep environment parallelism conservative.
BATCH_SIZE_RUN=4
BATCH_SIZE=64
BUFFER_SIZE=5000

# Keep the standard QMIX observation inputs explicit.
OBS_AGENT_ID=True
OBS_LAST_ACTION=True

# Keep core QMIX hyperparameters explicit for reproducibility.
LR=0.0005
GAMMA=0.99
EPSILON_ANNEAL_TIME=100000
TARGET_UPDATE_INTERVAL_OR_TAU=200
T_MAX=10050000

# arguments in different runs.
function update_hyperparams() {
    declare -n arg_dict=$1
    local iter=$2  # * Should notice that iter starts from 1.

    # arguments before "with"
    arg_dict["config"]="--config=$CONFIG"
    arg_dict["env_config"]="--env-config=$ENV_CONFIG"

    # arguments after "with"
    arg_dict["name"]="name=${EXPERIMENT_NAME}_run$((iter))"
    arg_dict["map_name"]="env_args.map_name=$MAP_NAME"

    arg_dict["batch_size_run"]="batch_size_run=$BATCH_SIZE_RUN"
    arg_dict["batch_size"]="batch_size=$BATCH_SIZE"
    arg_dict["buffer_size"]="buffer_size=$BUFFER_SIZE"

    arg_dict["obs_agent_id"]="obs_agent_id=$OBS_AGENT_ID"
    arg_dict["obs_last_action"]="obs_last_action=$OBS_LAST_ACTION"

    arg_dict["lr"]="lr=$LR"
    arg_dict["gamma"]="gamma=$GAMMA"
    arg_dict["epsilon_anneal_time"]="epsilon_anneal_time=$EPSILON_ANNEAL_TIME"
    arg_dict["target_update_interval_or_tau"]="target_update_interval_or_tau=$TARGET_UPDATE_INTERVAL_OR_TAU"
    arg_dict["t_max"]="t_max=$T_MAX"
}
# ! ============================================================================
# ? ============================================================================
# ? Should check before running experiments in a new environment.
# ? ============================================================================
# Set environment variable
CONDA_ENV_NAME="epymarl"

if [ -z "$SC2PATH" ]; then
    export SC2PATH="$HOME/.local/share/StarCraftII"
fi

export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"

# Set CUDA devices
CUDA_DEVICES=0

# Paths
if [[ -d "$HOME/autodl-tmp/epymarl_based" ]]; then
    WORK_DIR="$HOME/autodl-tmp/epymarl_based"
else
    WORK_DIR="$HOME/workspace/epymarl_based"
fi
LOG_DIR="$WORK_DIR/log"
PYTHON_SCRIPT="src/main.py"

SCRIPT_PATH="$(readlink -f "$0")"
SCREEN_SESSION="${SCREEN_SESSION:-${EXPERIMENT_NAME}}"

if [[ "${IN_TRAIN_SCREEN:-0}" != "1" ]]; then
    if ! command -v screen >/dev/null 2>&1; then
        echo "screen not found. Install screen first, then rerun this script." >&2
        exit 1
    fi
    if screen -list | grep -q "\\.${SCREEN_SESSION}[[:space:]]"; then
        echo "screen session already exists: $SCREEN_SESSION" >&2
        echo "Attach with: screen -r $SCREEN_SESSION"
        exit 1
    fi

    screen -dmS "$SCREEN_SESSION" bash -lc "
        source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" &&
        conda activate \"$CONDA_ENV_NAME\" &&
        cd \"$WORK_DIR\" &&
        IN_TRAIN_SCREEN=1 exec bash \"$SCRIPT_PATH\"
    "

    echo "Started training in detached screen session: $SCREEN_SESSION"
    echo "Attach with: screen -r $SCREEN_SESSION"
    echo "Detach after attaching: Ctrl-a d"
    echo "List sessions: screen -ls"
    exit 0
fi

# Environment parameters passed to the Python script.
BUFFER_CPU_ONLY=True
DEVICE=cuda

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
cd "$WORK_DIR" || exit

# Check if Python script exists
if [[ $PYTHON_SCRIPT = /* ]]; then
    PYTHON_SCRIPT_PATH="$PYTHON_SCRIPT"
else
    PYTHON_SCRIPT_PATH="$WORK_DIR/$PYTHON_SCRIPT"
fi

if ! [[ -e $PYTHON_SCRIPT_PATH ]]; then
    echo "$(timestamp) | FATAL    | bash         | Run Failed, $PYTHON_SCRIPT_PATH not found."
    exit 1
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
        if [ "$key" != "script_path" ]; then
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

echo "$(timestamp) | INFO     | bash         | Train script ends." | tee -a "$std_log_path"
echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path"
echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$std_log_path"
