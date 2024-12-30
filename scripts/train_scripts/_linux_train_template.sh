#!/bin/bash

# ! ============================================================================
# ! Should check every time before running.
# ! ============================================================================
# Experiment parameters
EXPERIMENT_NAME=_linux_train_template  # Experiment name for logging.
CONFIG=qmix  # Algorithm config name in src/config/alg
ENV_CONFIG=sc2  # Environment config in src/config/envs
MAP_NAME=3m  # Map name, e.g., 3m in StarCraftII.
REPEAT_TIMES=3  # Times to run the experiment.

# arguments in different runs.
function update_hyperparams() {
    declare -n arg_dict=$1
    local iter=$2  # * Should notice that iter starts from 1.

    # arguments before "with"
    arg_dict["config"]="--config=$CONFIG"
    arg_dict["env_config"]="--env-config=$ENV_CONFIG"

    # arguments after "with"
    arg_dict["name"]="name=${EXPERIMENT_NAME}_run$((iter))"  # name in tensorboard, sacred, and wandb

    arg_dict["map_name"]="env_args.map_name=$MAP_NAME"
    }
# ! ============================================================================
# ? ============================================================================
# ? Should check before running experiments in a new environment.
# ? ============================================================================
# Set environment variable
CONDA_ENV_NAME="marl_latest"                        # Conda environment name

if [ -z "$SC2PATH" ]; then
    export SC2PATH="$HOME/.local/share/StarCraftII"  # Path to StarCraft II game.
fi

# Set CUDA devices
CUDA_DEVICES=0  # Set visible devices for scripts.

# Paths
WORK_DIR="$HOME/workspace/epymarl_based"    # Path to work dir
LOG_DIR="$WORK_DIR/log"     # Log directory for logging terminal outputs.
PYTHON_SCRIPT="src/main.py"     # Path to python script in work dir. Can be absolute or relative to work dir.

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

if ! [[ -e $SCRIPT_PATH ]]; then
    echo "$(timestamp) | FATAL    | bash         | Run Failed, $PYTHON_SCRIPT_PATH not found." | tee -a "$LOGFILE"
fi

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

timestamp() {
    date +"%Y-%m-%d_%H-%M-%S"
}

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
