#!/bin/bash

# ! ============================================================================
# ! Should check every time before running.
# ! ============================================================================
# Experiment parameters
EXPERIMENT_NAME=pymarl2_qmix_rad_time_shifting_FiLMq   # Experiment name for logging.
CONFIG=pymarl2_qmix_FiLMq_agent_with_t  # Algorithm config name in src/config/alg
ENV_CONFIG=sc2v2  # Environment config in src/config/envs
MAP_NAME=protoss_5_vs_5  # Map name, e.g., 3m in StarCraftII.
REPEAT_TIMES=1  # Times to run the experiment.
OPTIMIZER=rad  # Optimizer name.

# arguments in different runs.
function update_hyperparams() {
    declare -n arg_dict=$1
    local iter=$2  # Notice iter starts from 1.

    # arguments before "with"
    arg_dict["config"]="--config=$CONFIG"
    arg_dict["env_config"]="--env-config=$ENV_CONFIG"

    # arguments after "with"
    arg_dict["name"]="name=${EXPERIMENT_NAME}_run$((iter))"  # name in tensorboard, sacred, and wandb

    arg_dict["map_name"]="env_args.map_name=$MAP_NAME"
    arg_dict["optimizer"]="optimizer=$OPTIMIZER"
    }
# ! ============================================================================
# ? ============================================================================
# ? Should check before running experiments in a new environment.
# ? ============================================================================
# Set environment variable
CONDA_ENVNAME="RAD_Opt"                        # Conda environment name

# export SC2PATH="$HOME/.local/share/StarCraftII"  # Path to StarCraft II game.

CUDA_DEVICES=0  # Set visible devices for scripts.
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Paths
WORK_DIR="$HOME/workspace/epymarl_based"    # Path to work dir
LOG_DIR="$WORK_DIR/log"     # Log directory
PYTHON_SCRIPT="src/main.py"     # Relative path to python script in work dir.
PYTHON_SCRIPT_PATH="${WORK_DIR}/${PYTHON_SCRIPT}"

BUFFER_CPU_ONLY=True  # Whether to use CPU only buffer.
DEVICE="cuda"  # Device to use for training.

# arguments for different environments.
function update_envparams() {
    declare -n arg_dict=$1
    local iter=$2  # Notice iter starts from 1.

    arg_dict["buffer_cpu_only"]="buffer_cpu_only=$BUFFER_CPU_ONLY"
    arg_dict["device"]="device=$DEVICE"
}
# ? ============================================================================

SEPERATOR="------------------------------------------------------------------------------------------------------------------------"

# Create log directory if it doesn't exist
if ! cd "$WORK_DIR"; then
    echo "Work dir: $WORK_DIR not found."
    exit 1
fi

mkdir -p "$LOG_DIR"

timestamp() {
    date +"%Y-%m-%d_%H-%M-%S"
}

# Python args passed to the Python command
declare -A args

log_file_path="$LOG_DIR/${EXPERIMENT_NAME}_$(timestamp)_log.log"
logerr_file_path="$LOG_DIR/${EXPERIMENT_NAME}_$(timestamp)_err.log"

# Generate log file names
echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash         | Train script starts." | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash         | Experiment name: $EXPERIMENT_NAME" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash         | Work dir: $WORK_DIR" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash         | Saving outputs to log files in $LOG_DIR" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash         | Running Python script in conda environment: $CONDA_ENVNAME." | tee -a "$log_file_path"

# Define function to run experiment
run_experiment() {
    update_hyperparams args "$1"
    update_envparams args "$1"

    echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash         | Run $i" | tee -a "$log_file_path"

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

    cmd="conda run -n $CONDA_ENVNAME --no-capture-output python ${PYTHON_SCRIPT_PATH} ${pre_args}with $post_args"
    echo "$(timestamp) | INFO     | bash         | Command: $cmd" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash         | Starting train process."
    
    # Run the Python script
    if ! eval "$cmd" 1>> "$log_file_path" 2>> "$logerr_file_path"; then
        echo "$(timestamp) | ERROR    | root         | Run Failed, see $logerr_file_path for more information." | tee -a "$log_file_path" "$logerr_file_path"
        echo "$(timestamp) | ERROR    | bash         | Train script ends." | tee -a "$log_file_path" "$logerr_file_path"
        echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$log_file_path" "$logerr_file_path"
        echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$log_file_path" "$logerr_file_path"
        exit 1
    fi

    echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | root         | Run $i finished." | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$log_file_path"

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

echo "$(timestamp) | INFO     | bash         | Train script ends." | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash         | $SEPERATOR" | tee -a "$log_file_path"
