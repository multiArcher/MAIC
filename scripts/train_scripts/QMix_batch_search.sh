#!/bin/bash

# Outdated script, not used anymore.
# Paths
WORK_DIR="$HOME/workspace/epymarl_based"  # Path to root dir
PYTHON_SCRIPT="src/main.py"     # Path to python script

# Experiment parameters
# ! ============================================================================
# ! Should check every time before running.
# ! ============================================================================
EXPERIMENT_NAME=QMIX_batch_search  # Experiment name for tensorboard, sacred, and wandb.
CONFIG=qmix                # Algorithm config name in src/config/alg
ENV_CONFIG=sc2                        # Environment config in src/config/envs
MAP_NAME=MMM2                       # Map name, e.g., 3m in StarCraftII.
REPEAT_TIMES=6                        # Times to run the experiment.

BUFFER_CPU_ONLY=False
DEVICE=cuda:1

BATCH_SIZE_LIST=(64 64 128 128 128 96)

function update_hyperparams() {
    declare -n arg_dict=$1
    local iter=$2   # Note that iter starts from 1.
    local batch_size=${BATCH_SIZE_LIST[$iter-1]}

    arg_dict["name"]="name=${EXPERIMENT_NAME}_batch$batch_size"
    arg_dict["batch_size"]="batch_size=$batch_size"
}
# ! ============================================================================
# ! ============================================================================

LOG_DIR="$WORK_DIR/log"                   # Log directory
PYTHON_SCRIPT_PATH="${WORK_DIR}/${PYTHON_SCRIPT}"

# Python args passed to the Python command
declare -A args

args["config"]="--config=$CONFIG"
args["env_config"]="--env-config=$ENV_CONFIG"

args["map_name"]="env_args.map_name=$MAP_NAME"
args["buffer_cpu_only"]="buffer_cpu_only=$BUFFER_CPU_ONLY"
args["device"]="device=$DEVICE"

# Set environment variable
export SC2PATH="$HOME/.local/share/StarCraftII"  # Path to StarCraft II game.
CONDA_ENVNAME="marl_base"                        # Conda environment name

SEPERATOR="========================================================================================================================"
# Create log directory if it doesn't exist
cd "$WORK_DIR" || exit 1
mkdir -p "$LOG_DIR"

timestamp() { 
    date +"%Y-%m-%d_%H-%M-%S" 
}

log_file_path="$LOG_DIR/${EXPERIMENT_NAME}_$(timestamp)_log.log"
logerr_file_path="$LOG_DIR/${EXPERIMENT_NAME}_$(timestamp)_err.log"

# Generate log file names
echo "$(timestamp) | INFO     | bash     | $SEPERATOR" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash     | $SEPERATOR" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash     | Train script starts." | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash     | Experiment name: $EXPERIMENT_NAME" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash     | Work dir: $WORK_DIR" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash     | Saving outputs to log files in $LOG_DIR" 
echo "$(timestamp) | INFO     | bash     | Running Python script in conda environment: $CONDA_ENVNAME." | tee -a "$log_file_path"

# Define function to run experiment
run_experiment() {
    update_hyperparams args "$1"

    echo "$(timestamp) | INFO     | bash     | $SEPERATOR" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash     | Run $i" | tee -a "$log_file_path"

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
    echo "$(timestamp) | INFO     | bash     | Command: $cmd" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash     | $SEPERATOR" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash     | Starting train process."
    
    # Run the Python script
    if ! eval "$cmd" 1>> "$log_file_path" 2>> "$logerr_file_path"; then
        echo "$(timestamp) | ERROR    | root     | Run Failed, see $logerr_file_path for more information." | tee -a "$log_file_path" "$logerr_file_path"
        echo "$(timestamp) | ERROR    | bash     | Train script ends." | tee -a "$log_file_path" "$logerr_file_path"
        echo "$(timestamp) | INFO     | bash     | $SEPERATOR" | tee -a "$log_file_path" "$logerr_file_path"
        echo "$(timestamp) | INFO     | bash     | $SEPERATOR" | tee -a "$log_file_path" "$logerr_file_path"
        exit 1
    fi

    echo "$(timestamp) | INFO     | bash     | $SEPERATOR" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | root     | Run $i finished." | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash     | $SEPERATOR" | tee -a "$log_file_path"

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

echo "$(timestamp) | INFO     | bash     | Train script ends." | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash     | $SEPERATOR" | tee -a "$log_file_path"
echo "$(timestamp) | INFO     | bash     | $SEPERATOR" | tee -a "$log_file_path"
