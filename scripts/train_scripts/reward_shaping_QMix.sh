#!/bin/bash

# Experiment parameters
EXPERIMENT_NAME=qmix_repeat_training  # Experiment name for tensorboard, sacred, and wandb.
CONFIG=reward_shaping                 # Algorithm config name in src/config/alg
ENV_CONFIG=sc2                        # Environment config in src/config/envs
MAP_NAME=MMM2                         # Map name, e.g., 3M in StarCraftII.
REPEAT_TIMES=1                        # Times to run the experiment.

# Paths
WORK_DIR="$HOME/workspace/pymarl2_based"  # Path to root dir
PYTHON_SCRIPT="$WORK_DIR/src/main.py"     # Path to python script
LOG_DIR="$WORK_DIR/log"                   # Log directory

# Python args passed to the Python command
declare -A args

args["script_path"]="$PYTHON_SCRIPT"

args["config"]="--config=$CONFIG"
args["env_config"]="--env-config=$ENV_CONFIG"

args["name"]="name=$EXPERIMENT_NAME"
args["map_name"]="env_args.map_name=$MAP_NAME"

# Set environment variable
export SC2PATH="$HOME/.local/share/StarCraftII"  # Path to StarCraft II game.
CONDA_ENVNAME="marl_base"                        # Conda environment name

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

timestamp() { 
    date +"%Y-%m-%d_%H-%M-%S" 
}

log_file_path="$LOG_DIR/${EXPERIMENT_NAME}_$(timestamp)_log.log"
logerr_file_path="$LOG_DIR/${EXPERIMENT_NAME}_$(timestamp)_err.log"

# Generate log file names
echo "$(timestamp) | INFO     | bash     | Saving outputs to log file $log_file_path" 

# Define function to run experiment
run_experiment() {
    local i=$1
    echo "$(timestamp) | INFO     | bash     | ################################################################################" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash     | Run $i" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash     | Running Python script in conda environment: $CONDA_ENVNAME." | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash     | Work dir: $WORK_DIR" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash     | ################################################################################" | tee -a "$log_file_path"

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

    cmd="conda run -n $CONDA_ENVNAME --no-capture-output python ${args["script_path"]} ${pre_args}with $post_args"
    echo "$(timestamp) | INFO     | bash     | Command: $cmd" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash     | Starting train process."
    
    # Run the Python script
    if ! eval "$cmd" 1>> "$log_file_path" 2>> "$logerr_file_path"; then
        echo "$(timestamp) | ERROR    | root     | Run Failed, see $logerr_file_path for more information." | tee -a "$log_file_path" "$logerr_file_path"
        exit 1
    fi

    echo "$(timestamp) | INFO     | bash     | ################################################################################" | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | root     | Run $i finished." | tee -a "$log_file_path"
    echo "$(timestamp) | INFO     | bash     | ################################################################################" | tee -a "$log_file_path"

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
