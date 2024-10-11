#!/bin/bash

# Set environment variable
export SC2PATH="$HOME/.local/share/StarCraftII"  # Path to Star Craft 2 game.

CONDA_ENVNAME="marl_base"  # Conda environment name

# Paths
WORK_DIR="$HOME/projects/pymarl2_based"   # Path to root dir
PYTHON_SCRIPT="$HOME/projects/pymarl2_based/src/main.py" # Path to python script

echo "Work dir: $WORK_DIR"
if ! cd "$WORK_DIR"; then
    echo "Failed to change directory to $WORK_DIR. Exiting script."
    exit 1
fi

# Python args passed to python command
args=(
    "$PYTHON_SCRIPT"
    "--config=reward_shaping"   #  src/config/alg
    "--env-config=sc2"  # src/config/envs
    "with"
    'name="rsqmix_p1"'
    "batch_size_run=4"
    "batch_size=64"
#    "buffer_cpu_only=False"
    'env_args.map_name="MMM2"'
    "env_args.window_size_x=800"
    "env_args.window_size_y=600"
)

# Run the python script in the conda environment
echo "Running Python script in conda environment: $CONDA_ENVNAME"
conda run -n $CONDA_ENVNAME --no-capture-output python "${args[@]}"
