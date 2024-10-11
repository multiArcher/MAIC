# Set environment variable
$env:SC2PATH = $env:SC2PATH  # Path to Star Craft 2 game.

$CONDA_ENVNAME = "marl_base"    # Conda environment name

# Paths
$WORK_DIR = "$HOME/Projects/pymarl2_based"   # Path to root dir
$PYTHON_SCRIPT = "$HOME/Projects/pymarl2_based/src/main.py"  # Path to python script

Write-Host "Work dir: $WORK_DIR"
Set-Location -Path $WORK_DIR
if (-not $?) {
    Write-Host "Failed to change directory to $WORK_DIR. Exiting script." -ForegroundColor Red
    exit 1
}

# Python arguments passed to python command
$python_args = @(
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
Write-Host "Running Python script in conda environment: $CONDA_ENVNAME"
conda run -n $CONDA_ENVNAME --no-capture-output python $python_args
