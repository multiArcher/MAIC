#! Not tested generated code. Can run but bot reliable.
# Experiment parameters
$EXPERIMENT_NAME = "qmix_repeat_training"  # Experiment name for tensorboard, sacred, and wandb.
$CONFIG = "reward_shaping"                 # Algorithm config name in src/config/alg
$ENV_CONFIG = "sc2"                        # Environment config in src/config/envs
$MAP_NAME = "MMM2"                         # Map name, e.g., 3M in StarCraftII.
$REPEAT_TIMES = 1                          # Times to run the experiment.

# Paths
$WORK_DIR = "$HOME\WorkSpace\pymarl2_based"  # Path to root dir
$PYTHON_SCRIPT = "$WORK_DIR\src\main.py"     # Path to python script
$LOG_DIR = "$WORK_DIR\log"                   # Log directory

# Python args passed to the Python command
$script_path = $PYTHON_SCRIPT
$pyton_args = @{
    "config"     = "--config=$CONFIG"
    "env_config" = "--env-config=$ENV_CONFIG"

    "name"       = "name=$EXPERIMENT_NAME"
    "map_name"   = "env_args.map_name=$MAP_NAME"
    "batch_size_run" = "batch_size_run=2"

}

# Set environment variable
$env:SC2PATH = "D:\Games\Data\StarCraftII\StarCraft II"  # Path to StarCraft II game.
$CONDA_ENVNAME = "marl_base"                     # Conda environment name

# Create log directory if it doesn't exist
if (!(Test-Path -Path $LOG_DIR)) {
    New-Item -Path $LOG_DIR -ItemType Directory
}

# Function to generate timestamp
function Get-Timestamp {
    return Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
}

$log_file_path = "$LOG_DIR\$EXPERIMENT_NAME$((Get-Timestamp))_log.log"
$logerr_file_path = "$LOG_DIR\$EXPERIMENT_NAME$((Get-Timestamp))_err.log"

# Output initial log
Write-Host "$(Get-Timestamp) | INFO     | PS       | Saving outputs to log file $log_file_path"

# Function to run experiment
function Run-Experiment {
    param(
        [int]$i
    )

    Write-Host "$(Get-Timestamp) | INFO     | PS       | ################################################################################" | Tee-Object -FilePath $log_file_path -Append
    Write-Host "$(Get-Timestamp) | INFO     | PS       | Run $i" | Tee-Object -FilePath $log_file_path -Append
    Write-Host "$(Get-Timestamp) | INFO     | PS       | Running Python script in conda environment: $CONDA_ENVNAME." | Tee-Object -FilePath $log_file_path -Append
    Write-Host "$(Get-Timestamp) | INFO     | PS       | Work dir: $WORK_DIR" | Tee-Object -FilePath $log_file_path -Append
    Write-Host "$(Get-Timestamp) | INFO     | PS       | ################################################################################" | Tee-Object -FilePath $log_file_path -Append

    # Construct command arguments
    $pre_args = ""
    $post_args = ""

    foreach ($key in $pyton_args.Keys) {
        if ($key -ne "script_path") {
            if ($pyton_args[$key] -match "^--") {
                $pre_args += "$($pyton_args[$key]) "
            } else {
                $post_args += "$($pyton_args[$key]) "
            }
        }
    }

    $cmd = "conda run -n $CONDA_ENVNAME --no-capture-output python $script_path $pre_args with $post_args"
    Write-Host "$(Get-Timestamp) | INFO     | PS       | Command: $cmd" | Tee-Object -FilePath $log_file_path -Append
    Write-Host "$(Get-Timestamp) | INFO     | PS       | Starting train process."

    # Run the Python script
    try {
        Invoke-Expression $cmd | Tee-Object -FilePath $log_file_path -Append
        Write-Host "$(Get-Timestamp) | INFO     | PS       | ################################################################################" | Tee-Object -FilePath $log_file_path -Append
        Write-Host "$(Get-Timestamp) | INFO     | PS       | Run $i finished." | Tee-Object -FilePath $log_file_path -Append
        Write-Host "$(Get-Timestamp) | INFO     | PS       | ################################################################################" | Tee-Object -FilePath $log_file_path -Append
    }
    catch {
        Write-Host "$(Get-Timestamp) | ERROR    | PS       | Run Failed, see $logerr_file_path for more information." | Tee-Object -FilePath $log_file_path $logerr_file_path -Append
        Exit 1
    }
}

# Run experiments
for ($i=1; $i -le $REPEAT_TIMES; $i++) {
    Run-Experiment -i $i
}

# Uncomment for parallel run experiments
# Start-Job {
#     for ($i=1; $i -le $REPEAT_TIMES; $i++) {
#         Run-Experiment -i $i
#     }
# }
# Wait-Job
