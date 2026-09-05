#!/usr/bin/env bash
# Run via bash -l to inherit the system's existing SC2PATH.
set -uo pipefail

cd /home/liuhongbo/workspace/epymarl_based
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate /home/liuhongbo/.local/miniconda3/envs/marl_stable
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mode=${1:-train}
run_dir=results/remote_bcrbc_8m_20260905_seed1_sc246_batch8
mkdir -p "$run_dir"

if [ "$mode" = pilot ]; then
    steps=600
    test_episodes=2
    log_interval=100
else
    steps=2000000
    test_episodes=16
    log_interval=5000
fi

python -u src/main.py --config=bcrbc_qmix --env-config=delayed_sc2 with \
    name=bcrbc_8m_${mode}_20260905 seed=1 env_args.map_name=8m \
    runner=delayed_episode batch_size_run=1 batch_size=8 buffer_size=5000 \
    buffer_cpu_only=True use_cuda=True bcrbc_use_comm=True \
    bcrbc_generative_eval=True test_nepisode="$test_episodes" \
    test_interval=50000 log_interval="$log_interval" \
    runner_log_interval="$log_interval" learner_log_interval="$log_interval" \
    save_model_interval=100000 t_max="$steps" \
    2>&1 | tee "$run_dir/$mode.log"
status=$?
printf '%s\n' "$status" > "$run_dir/$mode.exit"
exit "$status"
