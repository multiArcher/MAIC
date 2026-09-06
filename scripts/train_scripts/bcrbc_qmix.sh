#!/usr/bin/env bash
# Run from the project root with marl_stable activated; inherit SC2PATH.

EXPERIMENT_NAME=bcrbc_8m
MAP_NAME=8m
SEED=1
BATCH_SIZE_RUN=4
BATCH_SIZE=8
T_MAX=2000000

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u src/main.py --config=bcrbc_qmix --env-config=delayed_sc2 with \
    name="$EXPERIMENT_NAME" env_args.map_name="$MAP_NAME" seed="$SEED" \
    runner=delayed_parallel batch_size_run="$BATCH_SIZE_RUN" \
    batch_size="$BATCH_SIZE" buffer_size=5000 buffer_cpu_only=True \
    use_cuda=True bcrbc_use_comm=True bcrbc_generative_eval=True \
    test_nepisode=16 test_interval=50000 log_interval=5000 \
    runner_log_interval=5000 learner_log_interval=5000 \
    save_model_interval=100000 t_max="$T_MAX"
