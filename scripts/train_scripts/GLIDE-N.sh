#!/usr/bin/env bash
set -e

# 两个通用位置参数：seed、实验名。
SEED="$1"
NAME="$2"
shift 2

RUN_DIR="results/train/$NAME"
mkdir -p "$RUN_DIR"

git rev-parse HEAD > "$RUN_DIR/code_commit.txt"
git diff > "$RUN_DIR/code_diff.patch"
cp "$0" "$RUN_DIR/run.sh"

exec python -u src/main.py \
    --config=bcrbc_qmix \
    --env-config=delayed_sc2 with \
    name="$NAME" \
    seed="$SEED" \
    env_args.map_name=5m_vs_6m \
    runner=delayed_parallel \
    batch_size_run=16 \
    batch_size=16 \
    buffer_size=5000 \
    buffer_cpu_only=False \
    use_cuda=True \
    bcrbc_generation_horizon=1 \
    bcrbc_flow_steps=4 \
    test_nepisode=64 \
    test_interval=50000 \
    log_interval=10000 \
    runner_log_interval=10000 \
    learner_log_interval=10000 \
    save_model_interval=100000 \
    t_max=10000000 \
    "$@" \
    > "$RUN_DIR/train.log" 2>&1
