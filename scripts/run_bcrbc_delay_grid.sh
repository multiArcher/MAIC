#!/usr/bin/env bash
# Start with bash -l so the registered SC2PATH is inherited.
set -uo pipefail
cd /home/liuhongbo/workspace/epymarl_based
python=/home/liuhongbo/.local/miniconda3/envs/marl_stable/bin/python
run=bcrbc_8m_train_20260905__8m__2026-09-05_11-25-42-td_=1.0
output=results/bcrbc_delay_grid_20260905_seed1_step1900530
mkdir -p "$output"
"$python" -u scripts/evaluate_bcrbc_delay_grid.py \
    --config "results/sacred/$run/1/config.json" \
    --checkpoint "results/models/$run/1900530" \
    --output "$output" 2>&1 | tee "$output/sweep.log"
status=$?
printf '%s\n' "$status" > "$output/sweep.exit"
if [ "$status" -eq 0 ]; then
    "$python" scripts/plot_bcrbc_delay_grid.py "$output"
fi
exit "$status"
