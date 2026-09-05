# BCRBC 8m long-training experiment, 2026-09-05

- Host: `epic-cluster-compute-rtx4070-01-liuhongbo`
- Project: `/home/liuhongbo/workspace/epymarl_based`
- Base commit: `62f5d80e3781f9fbdf59cc1a0229a3c47edfa5da`
- Environment: `/home/liuhongbo/.local/miniconda3/envs/marl_stable`
- GPU: RTX 4070 Ti SUPER, 16 GB; PyTorch 2.5.1.
- SC2: inherit `/etc/profile.d/starcraft2_register.sh` through `bash -l`.
  Existing `SC2PATH=/opt/StarCraftII_46` selects SC2 4.6.

## Experiment

Single seed 1, map 8m, 2,000,000 training steps, batch 16, CPU replay
buffer 5,000 episodes. Keep the existing 128-dimensional, depth-4 model,
two 64-dimensional z tokens, 48-step context and 8 flow solver steps.
Communication and generative evaluation are enabled. Training observations
and communication have zero delay; evaluation uses the configured delays.
Evaluate 16 episodes every 50,000 training steps; log every 5,000 steps;
save checkpoints every 100,000 steps. Retro loss remains the existing disabled
ablation. No change to the algorithm or optimizer for this experiment.

Loss logging fix: BCRBCLearner now uses its supplied experiment logger, so
TD, reconstruction, flow, message reconstruction and total losses reach
Sacred and TensorBoard. Updated smoke-test logger substitutes accordingly.

## Execution

The 600-step pilot under SC2 4.6 completed at 647 environment steps in about one minute.
Its logged losses are finite and nonzero for the enabled objectives. Initial
greedy evaluation reaches the 120-step horizon with zero wins and zero return.
This is an initial policy observation, not a learning-effect conclusion.
An earlier setup pilot used SC2 4.10; its results are separate and must not be
combined with the SC2 4.6 learning curve.

Formal launch, remote local time 2026-09-05 11:10:

```bash
cd /home/liuhongbo/workspace/epymarl_based
tmux new-session -d -s bcrbc_8m_train_20260905 \
    bash -l scripts/run_bcrbc_remote.sh train
```

- Session: `bcrbc_8m_train_20260905`
- Log: `results/remote_bcrbc_8m_20260905_seed1_sc246/train.log`
- Completion exit code: same directory, `train.exit`
- Sacred run name prefix: `bcrbc_8m_train_20260905__8m__`
- Metrics: `python scripts/check_bcrbc_remote.py`
- Hourly monitor: Codex heartbeat `bcrbc-8m`, attached to this task.
- Formal run: `bcrbc_8m_train_20260905__8m__2026-09-05_11-10-06-td_=1.0`.

## Interpretation and follow-up

### 11:24 update: first long run failed, restarted with batch 8

The initial batch-16 run failed at 11:17 with CUDA OOM (14.38 GiB allocated
by PyTorch; only 52 MiB free). Last persisted losses were at t_env=548;
the only saved checkpoint was at step 32. No long-training effect can be
inferred from that failed run. Keep its logs and Sacred directory intact.

Restarted from seed 1 with batch 8, retaining the model and all other
hyperparameters. Current log directory is
`results/remote_bcrbc_8m_20260905_seed1_sc246_batch8/`.
Current tmux session remains `bcrbc_8m_train_20260905`.
The launcher now uses tee so attaching tmux shows live output too.
The metrics checker automatically selects the newest Sacred run.

Check hourly after startup, focusing on progress, errors, loss trends,
test win rate, return, episode length and checkpoint availability. Use the
raw Sacred test values rather than the console's recent-value average.
Do not alter hyperparameters mid-run. A single seed describes this run only;
it cannot establish an advantage over other methods. After completion,
evaluate a fixed saved checkpoint with and without generative completion
under the same configured delay to examine its contribution, and report
the learning curve and limitations. Do not start another long-training run
automatically.
