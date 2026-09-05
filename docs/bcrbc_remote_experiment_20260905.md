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

### 12:12 hourly observation

Batch-8 process remains running after 46 minutes; no exit file and no
nonfinite persisted metrics. Last loss sample: t_env=145772 (7.3% of 2M).
Latest raw losses: flow 0.07041, reconstruction 0.01252, message 0.01085,
TD 1.96610, total 2.05989. Training return at 145622 is 3.3063,
epsilon 0.72338. Raw delayed evaluation: at 50053 steps, win rate 1/16
and return 10.459; at 100064, win rate 0/16 and return 4.244, mean episode
length 103.375. Learning is not yet consistently successful. Keep running
with unchanged parameters; the current quiet log period is consistent with
the next 50K-step evaluation interval, while Python and SC2 remain active.

### 13:14 hourly observation

Process remains running after 1h48m; last persisted loss sample is at
t_env=356364 (17.8%). No nonfinite metrics or exit file. Latest losses:
flow 0.08894, reconstruction 0.01564, message 0.01486, TD 2.32340,
total 2.44284. Training return increased to 8.7606 at 356124, with
training win rate 0.03247 and epsilon 0.32359. Delayed evaluation remains
weak: all five evaluations from 150086 through 350150 steps won 0/16.
Latest evaluation return is 4.4199 and mean length 91.4375. Training
performance is improving, but this has not translated into delayed-test
wins. Continue unchanged; do not infer completion benefit from auxiliary
loss reduction alone.

### 15:16 hourly observation

Batch-8 process is still running after 3h51m. Latest persisted step is
t_env=908453 (45.4%); no exit file or nonfinite persisted metrics. Latest
losses: flow 0.15325, reconstruction 0.01687, message 0.01607, TD 1.04411,
total 1.23031. Latest training window has return 18.5536 and win rate
0.83766, with epsilon 0.05. Raw delayed evaluation at 700297, 750297,
800322, 850340 and 900346 steps won 1/16, 2/16, 5/16, 0/16 and 0/16,
respectively. Corresponding returns are 6.7305, 10.9434, 12.8047, 8.6230
and 7.9063. Latest mean evaluation length is 25.5625. Delayed performance
briefly improved at 800K but remains unstable and far below training
performance; no consistent compensation benefit is established. Continue
unchanged and retain the planned fixed-checkpoint completion on/off test.

### 16:18 hourly observation

Training remains RUNNING after 4h52m, t_env=1179399 (59.0%), with no exit
file or nonfinite persisted metrics. Latest raw losses: flow 0.13136,
reconstruction 0.01194, message 0.02030, TD 0.42114, total 0.58474.
Latest training win rate is 0.93048 and return 19.41594, epsilon 0.05.
Delayed tests at 950354, 1000357, 1050389, 1100390 and 1150400 steps won
0/16, 1/16, 1/16, 0/16 and 2/16. Latest test return is 10.28516 and
mean length 26.6875. Persistent train/test gap; no new failure. Keep the
original run unchanged. User-approved follow-up now lives in
docs/bcrbc_delay_grid_protocol.md: implement online diagnostics and dense
parallel evaluation only after this training finishes, then pause monitoring
after the follow-up finishes rather than immediately after training.

### 17:19 hourly observation

Training remains RUNNING after 5h54m, t_env=1440049 (72.0%), with no exit
file or nonfinite persisted metrics. Latest losses: flow 0.13839,
reconstruction 0.00943, message 0.01887, TD 0.19403, total 0.36071.
Latest training win rate is 0.93782, return 19.44301, epsilon 0.05.
Delayed evaluations at 1200408, 1250431, 1300466, 1350468 and 1400472
won 5/16, 1/16, 4/16, 0/16 and 6/16, with returns 13.35352, 7.72461,
12.64063, 3.25391 and 12.91406. Latest evaluation length is 26.875.
Delayed performance has some stronger points but remains highly variable;
one 16-episode result does not establish stable recovery. Keep running
unchanged. Follow-up diagnostics and grid implementation remain deferred
until original training completion as requested.

### 18:21 hourly observation

Training remains RUNNING after 6h55m, t_env=1720958 (86.0%), with no exit
file or nonfinite persisted metrics. Latest losses: flow 0.12661,
reconstruction 0.01000, message 0.02062, TD 0.17909, total 0.33633.
Latest training window: win rate 0.84783, return 18.64283, epsilon 0.05.
Delayed evaluations at 1500498, 1550502, 1600511, 1650516 and 1700518
won 1/16, 3/16, 2/16, 4/16 and 2/16, respectively. Latest test return
is 9.82617 and mean length 33.0625. No material change in the persistent
train/test gap. Original run continues unchanged; defer implementation
until completion, then follow docs/bcrbc_delay_grid_protocol.md.

### 19:22 hourly observation

Training remains RUNNING after 7h56m, t_env=1986761 (99.3%), no exit file
or nonfinite metrics. Latest flow 0.13455, reconstruction 0.00898, message
0.01857, TD 0.07386, total 0.23596. Training win rate 0.87958, return
18.84457. Delayed evaluations at 1750530, 1800551, 1850570, 1900581,
1950596 won 0/16, 4/16, 12/16, 4/16, 4/16; returns 6.33203, 11.86523,
17.79688, 11.80469, 11.12305. The isolated 75% result at 1.85M is not
sustained and must not motivate checkpoint selection. Keep final-checkpoint
protocol. Training is near completion but not yet confirmed complete;
next heartbeat should check exit code and start the approved implementation
if complete rather than only reporting the training summary.

Check hourly after startup, focusing on progress, errors, loss trends,
test win rate, return, episode length and checkpoint availability. Use the
raw Sacred test values rather than the console's recent-value average.
Do not alter hyperparameters mid-run. A single seed describes this run only;
it cannot establish an advantage over other methods. After completion,
evaluate a fixed saved checkpoint with and without generative completion
under the same configured delay to examine its contribution, and report
the learning curve and limitations. Do not start another long-training run
automatically.
