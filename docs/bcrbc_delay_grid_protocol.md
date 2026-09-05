# BCRBC delay-grid follow-up: user-approved protocol

## Timing and authority

Implement after the current seed-1 2M-step remote training finishes. Do not
change its running code or hyperparameters. Autonomously implement and run
a fixed-checkpoint parallel evaluation, not another long training run.
Preserve unrelated dirty files. Use minimal, readable research-style PEP8
code without defensive algorithm scaffolding or unnecessary preflight tests.

Host: epic-cluster-compute-rtx4070-01-liuhongbo.
Project: /home/liuhongbo/workspace/epymarl_based.
Conda: /home/liuhongbo/.local/miniconda3/envs/marl_stable.
Inherit SC2PATH from /etc/profile.d/starcraft2_register.sh via bash -l.

## Fixed evaluation

- Lock the final saved checkpoint and actual step once; retain source config.
- Existing delayed_parallel runner, initially 4 environments. Inspect the
  user's recent parallel implementation before editing. Reduce evaluation
  concurrency only on actual resource failures.
- Gaussian observation/communication means independently take integer ticks
  -5..5 divided by 5: -1,-0.8,...,1. Both standard deviations remain 1.
- Preserve ceil/clamp sampling, observation cap 2 and communication cap 16.
- 121 mean pairs, completion on/off, communication on in both modes.
- Add true-zero controls: 11 observation-only, 11 communication-only and one
  both-zero condition. Mean zero is not true zero delay. Total 144 conditions.
- 32 episodes for each seed 101,102,103 in each mode: 27648 episodes total.
  These are evaluation seeds, not independent training seeds.
- Greedy evaluation. Isolate generation and delay random streams as needed;
  equal seed schedules do not imply identical trajectories after actions differ.
- Sequential grid jobs, parallel environments inside each job. First run a
  small actual parallel pilot, then the fixed full protocol. No model tuning.

## Online metrics

Capture the actual current latent used for action selection at environment
step k, before future arrival corrections. Fresh o_k is analysis-only: never
feed it to policy inputs, messages, generation state, KV cache or actions.
Score only currently missing/stale slots and also save counts/missing fraction:

1. Generated latent MSE vs E(o_k), averaging both token and feature axes.
2. Generated observation MSE: D(generated z_k) vs o_k.
3. Tokenizer MSE: D(E(o_k)) vs o_k on the same slots (reference, not lower bound).
4. Stale observation MSE: delivered observation vs o_k on the same trajectory.
5. Completion gain: stale observation MSE minus generated observation MSE.

Accumulate sums/counts, exclude padding and terminal bootstrap-only slots.
No missing samples means unavailable, not zero. Separate never-arrived and
stale-arrived cases. Do not resample predictions for diagnostics or perturb
rollout random streams. Off-mode errors must not be labeled generated errors.

Existing BCRBCMAC.z_diagnostics is unused in src and not directly suitable:
it regenerates retrospectively, uses oracle messages in parts, and its latent
error reduction omits token-axis averaging. Implement online measurements.

## Files and outputs

Suggested scripts: scripts/evaluate_bcrbc_delay_grid.py and
scripts/plot_bcrbc_delay_grid.py. Reuse existing loading/evaluation interfaces;
add only needed runner/MAC/wrapper instrumentation. Do not refactor algorithms.

Remote root: results/bcrbc_delay_grid_20260905_seed1_step<actual-step>/.
Store protocol/config manifest, per-job logs/Sacred paths, per-seed CSV,
aggregate CSV and per-episode metrics. Save completed jobs immediately and
resume without duplicating them. Keep raw results distinct from summaries.

Produce common-scale on/off win-rate heatmaps, difference heatmap, latent and
observation error heatmaps, completion-gain heatmap and canonical CSV tables.
Label axes Gaussian mu, not actual average delay. No interpolated surfaces.
Report uncertainty using episodes and acknowledge single training seed.

Analyze no-delay vs delay, observation vs communication and error vs victory.
Copy compact tables, figures and report locally; retain full remote logs.
Then pause bcrbc-8m monitor. No automatic new long training or algorithm tuning.

## Stage

Original training completed at 19:24:04 on 2026-09-05, exit 0, runtime
7h58m22s. Last numerical saved checkpoint is 1900530 (no 2M checkpoint was
saved by the interval-based launcher). Use that fixed checkpoint, not best_model.

Online instrumentation and sweep scripts implemented in the local checkout
and synced to the remote after training ended. User's delayed_parallel runner
is a subclass of ParallelRunner; optional diagnostics hook lives in the shared
loop, oracle reads use a separate worker request. MAC stores decision_z from
the actual forward. DelayModel has an optional generator used by the evaluation
script to isolate communication randomness from flow noise. Existing training
defaults are unchanged. Scripts directly load the original Sacred config and
MAC checkpoint; they do not instantiate a learner or replay training buffer.

Pilot: scripts/evaluate_bcrbc_delay_grid.py with --pilot --episodes 4 --seeds
101, parallel=4. Four jobs completed: true-zero and means (-0.4,-0.4), each
completion off/on. Results in the output root with suffix _pilot. True-zero
off won 4/4, missing_count=0 and missing errors are null. Light-delay on won
4/4, missing latent MSE 0.85714, decoded MSE 0.06033 vs stale MSE 0.03776;
within-trajectory completion gain -0.02257. Small samples, no effectiveness
conclusion. Never-arrived and stale-arrived errors differ strongly.

Full sweep launched in tmux bcrbc_delay_grid_20260905 via
bash -l scripts/run_bcrbc_delay_grid.sh. Output root is exactly
results/bcrbc_delay_grid_20260905_seed1_step1900530/. Read sweep.log and
sweep.exit (only appears on exit), job_*/result.json and results.csv. There
are 864 jobs / 27648 episodes. Do not relaunch from scratch; script skips
completed result.json jobs if restarted with identical parameters.

Remote py_compile passed for both Python scripts, diagnostics, MAC, shared
runner and delay model. Plotting initially failed because pandas was absent;
installed pandas 3.0.5 in remote marl_stable (numpy unchanged), then pilot
aggregation and all heatmaps completed successfully. Pilot win-rate figure
and raw CSV copied locally under results/..._pilot for inspection. Full sweep
confirmed through job 4/864 at about 20:33, still in progress. Latest runner
also records per-episode wins in JSONL; initial pilot predates that addition.
Once sweep completes, launcher invokes plotting script; still inspect its
output because sweep.exit reports evaluation status, not plotting status.
Continue via existing bcrbc-8m heartbeat: monitor sweep rather than retraining
or reimplementing. Finish analysis, copy compact outputs locally, pause monitor.

### 21:24 observation

Full sweep remains active after 55 minutes. Completed 84/864 jobs (2688
episodes); job_0084 is running. No sweep.exit file; tmux and evaluation
processes alive. No restart or parameter change. Grid order means these
early results cover mainly low observation-delay means, not the full range;
defer overall efficacy conclusions until the matrix and controls complete.

### 22:27 observation

Sweep still active after 1h58m. Completed 189/864 jobs (6048 episodes,
21.9%); job_0189 running. No sweep.exit file and no failure in scheduler
tail. No restart or configuration changes. Continue hourly monitoring.

### 23:29 observation

Sweep active after 3h00m; completed 279/864 jobs (8928 episodes, 32.3%).
job_0279 running, tmux present, no sweep.exit or scheduler error. Continued
without configuration changes. Full-grid conclusions remain deferred.

### 2026-09-06 00:30 observation

Sweep active after 4h01m; completed 375/864 jobs (12000 episodes, 43.4%).
job_0375 running, tmux present, no sweep.exit or scheduler error. No changes
to experiment settings. Continue the existing hourly monitor.
