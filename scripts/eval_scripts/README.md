# BCRBC observation-delay evaluation

The only experiment-parameter entry is the block at the top of
`bcrbc_delay_grid.py`. Copy it to `bcrbc_delay_grid.local.py` and edit the training
run, checkpoint step, test map and grid there. Local parameter copies are ignored
by Git. `evaluate_bcrbc.py` executes one batch; it has no second parameter table.
Slurm scripts only select hardware and activate the environment, then run:

```bash
python -u scripts/eval_scripts/bcrbc_delay_grid.local.py
```

Defaults: 16 parallel environments, 64 episodes per condition, one seed stream.
The Gaussian grid uses means `[-2, -1, 0, 1, 2]` and standard deviations
`[0, 0.5, 1, 1.5, 2]`. Delay is `min(8, ceil(max(0, Normal(mean, std))))`.
Thus `N(0, 0)` is exactly zero delay; negative means with zero variance also give
zero delay. A separate condition samples discrete uniform delays from 0 through 8.
A large-variance clipped Gaussian is **not** uniform. The cap also determines the
controller's mutable correction tail; both use `env_args.max_delay`.

## What is compared

All paths use the same frozen model and the actual executed action history:

- **Generated**: delayed local observations, sequential flow completion; this path
  alone chooses actions that affect the environment.
- **Mask**: exactly the same received history, learned MASK tokens at missing
  positions, no flow generation.
- **Reference**: complete, undelayed **local** observation history. This is a shadow
  policy, not a centralized controller. It never feeds observations or cache into
  either delayed path.

The two shadow paths have independent caches and never sample actions or noise.
Main metrics include only decisions with a missing current observation and more
than one legal action. Dead/no-op-only agents are excluded.

| TensorBoard metric | Meaning |
| --- | --- |
| `test_action/generated_agreement` | Fraction of generated greedy actions matching reference; higher is better |
| `test_action/mask_agreement` | Same fraction for Mask-only history |
| `test_action/generated_kl` | KL(reference Q-softmax || generated Q-softmax), temperature 1; lower is closer |
| `test_action/mask_kl` | Same KL for Mask-only history |
| `test_action/decision_count` | Eligible agent decisions in the denominator |

Illegal actions are excluded before softmax and argmax. Q-softmax KL is a
comparison of Q-induced distributions, **not** the epsilon-greedy behavior-policy
KL. If there are no eligible decisions, agreement/KL are undefined (blank in the
summary and not logged), not zero. No-delay conditions still report win rate.

Training retains TD, tokenizer reconstruction, random-signal flow, final generated
latent, generated-observation reconstruction and total losses. These are optimized
objectives, not the removed vector-distance evaluation panels. Retro loss remains
an explicitly disabled ablation and is logged only when enabled.

Action agreement measures recovery of the model's own reference decisions. It
does not prove those decisions are optimal or establish the causal contribution
to win rate. These are matched-trajectory diagnostics, not independent Mask-only
control rollouts.

## Results and incremental evaluation

```text
results/evaluate/<train_run>/checkpoint_<step>/test_map_<map>/
  action_consistency_v1_<fingerprint>/
    manifest.json
    mu_<mean>_std_<std>_cap_<cap>/batch_000/
      job.json, run.log, episodes.jsonl, trajectories.jsonl, result.json
    uniform_0_<cap>/batch_000/...
    summary.csv, summary.json, figures/
```

Completed `result.json` batches are reused. Adding grid points evaluates new
points; increasing episode count adds fixed-size batches with stable seeds.
Episode counts round up to a multiple of the parallel count. An interrupted batch
is rerun from scratch. Summaries are rebuilt from complete per-episode records,
weighted by decision counts rather than averaging batch percentages. Previously
evaluated points remain in the summary when extending or narrowing the grid.

The fingerprint separates checkpoints, saved training configurations, model/eval
source, maps, parallel count and seed stream. Changing model/evaluation code starts
a new protocol directory; changing only grid points or episode count extends the
same directory. The manifest stores both training and test maps. This directory
layout does not make models with different agent/observation/action dimensions
interchangeable. Old checkpoints with communication token slots are not supported.

Per-episode records also store the **sampled packet delay** histogram (not the
age of the freshest delivered packet). Summary statistics describe the actual
clipped, discretized distribution seen in the finite evaluation episodes.

Only the first four episodes per condition save action trajectories: reference,
generated, Mask and actual actions, with missing/eligible flags. Plots show action
indices on the same trajectory; black in the missing-observation panel means
the current observation is unavailable. Grid plots keep common scales for the
two agreement panels and for the two KL panels. Uniform results are listed in
the summary, outside the Gaussian grid.

Regenerate plots without rerunning environments:

```bash
python scripts/eval_scripts/plot_bcrbc_delay_grid.py <protocol-result-directory>
```
