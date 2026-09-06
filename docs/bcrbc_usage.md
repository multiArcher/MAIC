# BCRBC usage

Run from the project root with the experiment's Conda environment activated.
Inherit `SC2PATH` from the host environment.

## Training and evaluation

```bash
bash scripts/train_scripts/bcrbc_qmix.sh
```

The launcher uses `delayed_sc2` and `delayed_parallel`. Training observations and
communication have zero delay. Evaluation uses configured observation and
communication delays, with generative completion enabled.

Algorithm defaults are in `src/config/algs/bcrbc_qmix.yaml`; observation-delay
settings are in `src/config/envs/delayed_sc2.yaml`. Communication-delay settings
are in the algorithm config. Gaussian and uniform sampling are supported.

For checkpoint evaluation, copy `bcrbc_key_conditions.py` or
`bcrbc_delay_grid.py` beside the original as a `*.local.py` file. Set the training
run, checkpoint step, output directory and experiment parameters at the top.
Replace placeholder paths in the tracked examples before running.

```bash
python scripts/eval_scripts/bcrbc_key_conditions.local.py
python scripts/eval_scripts/plot_bcrbc_delay_grid.py results/evaluate/EXPERIMENT_NAME/key_conditions
```

`None` in evaluation conditions means true zero delay; a Gaussian mean of zero
can still produce positive delay. Evaluation outputs belong under
`results/evaluate/<experiment>/`, with stages in separate subdirectories.

## Local and remote changes

Change code locally, commit and push, then pull on the execution host. Keep
tracked configs and scripts at their default values. Host-specific parameters
belong in command-line overrides or ignored `*.local.sh`, `*.local.py` and
`*.local.yaml` copies; change only parameters in these copies. Save actual run
configuration and code commit with experiment results.

Agent-only plans, diagnostics and archived tests belong under the ignored
`.agents/` directory. They are not part of the algorithm distribution.
