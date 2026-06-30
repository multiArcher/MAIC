# BC-RBC Usage Notes

## No-delay training

Use `bcrbc_qmix` with the standard `sc2` env first:

```bash
conda run -n marl_stable --no-capture-output python src/main.py --config=bcrbc_qmix --env-config=sc2 with name=bcrbc_8m env_args.map_name=8m obs_last_action=True batch_size_run=4 batch_size=8 t_max=20000 test_interval=5000 test_nepisode=16 buffer_cpu_only=True device=cuda
```

## Delayed-observation evaluation or training

Use `delayed_sc2` to delay local observations while keeping global state and available actions fresh:

```bash
conda run -n marl_stable --no-capture-output python src/main.py --config=bcrbc_qmix --env-config=delayed_sc2 with name=bcrbc_delay1 env_args.map_name=3m env_args.delay=1 env_args.max_delay=1 obs_last_action=True batch_size_run=1 batch_size=1 t_max=120 test_interval=120 test_nepisode=1 buffer_cpu_only=True device=cpu
```

Delay config fields live under `env_args`:

- `delay_type`: `fixed`, `uniform`, or `gaussian`
- `delay`: fixed delay in timesteps
- `delay_mean`: Gaussian delay mean
- `delay_std`: Gaussian delay standard deviation
- `max_delay`: hard delay cap
- `delay_per_agent`: sample delay independently per agent when true

## Smoke test

```bash
conda run -n marl_stable --no-capture-output python scripts/smoke_tests/bcrbc_fake_train.py
```
