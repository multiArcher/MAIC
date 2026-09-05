"""Fixed-checkpoint delay sweep; sequential jobs with parallel SC2 episodes."""

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class EvaluationLogger:
    def __init__(self):
        self.metrics = {}

    def log_stat(self, name, value, step):
        self.metrics[name] = float(value)
        print(f"{name}: {value}", flush=True)


def evaluate_job(job_path):
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from components.episode_buffer import EpisodeBatch
    from components.transforms import OneHot
    from controllers.bcrbc_mac import BCRBCMAC
    from modules.bcrbc.evaluation_diagnostics import EvaluationDiagnostics
    from run import parse_buffer_scheme
    from runners.delayed_parallel_runner import DelayedParallelRunner

    job = json.loads(job_path.read_text())
    config = json.loads(Path(job["config"]).read_text())
    config.update(
        seed=job["seed"], runner="delayed_parallel",
        batch_size_run=job["parallel"], test_nepisode=job["episodes"],
        bcrbc_generative_eval=job["completion"], render=False,
    )
    config["env_args"]["seed"] = job["seed"]
    config["env_args"]["delay_mean"] = job["obs_mean"] or 0.0
    config["env_args"]["delay_std"] = 1.0
    config["env_args"]["max_delay"] = 0 if job["obs_mean"] is None else 2
    config["comm_gaussian_delay_mean"] = job["comm_mean"] or 0.0
    config["comm_gaussian_delay_std"] = 1.0
    args = SimpleNamespace(**config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    logger = EvaluationLogger()
    runner = DelayedParallelRunner(args, logger)
    info = runner.get_env_info()
    args.n_agents = info["n_agents"]
    args.n_actions = info["n_actions"]
    args.state_shape = info["state_shape"]
    scheme = parse_buffer_scheme(info, args.common_reward)
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(args.n_actions)])}
    template = EpisodeBatch(scheme, groups, 1, 1, preprocess=preprocess)
    mac = BCRBCMAC(template.scheme, groups, args).to(args.device)
    mac.load_models(job["checkpoint"])
    # Communication draws must not depend on the number of diffusion draws.
    mac.comm_delay.delay_model.generator = torch.Generator(device=args.device)
    mac.comm_delay.delay_model.generator.manual_seed(args.seed + 10000)
    if job["comm_mean"] is None:
        mac.comm_delay.delay_model.max_delay = 0
    runner.setup(scheme, groups, preprocess, mac)
    runner.log_train_stats_t = runner.t_env
    episode_path = job_path.parent / "episodes.jsonl"
    episode_path.write_text("")
    runner.diagnostics = EvaluationDiagnostics(episode_path, args.batch_size_run)
    with torch.no_grad():
        for _ in range(args.test_nepisode // args.batch_size_run):
            runner.run(test_mode=True)
    runner.close_env()
    for process in runner.ps:
        process.join()
    rows = [json.loads(line) for line in episode_path.read_text().splitlines()]
    totals = {}
    for row in rows:
        for key, value in row.items():
            if key.endswith(("_sum", "_count")) or key == "agent_steps":
                totals[key] = totals.get(key, 0) + value
    result = {**job, **logger.metrics, **totals}
    for key, value in totals.items():
        if key.endswith("_sum"):
            group = next(g for g in ("missing", "never_arrived", "stale_arrived")
                         if key.startswith(g + "_"))
            count = totals[group + "_count"]
            result[key[:-4]] = value / count if count else None
    (job_path.parent / "result.json").write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--job", type=Path)
    options = parser.parse_args()
    if options.job:
        evaluate_job(options.job)
        return
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    means = [tick / 5 for tick in range(-5, 6)]
    conditions = list(itertools.product(means, means))
    conditions += [(mu, None) for mu in means]
    conditions += [(None, mu) for mu in means] + [(None, None)]
    if options.pilot:
        conditions = [(None, None), (-0.4, -0.4)]
    jobs = []
    for index, ((obs_mean, comm_mean), completion, seed) in enumerate(
        itertools.product(conditions, (False, True), options.seeds)
    ):
        job = {
            "config": str(options.config.resolve()),
            "checkpoint": str(options.checkpoint.resolve()),
            "obs_mean": obs_mean, "comm_mean": comm_mean,
            "completion": completion, "seed": seed,
            "parallel": options.parallel, "episodes": options.episodes,
        }
        directory = output / f"job_{index:04d}"
        directory.mkdir(exist_ok=True)
        path = directory / "job.json"
        path.write_text(json.dumps(job, indent=2))
        jobs.append(path)
    (output / "manifest.json").write_text(json.dumps(
        [json.loads(path.read_text()) for path in jobs], indent=2
    ))
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for index, path in enumerate(jobs):
        result_path = path.parent / "result.json"
        if not result_path.exists():
            print(f"Running {index + 1}/{len(jobs)}: {path.parent.name}", flush=True)
            with (path.parent / "run.log").open("w") as log:
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--job", str(path)],
                    cwd=ROOT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, check=True,
                )
        results = [json.loads(p.read_text()) for p in output.glob("job_*/result.json")]
        fields = sorted(set().union(*(row.keys() for row in results)))
        with (output / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    print("Evaluation grid completed", flush=True)


if __name__ == "__main__":
    main()
