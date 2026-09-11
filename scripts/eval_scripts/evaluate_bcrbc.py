"""Fixed-checkpoint delay sweep; sequential jobs with parallel SC2 episodes."""

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


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

    job = json.loads(job_path.read_text(encoding="utf-8"))
    config = json.loads(Path(job["config"]).read_text(encoding="utf-8"))
    config.update(
        seed=job["seed"], runner="delayed_parallel",
        batch_size_run=job["parallel"], test_nepisode=job["episodes"],
        render=False,
    )
    condition = job["condition"]
    config["env_args"].update(
        seed=job["seed"], map_name=job["test_map"],
        delay_type=condition["delay_type"], delay_mean=condition["mean"] or 0.0,
        delay_std=condition["std"] or 0.0, max_delay=condition["cap"],
    )
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
    runner.setup(scheme, groups, preprocess, mac)
    runner.log_train_stats_t = runner.t_env
    episode_path = job_path.parent / "episodes.jsonl"
    episode_path.write_text("", encoding="utf-8")
    (job_path.parent / "trajectories.jsonl").write_text("", encoding="utf-8")
    runner.diagnostics = EvaluationDiagnostics(
        episode_path, args.batch_size_run, episode_offset=job["episode_offset"],
    )
    with torch.no_grad():
        for _ in range(args.test_nepisode // args.batch_size_run):
            runner.run(test_mode=True)
    runner.close_env()
    for process in runner.ps:
        process.join()
    rows = [
        json.loads(line)
        for line in episode_path.read_text(encoding="utf-8").splitlines()
    ]
    result = {"condition": condition, "episodes": rows}
    temporary = job_path.parent / "result.tmp"
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(job_path.parent / "result.json")

if __name__ == "__main__":
    evaluate_job(Path(sys.argv[1]))
