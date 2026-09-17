"""One frozen-model/condition batch. Invoked by delay_study.py."""
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def main(job_path):
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from components.episode_buffer import EpisodeBatch
    from components.transforms import OneHot
    from controllers.bcrbc_mac import BCRBCMAC
    from runners.delayed_parallel_runner import DelayedParallelRunner
    from run import parse_buffer_scheme
    from evaluate_bcrbc import EvaluationLogger
    from study_diagnostics import StudyDiagnostics

    job = json.loads(job_path.read_text())
    config = json.loads(Path(job["config"]).read_text())
    if "bcrbc_time_block_every" not in config:
        config["bcrbc_time_block_every"] = job["model"]["legacy_time_block_every"]
    # Preserve all architecture and completion settings from the training run.
    config.update(batch_size_run=job["parallel"], test_nepisode=job["parallel"],
                  runner="delayed_parallel", render=False, evaluation_epsilon=0.0)
    config["env_args"].update(seed=job["seed"], max_delay=job["condition"]["cap"])
    args = SimpleNamespace(**config)
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    torch.set_num_threads(1)
    (job_path.parent / "effective_config.json").write_text(json.dumps(config, indent=2))
    runner = DelayedParallelRunner(args, EvaluationLogger())
    for index, connection in enumerate(runner.parent_conns):
        connection.send(("set_evaluation_delay", dict(condition=job["condition"], seed=job["seed"] + index)))
    for connection in runner.parent_conns:
        connection.recv()
    info = runner.get_env_info()
    args.n_agents, args.n_actions, args.state_shape = info["n_agents"], info["n_actions"], info["state_shape"]
    scheme = parse_buffer_scheme(info, args.common_reward)
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(args.n_actions)])}
    template = EpisodeBatch(scheme, groups, 1, 1, preprocess=preprocess)
    mac = BCRBCMAC(template.scheme, groups, args).to(args.device)
    mac.load_models(job["checkpoint"])
    # Capture actual policy output, never re-sample the executed branch for scoring.
    forward = mac.forward
    def capture(*values, **keywords):
        result = forward(*values, **keywords)
        mac.decision_z = result["z"].detach()
        return result
    mac.forward = capture
    select = mac.select_actions
    def timed_select(*values, **keywords):
        if args.use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        actions = select(*values, **keywords)
        if args.use_cuda:
            torch.cuda.synchronize()
        mac.decision_ms = 1000 * (time.perf_counter() - start)
        return actions
    mac.select_actions = timed_select
    runner.setup(scheme, groups, preprocess, mac)
    runner.diagnostics = StudyDiagnostics(job_path.parent, job["parallel"],
                                         job["episode_offset"], job["mask_intervention"], job["feature_groups"])
    if args.use_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        runner.run(test_mode=True)
    seconds = time.perf_counter() - start
    result = dict(seconds=seconds, episodes=job["parallel"],
                  device_name=torch.cuda.get_device_name() if args.use_cuda else "CPU",
                  torch_version=str(torch.__version__),
                  diagnostic_inclusive_peak_bytes=torch.cuda.max_memory_allocated() if args.use_cuda else 0,
                  logger_metrics=runner.logger.metrics)
    for connection in runner.parent_conns:
        connection.send(("get_final_info", None))
    final_infos = [connection.recv() for connection in runner.parent_conns]
    episode_path = job_path.parent / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text().splitlines()]
    for row, info in zip(rows, final_infos):
        row.update(dead_allies=info.get("dead_allies"), dead_enemies=info.get("dead_enemies"))
    episode_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    runner.close_env()
    for process in runner.ps:
        process.join()
    path = job_path.parent / "result.tmp"
    path.write_text(json.dumps(result, indent=2))
    path.replace(job_path.parent / "result.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
