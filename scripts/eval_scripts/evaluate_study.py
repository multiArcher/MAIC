"""Evaluate batches while reusing compatible environment workers."""
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def evaluate_batch(job_path, runner=None):
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
    if runner is None:
        print(f"  Starting {job['parallel']} environment workers...", file=sys.__stdout__, flush=True)
        runner = DelayedParallelRunner(args, EvaluationLogger())
        runner.evaluation_seed = job["seed"]
        runner.evaluation_batch_index = 0
    else:
        runner.args = args
        runner.logger = EvaluationLogger()
    args.env_info = runner.get_env_info()
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
    print("  Model loaded; resetting environments and running episodes...",
          file=sys.__stdout__, flush=True)
    # Capture actual policy output, never re-sample the executed branch for scoring.
    forward = mac.forward
    def capture(*values, **keywords):
        result = forward(*values, **keywords)
        mac.decision_z = result["z"].detach()
        return result
    mac.forward = capture
    select = mac.select_actions
    last_progress = time.perf_counter()
    def timed_select(*values, **keywords):
        nonlocal last_progress
        if args.use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        actions = select(*values, **keywords)
        if args.use_cuda:
            torch.cuda.synchronize()
        mac.decision_ms = 1000 * (time.perf_counter() - start)
        now = time.perf_counter()
        if now - last_progress >= 10:
            print(f"  Episode step {runner.t}/{runner.episode_limit}",
                  file=sys.__stdout__, flush=True)
            last_progress = now
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
                  environment_seed=runner.evaluation_seed,
                  environment_batch_index=runner.evaluation_batch_index,
                  worker_pids=[process.pid for process in runner.ps],
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
    # Only environment workers persist; release batch-local models and caches.
    del mac.forward
    del mac.select_actions
    runner.mac = None
    runner.diagnostics = None
    runner.batch = None
    runner.evaluation_batch_index += 1
    path = job_path.parent / "result.tmp"
    path.write_text(json.dumps(result, indent=2))
    path.replace(job_path.parent / "result.json")
    return runner


def environment_key(job):
    config = json.loads(Path(job["config"]).read_text())
    env_args = dict(config["env_args"])
    env_args.pop("seed", None)
    env_args["max_delay"] = job["condition"]["cap"]
    return json.dumps(dict(
        env=config["env"], env_args=env_args, parallel=job["parallel"],
        common_reward=config["common_reward"],
        reward_scalarisation=config["reward_scalarisation"],
    ), sort_keys=True)


def close_runner(runner):
    runner.close_env()
    for process in runner.ps:
        process.join()


def main(job_path):
    request = json.loads(job_path.read_text())
    paths = [Path(path) for path in request["jobs"]] if "jobs" in request else [job_path]
    # Keep full batches together; a smaller final batch needs fewer workers.
    groups = {}
    for path in paths:
        job = json.loads(path.read_text())
        groups.setdefault(environment_key(job), []).append(path)
    total = len(paths)
    completed = 0
    started = time.perf_counter()
    for paths in groups.values():
        runner = None
        try:
            for path in paths:
                job = json.loads(path.read_text())
                print(
                    f"[{completed + 1}/{total}] {path.parent.parent.parent.name} | "
                    f"{path.parent.parent.name} | episodes "
                    f"{job['episode_offset'] + 1}-{job['episode_offset'] + job['parallel']} | "
                    f"{'new environments' if runner is None else 'reuse environments'}",
                    flush=True,
                )
                with (path.parent / "run.log").open("w") as log:
                    with redirect_stdout(log), redirect_stderr(log):
                        runner = evaluate_batch(path, runner)
                result = json.loads((path.parent / "result.json").read_text())
                rows = [json.loads(line) for line in
                        (path.parent / "episodes.jsonl").read_text().splitlines()]
                win_rate = sum(row["won"] for row in rows) / len(rows)
                completed += 1
                elapsed = time.perf_counter() - started
                remaining = elapsed / completed * (total - completed)
                print(
                    f"  Done {completed}/{total} ({completed / total:.1%}) | "
                    f"win={win_rate:.1%} | rollout={result['seconds']:.1f}s | "
                    f"elapsed={elapsed / 60:.1f}min | ETA~{remaining / 60:.1f}min",
                    flush=True,
                )
        finally:
            if runner is not None:
                close_runner(runner)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
