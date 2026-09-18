"""Observation-delay evaluation. Edit this single parameter block in a .local.py copy."""

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TRAIN_RUN = "TRAIN_RUN_NAME"
CHECKPOINT_STEP = "CHECKPOINT_STEP"
TEST_MAP = "5m_vs_6m"
MEANS = [-2, -1, 0, 1, 2]
STANDARD_DEVIATIONS = [0, 0.5, 1, 1.5, 2]
MAX_DELAY = 8
EPISODES = 64
PARALLEL = 16
SEED = 101
PROTOCOL = "action_consistency_v1"


def conditions():
    grid = [
        dict(delay_type="gaussian", mean=mean, std=std, cap=MAX_DELAY)
        for mean, std in itertools.product(MEANS, STANDARD_DEVIATIONS)
    ]
    grid.append(dict(delay_type="uniform", mean=None, std=None, cap=MAX_DELAY))
    return grid


def condition_name(condition):
    if condition["delay_type"] == "uniform":
        return f"uniform_0_{condition['cap']}"
    return f"mu_{condition['mean']:g}_std_{condition['std']:g}_cap_{condition['cap']}"


def summarize_episodes(rows):
    count = sum(row["decision_count"] for row in rows)
    result = {
        "episodes": len(rows),
        "win_rate": sum(row["won"] for row in rows) / len(rows),
        "return_mean": sum(row["return"] for row in rows) / len(rows),
        "length_mean": sum(row["length"] for row in rows) / len(rows),
        "decision_count": count,
    }
    for path in ("generated", "mask"):
        for metric in ("agreement", "kl"):
            name = f"{path}_{metric}"
            total = sum(row[name + "_sum"] for row in rows)
            result[name] = total / count if count else None
    histogram = {}
    for row in rows:
        for delay, frequency in row["sampled_delay_histogram"].items():
            histogram[delay] = histogram.get(delay, 0) + frequency
    packets = sum(histogram.values())
    mean = sum(int(delay) * frequency for delay, frequency in histogram.items()) / packets
    variance = sum(
        (int(delay) - mean) ** 2 * frequency
        for delay, frequency in histogram.items()
    ) / packets
    result.update(
        sampled_delay_mean=mean, sampled_delay_std=variance ** 0.5,
        sampled_delay_histogram=histogram,
    )
    return result


def refresh_summary(output):
    """Rebuild from complete batches only; never average batch percentages."""
    summary = []
    for directory in sorted(output.iterdir()):
        batches = sorted(directory.glob("batch_*/result.json"))
        if not batches:
            continue
        rows = []
        for path in batches:
            batch_result = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(batch_result["episodes"])
        summary.append({
            **batch_result["condition"],
            "condition": directory.name,
            **summarize_episodes(rows),
        })
    if summary:
        path = output / "summary.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8",
        )
    return summary


def source_digest():
    files = list((ROOT / "src/modules/bcrbc").rglob("*.py"))
    files += [ROOT / path for path in (
        "src/controllers/bcrbc_mac.py",
        "src/runners/delayed_parallel_runner.py",
        "src/envs/wrappers/delayed_wrapper.py",
        "src/components/delay_model.py",
        "scripts/eval_scripts/evaluate_bcrbc.py",
    )]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_text(encoding="utf-8").encode())
    return digest.hexdigest()


def run_batch(directory, job, environment):
    """A result file is the completion marker; interrupted batches are rerun."""
    if (directory / "result.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    job_path = directory / "job.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    print(f"Evaluating {directory.parent.name}/{directory.name}", flush=True)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/eval_scripts/evaluate_bcrbc.py"),
             str(job_path)],
            cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=True,
        )


def main():
    config = ROOT / "results/sacred" / TRAIN_RUN / "1/config.json"
    checkpoint = ROOT / "results/models" / TRAIN_RUN / str(CHECKPOINT_STEP)
    training_config = json.loads(config.read_text(encoding="utf-8"))
    metadata = {
        "protocol": PROTOCOL, "train_run": TRAIN_RUN,
        "train_map": training_config["env_args"]["map_name"], "test_map": TEST_MAP,
        "checkpoint_step": str(CHECKPOINT_STEP),
        "checkpoint_sha256": hashlib.sha256(
            (checkpoint / "agent.th").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "source_sha256": source_digest(), "seed": SEED, "parallel": PARALLEL,
        "q_softmax_temperature": 1.0,
    }
    protocol_hash = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()[:12]
    output = (ROOT / "results/evaluate" / TRAIN_RUN
              / f"checkpoint_{CHECKPOINT_STEP}" / f"test_map_{TEST_MAP}"
              / f"{PROTOCOL}_{protocol_hash}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # Round up to complete parallel batches so later extensions keep batch IDs stable.
    batch_count = (EPISODES + PARALLEL - 1) // PARALLEL
    for condition in conditions():
        directory = output / condition_name(condition)
        for batch_index in range(batch_count):
            job = {
                "config": str(config), "checkpoint": str(checkpoint),
                "test_map": TEST_MAP, "condition": condition,
                "seed": SEED + batch_index * PARALLEL,
                "parallel": PARALLEL, "episodes": PARALLEL,
                "episode_offset": batch_index * PARALLEL,
            }
            run_batch(directory / f"batch_{batch_index:03d}", job, environment)
            refresh_summary(output)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/eval_scripts/plots/plot_bcrbc_delay_grid.py"),
         str(output)], check=True,
    )
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
