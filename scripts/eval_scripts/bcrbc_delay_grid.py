"""Fixed-checkpoint evaluation. Edit the experiment settings below."""

import csv
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TRAIN_RUN = "TRAIN_RUN_NAME"  # Set in a .local.py copy for each experiment.
CONFIG = ROOT / "results" / "sacred" / TRAIN_RUN / "1" / "config.json"
CHECKPOINT = ROOT / "results" / "models" / TRAIN_RUN / "CHECKPOINT_STEP"
# Use a new experiment directory when changing the checkpoint or conditions.
OUTPUT = ROOT / "results/evaluate/EXPERIMENT_NAME/delay_grid"
SEEDS = [101]
EPISODES = 16
PARALLEL = 4
COMPLETION = [True]
MEANS = [-1, -0.5, 0, 0.5, 1]
CONDITIONS = list(itertools.product(MEANS, MEANS))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    jobs = []
    for index, ((obs_mean, comm_mean), completion, seed) in enumerate(
        itertools.product(CONDITIONS, COMPLETION, SEEDS)
    ):
        job = {
            "config": str(CONFIG), "checkpoint": str(CHECKPOINT),
            "obs_mean": obs_mean, "comm_mean": comm_mean,
            "completion": completion, "seed": seed,
            "parallel": PARALLEL, "episodes": EPISODES,
        }
        directory = OUTPUT / f"job_{index:04d}"
        directory.mkdir(exist_ok=True)
        job_path = directory / "job.json"
        job_path.write_text(json.dumps(job, indent=2))
        jobs.append(job)
        print(f"Evaluating {directory.name}", flush=True)
        with (directory / "run.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/eval_scripts/evaluate_bcrbc.py"),
                 str(job_path)],
                cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, check=True,
            )
        results = [
            json.loads(path.read_text())
            for path in sorted(OUTPUT.glob("job_*/result.json"))
        ]
        fields = sorted(set().union(*(row.keys() for row in results)))
        with (OUTPUT / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    (OUTPUT / "manifest.json").write_text(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    main()
