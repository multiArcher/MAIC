"""Edit this parameter block in a .local.py copy, then run it in the test environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = "delay_robustness"
# Each entry: id, config (saved training JSON), checkpoint (directory).
# For a legacy config missing bcrbc_time_block_every, explicitly supply
# legacy_time_block_every=4 in that model entry; never change the method.
MODELS = []
MEANS = [-2, -1, 0, 1, 2]
STDS = [0, 0.5, 1, 1.5, 2]
CAP = 8
EPISODES = 64
PARALLEL = 8
SEED = 101
MASK_INTERVENTION = False
# Optional raw-observation feature slices, specified per map from its actual layout.
FEATURE_GROUPS = {}


def conditions():
    cases = [{"id": "fixed_0", "kind": "fixed", "value": 0, "cap": CAP}]
    cells = []
    for mean in MEANS:
        for std in STDS:
            name = "fixed_0" if std == 0 and mean <= 0 else f"gaussian_{mean:g}_{std:g}"
            cells.append({"mean": mean, "std": std, "condition_id": name})
            if name != "fixed_0":
                cases.append(dict(id=name, kind="gaussian", mean=mean, std=std, cap=CAP))
    for value in (4, 8):
        cases.append(dict(id=f"fixed_{value}", kind="fixed", value=value, cap=CAP))
    for high in (1, 2, 4, 8):
        cases.append(dict(id=f"uniform_0_{high}", kind="uniform", low=0, high=high, cap=CAP))
    for name, means, probability in [("balanced", [0, 2], 0.5), ("rare_severe", [0, 4], 0.1)]:
        cases.append(dict(id=f"mixture_{name}", kind="mixture", means=means,
                          stds=[1, 1], high_probability=probability, cap=CAP))
    cases.append(dict(id="periodic_16", kind="periodic", means=[0, 2], stds=[1, 1], period=16, cap=CAP))
    cases.append(dict(id="markov_09", kind="markov", means=[0, 2], stds=[1, 1], stay_probability=0.9, cap=CAP))
    return cases, cells


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert MODELS, "Populate MODELS in a .local.py copy before evaluating."
    assert len({m["id"] for m in MODELS}) == len(MODELS), "Model IDs must be unique."
    output = ROOT / "results/evaluate" / STUDY
    output.mkdir(parents=True, exist_ok=True)
    cases, cells = conditions()
    source_files = sorted((ROOT / "src/modules/bcrbc").rglob("*.py"))
    source_files += sorted((ROOT / "scripts/eval_scripts").glob("*.py"))
    source_files += [ROOT / p for p in (
        "src/components/evaluation_delay.py", "src/components/delay_model.py",
        "src/controllers/bcrbc_mac.py", "src/runners/delayed_parallel_runner.py",
        "src/envs/wrappers/delayed_wrapper.py")]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    manifest = dict(protocol="delay_study_v2", models=MODELS, conditions=cases,
                    gaussian_cells=cells, episodes=EPISODES, parallel=PARALLEL,
                    seed=SEED, mask_intervention=MASK_INTERVENTION,
                    feature_groups=FEATURE_GROUPS, sources=sources,
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    path = output / "manifest.json"
    # A study directory must not silently mix incompatible data.
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous == manifest, "Study definition changed; use a new STUDY name."
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for model in MODELS:
        config_path = ROOT / model["config"]
        checkpoint = ROOT / model["checkpoint"]
        config = json.loads(config_path.read_text())
        for condition in cases:
            for offset in range(0, EPISODES, PARALLEL):
                directory = output / "runs" / model["id"] / condition["id"] / f"batch_{offset:04d}"
                directory.mkdir(parents=True, exist_ok=True)
                job = dict(model=model, config=str(config_path), checkpoint=str(checkpoint),
                           config_sha256=digest(config_path), checkpoint_sha256=digest(checkpoint / "agent.th"),
                           condition=condition, parallel=min(PARALLEL, EPISODES - offset),
                           seed=SEED + offset, episode_offset=offset,
                           mask_intervention=MASK_INTERVENTION,
                           feature_groups=FEATURE_GROUPS.get(config["env_args"]["map_name"], {}))
                job_path = directory / "job.json"
                if (directory / "result.json").exists():
                    assert json.loads(job_path.read_text()) == job, "Checkpoint/config changed; use a new STUDY."
                    continue
                job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
                print(model["id"], condition["id"], offset, flush=True)
                with (directory / "run.log").open("w") as log:
                    subprocess.run([sys.executable, str(ROOT / "scripts/eval_scripts/evaluate_study.py"),
                                    str(job_path)], cwd=ROOT, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/eval_scripts/summarize_study.py"), str(output)], check=True)


if __name__ == "__main__":
    main()
