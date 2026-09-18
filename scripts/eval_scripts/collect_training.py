"""Read training logs without loading a model or launching environments."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def metric_name(tag):
    # These namespaces differ across logger backends but carry the same metrics.
    prefix, _, name = tag.partition("/")
    return name if prefix in ("metric", "running", "loss", "q_values") else tag


def read_tensorboard(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util.tensor_util import make_ndarray

    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0, "tensors": 0})
    accumulator.Reload()
    series = {}
    for tag in accumulator.Tags()["scalars"]:
        series[metric_name(tag)] = [(e.step, e.value) for e in accumulator.Scalars(tag)]
    for tag in accumulator.Tags()["tensors"]:
        values = [(e.step, make_ndarray(e.tensor_proto)) for e in accumulator.Tensors(tag)]
        if all(value.size == 1 for _, value in values):
            series.setdefault(metric_name(tag), [(step, float(value.item())) for step, value in values])
    return series


def align_runs(raw):
    output = []
    for (map_name, metric), group in raw.groupby(["map", "metric"]):
        runs = [run.sort_values("step").drop_duplicates("step", keep="last")
                for _, run in group.groupby("run_id")]
        start = max(run.step.min() for run in runs)
        end = min(run.step.max() for run in runs)
        steps = np.unique(np.concatenate([run.step.to_numpy() for run in runs]))
        steps = steps[(steps >= start) & (steps <= end)]
        if not len(steps):
            continue
        values = np.stack([np.interp(steps, run.step, run.value) for run in runs])
        means = values.mean(axis=0)
        stds = values.std(axis=0, ddof=1) if len(runs) > 1 else np.full(len(steps), np.nan)
        output.extend(dict(map=map_name, metric=metric, step=int(step),
                           value=mean, value_std=std, value_n=len(runs))
                      for step, mean, std in zip(steps, means, stds))
    return pd.DataFrame(output, columns=["map", "metric", "step", "value", "value_std", "value_n"])


def main(root):
    manifest = json.loads((root / "manifest.json").read_text())
    rows, report = [], []
    expected = {"test_battle_won_mean", "test_return_mean", "battle_won_mean",
                "return_mean", "ep_length_mean", "epsilon", "grad_norm",
                "td_loss", "total_loss", "rec_loss", "flow_loss"}
    for model in manifest.get("models", []):
        config_path = ROOT / model["config"]
        config = json.loads(config_path.read_text())
        metrics_path = config_path.parent / "metrics.json"
        series, sources = {}, {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            for tag, values in metrics.items():
                key = metric_name(tag)
                series[key] = list(zip(values["steps"], values["values"]))
                sources[key] = str(metrics_path)
        if "tensorboard" in model:
            directory = ROOT / model["tensorboard"]
            for key, values in read_tensorboard(directory).items():
                if key not in series:
                    series[key] = values
                    sources[key] = str(directory)
        for metric, values in series.items():
            for step, value in values:
                if value is not None and np.isscalar(value) and np.isfinite(value):
                    rows.append(dict(run_id=model["id"], map=config["env_args"]["map_name"],
                                     metric=metric, step=step, value=value, source=sources[metric]))
        report.append(dict(run_id=model["id"], sources=sources,
                           missing_expected=sorted(expected - series.keys())))
        print(f"Training logs: {model['id']}, {len(series)} metrics; "
              f"missing: {', '.join(sorted(expected - series.keys()))}", flush=True)
    raw = pd.DataFrame(rows, columns=["run_id", "map", "metric", "step", "value", "source"])
    raw = raw.drop_duplicates(["run_id", "map", "metric", "step"], keep="last")
    for item in report:
        item["missing_from_other_runs"] = sorted(set(raw.metric) - item["sources"].keys())
    tables = root / "tables"
    tables.mkdir(exist_ok=True)
    raw.to_csv(tables / "training_per_run.csv", index=False)
    align_runs(raw).to_csv(tables / "training_summary.csv", index=False)
    (tables / "training_sources.json").write_text(json.dumps(report, indent=2))
    if not len(raw):
        print("No training scalar data found; training plots will be skipped.", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
