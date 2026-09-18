"""Continuous episode-time diagnostics across delay regime changes."""
from collections import defaultdict
import gzip
import json

import pandas as pd

from study_plotting import load, save, plt, percent_axis, error_limit, aggregate_runs, band


def main():
    root, summary = load(raw=True)
    curves = []
    dynamic = summary[summary.condition_id.str.startswith(("periodic", "markov"))]
    for row in dynamic.itertuples():
        totals = defaultdict(lambda: dict(count=0, error=0., agreement=0.,
                                         episodes=0, high=0))
        directory = root / "runs" / row.model_id / row.condition_id
        for path in directory.glob("batch_*/decisions.jsonl.gz"):
            if not (path.parent / "result.json").exists():
                continue
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    decision = json.loads(line)
                    total = totals[decision["step"]]
                    if decision["agent"] == 0:
                        total["episodes"] += 1
                        total["high"] += decision["regime"] == 1
                    if decision["eligible"]:
                        total["count"] += 1
                        total["error"] += decision["generated_z_mse"]
                        total["agreement"] += decision["agreement"]
        data = pd.DataFrame.from_dict(totals, orient="index").sort_index()
        count = data["count"].replace(0, float("nan"))
        data["mse"] = data.error / count
        data["agreement"] = data.agreement / count
        data["high_fraction"] = data.high / data.episodes
        data = data.rename_axis("step").reset_index()
        data["map"] = row.map
        data["model_id"] = row.model_id
        data["condition_id"] = row.condition_id
        curves.append(data)

    if not curves:
        return
    runs = pd.concat(curves, ignore_index=True)
    runs.to_csv(root / "tables/dynamic_per_run.csv", index=False)
    summary = aggregate_runs(runs, ["map", "condition_id", "step"],
                             ["mse", "agreement", "high_fraction"])
    summary.to_csv(root / "tables/run_dynamic.csv", index=False)
    limit = error_limit(summary.mse + summary.mse_std.fillna(0))
    for (map_name, condition), data in summary.groupby(["map", "condition_id"]):
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
        band(axes[0], data, "step", "mse")
        axes[0].set(ylabel="Latent MSE", ylim=(0, limit))
        band(axes[1], data, "step", "agreement")
        axes[1].set_ylabel("Action agreement")
        percent_axis(axes[1])
        # Markov switch times differ by episode, unlike the periodic schedule.
        band(axes[2], data, "step", "high_fraction")
        axes[2].set(ylabel="Episodes in high-delay regime", xlabel="Environment step")
        percent_axis(axes[2])
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.suptitle(f"{map_name}, {condition}: run mean ± SD")
        save(root, f"dynamic_{map_name}_{condition}", fig)


if __name__ == "__main__":
    main()
