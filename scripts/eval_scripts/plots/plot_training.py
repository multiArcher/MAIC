"""Training-step curves, separate from post-training delay evaluation."""
import re
from pathlib import Path
import sys

import pandas as pd

from study_plotting import save, plt, band, percent_axis


def main():
    root = Path(sys.argv[1])
    data = pd.read_csv(root / "tables/training_summary.csv")
    raw = pd.read_csv(root / "tables/training_per_run.csv")
    for (map_name, metric), rows in data.groupby(["map", "metric"]):
        fig, ax = plt.subplots()
        originals = raw[(raw["map"] == map_name) & (raw.metric == metric)]
        for _, run in originals.groupby("run_id"):
            run = run.sort_values("step")
            ax.plot(run.step, run.value, color="grey", alpha=0.2, linewidth=0.7)
        band(ax, rows, "step", "value", "Run mean ± SD")
        ax.set(xlabel="Environment steps", ylabel=metric,
               title=f"{map_name}: {metric} ({int(rows.value_n.max())} runs)")
        if "battle_won" in metric or metric.endswith(("agreement", "epsilon", "fraction")):
            percent_axis(ax)
        ax.grid(alpha=0.2)
        ax.legend()
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", metric)
        save(root, f"{map_name}_{name}", fig, category="training")


if __name__ == "__main__":
    main()
