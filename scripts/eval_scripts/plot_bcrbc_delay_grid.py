"""Aggregate completed on/off grid results and draw delay heatmaps."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    data = pd.read_csv(options.output / "results.csv")
    aggregates = []
    keys = ["obs_mean", "comm_mean", "completion"]
    for condition, frame in data.groupby(keys, dropna=False):
        row = dict(zip(keys, condition))
        n = frame["episodes"].sum()
        wins = (frame["running/test_battle_won_mean"] * frame["episodes"]).sum()
        p = wins / n
        scale = 1 + 1.96 ** 2 / n
        center = (p + 1.96 ** 2 / (2 * n)) / scale
        radius = 1.96 * np.sqrt(p * (1 - p) / n + 1.96 ** 2 / (4 * n ** 2)) / scale
        row.update(episodes=n, wins=wins, win_rate=p,
                   win_ci_low=center - radius, win_ci_high=center + radius,
                   evaluation_seeds=len(frame))
        row["return_mean"] = (
            frame["metric/test_return_mean"] * frame["episodes"]
        ).sum() / n
        for column in frame.columns:
            if column.endswith("_sum"):
                group = next(g for g in ("missing", "never_arrived", "stale_arrived")
                             if column.startswith(g + "_"))
                count = frame[group + "_count"].sum()
                row[column[:-4]] = frame[column].sum() / count if count else np.nan
        row["missing_count"] = frame["missing_count"].sum()
        row["missing_fraction"] = row["missing_count"] / frame["agent_steps"].sum()
        aggregates.append(row)
    table = pd.DataFrame(aggregates)
    table.to_csv(options.output / "aggregate.csv", index=False)
    grid = table.dropna(subset=["obs_mean", "comm_mean"])
    means = sorted(set(grid["obs_mean"]) | set(grid["comm_mean"]))
    modes = sorted(grid["completion"].unique())
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    figures = options.output / "figures"
    figures.mkdir(exist_ok=True)
    metrics = ["win_rate", "return_mean", "missing_used_z_mse",
               "missing_used_obs_mse", "missing_tokenizer_mse",
               "missing_stale_obs_mse", "missing_completion_gain"]
    for metric in metrics:
        matrices = []
        for completion in modes:
            subset = grid[grid["completion"] == completion]
            matrix = subset.pivot(index="obs_mean", columns="comm_mean", values=metric)
            matrices.append(matrix.reindex(index=means, columns=means).to_numpy())
        finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
        if not finite.size:
            continue
        low, high = (0, 1) if metric == "win_rate" else (finite.min(), finite.max())
        panels = 3 if len(modes) == 2 else 1
        fig, axes = plt.subplots(
            1, panels, figsize=(5 * panels, 4.6), layout="constrained",
            squeeze=False,
        )
        axes = axes[0]
        titles = ["Completion on" if mode else "Completion off" for mode in modes]
        for axis, matrix, title in zip(axes, matrices, titles):
            picture = axis.imshow(matrix, origin="lower", interpolation="none",
                                  cmap="cividis", vmin=low, vmax=high)
            axis.set_title(title)
            fig.colorbar(picture, ax=axis)
        if len(modes) == 2:
            difference = matrices[1] - matrices[0]
            bound = np.nanmax(np.abs(difference))
            if not np.isfinite(bound) or bound == 0:
                bound = 1.0
            picture = axes[2].imshow(
                difference, origin="lower", interpolation="none",
                cmap="PuOr", vmin=-bound, vmax=bound,
            )
            axes[2].set_title("On minus off")
            fig.colorbar(picture, ax=axes[2])
        for axis in axes:
            axis.set_xticks(range(len(means)), [f"{mu:g}" for mu in means], rotation=45)
            axis.set_yticks(range(len(means)), [f"{mu:g}" for mu in means])
            axis.set_xlabel("Communication Gaussian mean")
            axis.set_ylabel("Observation Gaussian mean")
        fig.suptitle(metric + " | sigma=1; blank cells are unavailable")
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"{metric}.{extension}", dpi=220)
        plt.close(fig)
    metadata = {
        "completed_jobs": len(data),
        "episodes": int(data["episodes"].sum()),
        "checkpoint": data["checkpoint"].unique().tolist(),
        "uncertainty": "Wilson 95% episode-level intervals; one training seed",
        "interpretation": "On/off trajectories differ. Used-vector errors describe each actual rollout; completion gain is within-trajectory stale minus decoded error. Off-mode gain is not generative benefit.",
    }
    (options.output / "summary.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
