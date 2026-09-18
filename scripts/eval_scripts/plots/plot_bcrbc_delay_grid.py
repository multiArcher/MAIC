"""Plot saved decision diagnostics; no environment or model execution."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_grid(output):
    rows = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for cap in sorted({row["cap"] for row in rows}):
        gaussian = [row for row in rows if row["delay_type"] == "gaussian" and row["cap"] == cap]
        if not gaussian:
            continue
        means = sorted({row["mean"] for row in gaussian})
        stds = sorted({row["std"] for row in gaussian})
        fields = [
            ("win_rate", "Generated policy win rate"),
            ("generated_agreement", "Generated / reference agreement"),
            ("mask_agreement", "Mask / reference agreement"),
            ("generated_kl", "KL(reference || generated)"),
            ("mask_kl", "KL(reference || mask)"),
            ("decision_count", "Eligible agent decisions"),
        ]
        kl_max = max(
            (row[key] for row in gaussian for key in ("generated_kl", "mask_kl")
             if row[key] is not None), default=1,
        )
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        for axis, (field, title) in zip(axes.flat, fields):
            values = np.full((len(stds), len(means)), np.nan)
            for row in gaussian:
                value = row[field]
                if value is not None:
                    values[stds.index(row["std"]), means.index(row["mean"])] = value
            maximum = 1 if field in ("win_rate", "generated_agreement", "mask_agreement") else (
                kl_max if field.endswith("kl") else None
            )
            picture = axis.imshow(values, origin="lower", cmap="cividis", vmin=0, vmax=maximum)
            axis.set_xticks(range(len(means)), means)
            axis.set_yticks(range(len(stds)), stds)
            axis.set(xlabel="Gaussian mean (steps)", ylabel="Gaussian std (steps)", title=title)
            for (y, x), value in np.ndenumerate(values):
                label = "—" if np.isnan(value) else (
                    f"{value:.0f}" if field == "decision_count" else f"{value:.2f}"
                )
                axis.text(x, y, label, ha="center", va="center", fontsize=9,
                          color="white" if np.isnan(value) or picture.norm(value) < .5 else "black")
            fig.colorbar(picture, ax=axis, shrink=.75)
        counts = sorted({row["episodes"] for row in gaussian})
        fig.suptitle(
            f"Delay cap {cap}; episodes per point: {counts}\n"
            "Agreement/KL: missing observation and >1 legal action only; — = no eligible decisions",
            fontsize=12,
        )
        fig.savefig(figures / f"delay_grid_cap_{cap}.png", dpi=200)
        fig.savefig(figures / f"delay_grid_cap_{cap}.pdf")
        plt.close(fig)
    for path in sorted(output.glob("*/batch_*/trajectories.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            plot_trajectory(json.loads(line), path.parent.parent / "figures")


def plot_trajectory(episode, directory):
    steps = episode["steps"]
    paths = ("reference", "generated", "mask", "actual")
    actions = np.asarray([[step[path] for path in paths] for step in steps])
    missing = np.asarray([step["missing"] for step in steps])
    agents = actions.shape[-1]
    fig, axes = plt.subplots(agents + 1, 1, figsize=(14, 1.4 * (agents + 1)),
                             sharex=True, constrained_layout=True)
    action_count = int(actions.max()) + 1
    colormap = plt.get_cmap("tab20", action_count)
    for agent, axis in enumerate(axes[:-1]):
        image = axis.imshow(actions[:, :, agent].T, aspect="auto", interpolation="nearest",
                            cmap=colormap, vmin=-.5, vmax=action_count - .5)
        axis.set_yticks(range(4), paths)
        axis.set_ylabel(f"Agent {agent}")
    axes[-1].imshow(missing.T, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axes[-1].set_yticks(range(agents), range(agents))
    axes[-1].set(xlabel="Environment step", ylabel="Missing obs\n(black = missing)")
    fig.colorbar(image, ax=list(axes[:-1]), ticks=range(action_count),
                 label="Action index", fraction=.015)
    fig.suptitle(f"Episode {episode['episode']}: three policies on the SAME executed trajectory")
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / f"actions_episode_{episode['episode']:03d}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_grid(Path(sys.argv[1]))
