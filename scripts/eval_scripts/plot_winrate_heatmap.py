from study_plotting import load, save, plt, json
import numpy as np


def main():
    root, data = load()
    cells = json.loads((root / "manifest.json").read_text())["gaussian_cells"]
    means, stds = sorted({c["mean"] for c in cells}), sorted({c["std"] for c in cells})
    for (map_name, model), rows in data.groupby(["map", "model_id"]):
        values = rows.set_index("condition_id").win_rate
        grid = np.full((len(stds), len(means)), np.nan)
        for cell in cells:
            grid[stds.index(cell["std"]), means.index(cell["mean"])] = values.get(cell["condition_id"], np.nan)
        fig, ax = plt.subplots()
        im = ax.imshow(grid, origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax.set(xticks=range(len(means)), xticklabels=means, yticks=range(len(stds)),
               yticklabels=stds, xlabel="Gaussian location (before clipping)",
               ylabel="Gaussian scale (before clipping)", title=f"{map_name}: {model}")
        for (y, x), value in np.ndenumerate(grid):
            if np.isfinite(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, label="Win rate")
        save(root, f"winrate_{map_name}_{model}", fig)


if __name__ == "__main__":
    main()
