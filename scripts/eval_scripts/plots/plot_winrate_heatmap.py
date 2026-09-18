from study_plotting import load, save, plt, json, PercentFormatter
import numpy as np


def main():
    root, data = load()
    manifest = json.loads((root / "manifest.json").read_text())
    for family in ("gaussian", "uniform"):
        if family + "_cells" not in manifest:
            continue
        cells = manifest[family + "_cells"]
        means = sorted({c["mean"] for c in cells})
        stds = sorted({c["std"] for c in cells})
        for map_name, rows in data.groupby("map"):
            values = rows.set_index("condition_id")
            grid = np.full((len(stds), len(means)), np.nan)
            fig, ax = plt.subplots(figsize=(8, 5))
            for cell in cells:
                key = cell["condition_id"]
                if key not in values.index:
                    continue
                row = values.loc[key]
                y, x = stds.index(cell["std"]), means.index(cell["mean"])
                grid[y, x] = row.win_rate
                label = f"{row.win_rate:.0%}"
                if row.win_rate_n > 1:
                    label += f" ± {100 * row.win_rate_std:.1f}"
                ax.text(x, y, label, ha="center", va="center", color="white", fontsize=8)
            im = ax.imshow(grid, origin="lower", vmin=0, vmax=1, cmap="viridis")
            ax.set(xticks=range(len(means)), xticklabels=means,
                   yticks=range(len(stds)), yticklabels=stds,
                   xlabel="Raw mean", ylabel="Raw standard deviation",
                   title=f"{map_name}: {family} (run mean ± SD, percentage points)")
            fig.colorbar(im, ax=ax, label="Win rate", format=PercentFormatter(1))
            save(root, f"winrate_{family}_{map_name}", fig)


if __name__ == "__main__":
    main()
