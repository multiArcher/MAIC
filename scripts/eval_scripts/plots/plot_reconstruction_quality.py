from study_plotting import load, save, plt, bucket_data, error_limit, band
import pandas as pd
from matplotlib.ticker import MaxNLocator


def main():
    root, _ = load()
    data = pd.concat([bucket_data(root, key) for key in ("age", "missing_length")])
    data["value"] = data.value.astype(int)
    limits = {space: error_limit(pd.concat([
        data[f"{path}_{space}_mse"] + data[f"{path}_{space}_mse_std"].fillna(0)
        for path in ("mask", "generated")])) for space in ("z", "obs")} if len(data) else {}
    for (grouping, map_name, condition), rows in data.groupby(["grouping", "map", "condition_id"]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, space in zip(axes, ("z", "obs")):
            for path in ("mask", "generated"):
                metric = f"{path}_{space}_mse"
                line = band(ax, rows[rows.value >= 0], "value", metric, path)
                never = rows[rows.value == -1]
                ax.errorbar(never.value, never[metric],
                            yerr=never[metric + "_std"].fillna(0),
                            fmt="x", color=line.get_color())
            label = "Observation age (-1 = never arrived)" if grouping == "age" else "Consecutive missing steps"
            ax.set(xlabel=label, ylabel=f"{space} MSE", ylim=(0, limits[space]))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.legend()
        fig.suptitle(f"{map_name}, {condition}: run mean ± SD")
        save(root, f"reconstruction_{grouping}_{map_name}_{condition}", fig)


if __name__ == "__main__":
    main()
