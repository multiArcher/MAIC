from study_plotting import load, save, plt, percent_axis
import pandas as pd


def main():
    root, raw = load(raw=True)
    summary = pd.read_csv(root / "tables/run_efficiency.csv")
    for map_name, rows in raw.groupby("map"):
        fig, ax = plt.subplots()
        for hardware, group in rows.groupby("hardware"):
            points = ax.scatter(group.decision_ms, group.win_rate, alpha=0.3,
                                label=f"{hardware}: individual runs")
            means = summary[(summary["map"] == map_name) & (summary.hardware == hardware)]
            ax.errorbar(means.decision_ms, means.win_rate,
                        xerr=means.decision_ms_std.fillna(0),
                        yerr=means.win_rate_std.fillna(0), fmt="D", capsize=2,
                        color=points.get_facecolor()[0], label=f"{hardware}: mean ± SD")
        ax.set(xlabel="Action-selection batch latency (ms, excludes scoring)",
               ylabel="Win rate", title=map_name)
        percent_axis(ax)
        ax.legend(fontsize=7)
        save(root, f"efficiency_{map_name}", fig)


if __name__ == "__main__":
    main()
