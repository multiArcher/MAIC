from study_plotting import load, save, plt, percent_axis


def main():
    root, data = load()
    for map_name, rows in data.groupby("map"):
        fig, ax = plt.subplots()
        for family, group in rows.groupby("distribution"):
            ax.errorbar(group.sampled_delay_mean, group.win_rate,
                        yerr=group.win_rate_std.fillna(0), fmt="o",
                        capsize=2, label=family)
        ax.set(xlabel="Measured packet delay mean", ylabel="Win rate",
               title=f"{map_name}: run mean ± SD")
        percent_axis(ax)
        ax.legend()
        save(root, f"distributions_{map_name}", fig)


if __name__ == "__main__":
    main()
