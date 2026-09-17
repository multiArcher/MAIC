from study_plotting import load, save, plt, json


def main():
    root, data = load()
    for map_name, rows in data.groupby("map"):
        fig, ax = plt.subplots()
        for (model, family), group in rows.groupby(["model_id", "distribution"]):
            group = group.sort_values("sampled_delay_mean")
            ax.errorbar(group.sampled_delay_mean.to_numpy(), group.win_rate.to_numpy(),
                        yerr=[(group.win_rate-group.win_low).clip(lower=0).to_numpy(),
                              (group.win_high-group.win_rate).clip(lower=0).to_numpy()],
                        fmt="o", capsize=2, label=f"{model}: {family}")
        ax.set(xlabel="Measured packet delay mean", ylabel="Win rate", ylim=(0, 1), title=map_name)
        ax.legend(fontsize=7)
        save(root, f"distributions_{map_name}", fig)


if __name__ == "__main__":
    main()
