from study_plotting import load, save, plt


def main():
    root, data = load()
    for map_name, rows in data.groupby("map"):
        fig, ax = plt.subplots()
        for (model, hardware), group in rows.groupby(["model_id", "hardware"]):
            ax.scatter(group.decision_ms, group.win_rate, label=f"{model}: {hardware}")
        ax.set(xlabel="Action-selection batch latency (ms, excludes scoring)", ylabel="Win rate", title=map_name)
        ax.legend()
        save(root, f"efficiency_{map_name}", fig)


if __name__ == "__main__":
    main()
