from study_plotting import load, save, plt, bucket_data


def main():
    root, _ = load()
    data = bucket_data(root, "dynamic")
    data = data[data.condition_id.str.startswith(("periodic", "markov"))]
    for (map_name, model, condition), rows in data.groupby(["map", "model_id", "condition_id"]):
        parts = rows.value.str.split(":", expand=True).astype(int)
        rows = rows.assign(regime=parts[0], elapsed=parts[1])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for regime, group in rows.groupby("regime"):
            totals = group.groupby("elapsed").sum(numeric_only=True)
            axes[0].plot(totals.index, totals.generated_z_mse_sum/totals["count"], label=f"regime {regime}")
            axes[1].plot(totals.index, totals.agreement_sum/totals["count"], label=f"regime {regime}")
        for ax, label in zip(axes, ["Latent MSE", "Action agreement"]):
            ax.set(xlabel="Steps since regime entry (includes initial regime)", ylabel=label)
            ax.legend()
        fig.suptitle(f"{map_name}: {model}, {condition}")
        save(root, f"dynamic_{map_name}_{model}_{condition}", fig)


if __name__ == "__main__":
    main()
