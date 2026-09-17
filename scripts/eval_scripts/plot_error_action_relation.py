from study_plotting import load, save, plt, bucket_data


def main():
    root, _ = load()
    data = bucket_data(root, "error_bin")
    data["value"] = data.value.astype(int)
    for (map_name, model), rows in data.groupby(["map", "model_id"]):
        fig, ax = plt.subplots()
        for condition, group in rows.groupby("condition_id"):
            totals = group.groupby("value").sum(numeric_only=True)
            ax.plot(totals.index, 1-totals.agreement_sum/totals["count"], "o-", label=condition)
        ax.set(xlabel="floor(log10(latent MSE))", ylabel="Action disagreement fraction",
               title=f"{map_name}: {model} (association, not causation)")
        ax.legend(fontsize=6, bbox_to_anchor=(1, 1))
        save(root, f"error_action_{map_name}_{model}", fig)


if __name__ == "__main__":
    main()
