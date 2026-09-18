from study_plotting import load, save, plt, bucket_data, percent_axis, band


def main():
    root, _ = load()
    data = bucket_data(root, "error_bin")
    if data.empty:
        return
    data["value"] = data.value.astype(int)
    data["disagreement"] = 1 - data.agreement
    data["disagreement_std"] = data.agreement_std
    data["disagreement_n"] = data.agreement_n
    for map_name, rows in data.groupby("map"):
        fig, ax = plt.subplots()
        for condition, group in rows.groupby("condition_id"):
            band(ax, group, "value", "disagreement", condition)
        ax.set(xlabel="floor(log10(latent MSE))", ylabel="Action disagreement",
               title=f"{map_name}: run mean ± SD (association, not causation)")
        percent_axis(ax)
        ax.legend(fontsize=6, bbox_to_anchor=(1, 1))
        save(root, f"error_action_{map_name}", fig)


if __name__ == "__main__":
    main()
