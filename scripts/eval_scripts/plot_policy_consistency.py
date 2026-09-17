from study_plotting import load, save, plt


def main():
    root, data = load()
    metrics = ["agreement", "q_softmax_kl", "reference_q_gap"]
    metrics += [m for m in ["intervention_agreement", "correction", "damage"] if m in data]
    for map_name, rows in data.groupby("map"):
        for metric in metrics:
            fig, ax = plt.subplots()
            for model, group in rows.groupby("model_id"):
                ax.errorbar(group.sampled_delay_mean.to_numpy(), group[metric].to_numpy(),
                            yerr=[(group[metric]-group[metric+"_low"]).clip(lower=0).to_numpy(),
                                  (group[metric+"_high"]-group[metric]).clip(lower=0).to_numpy()], fmt="o", label=model)
            ax.set(xlabel="Measured packet delay mean", ylabel=metric, title=f"{map_name}: matched-trajectory diagnostic")
            ax.legend()
            save(root, f"policy_{map_name}_{metric}", fig)


if __name__ == "__main__":
    main()
