from study_plotting import load, save, plt, bucket_data


def main():
    root, _ = load()
    for grouping in ("age", "missing_length"):
        data = bucket_data(root, grouping)
        data["value"] = data.value.astype(int)
        for (map_name, model, condition), rows in data.groupby(["map", "model_id", "condition_id"]):
            summed = rows.groupby("value").sum(numeric_only=True)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, space in zip(axes, ["z", "obs"]):
                for path in ("mask", "generated"):
                    values = summed[f"{path}_{space}_mse_sum"]/summed["count"]
                    valid = summed.index >= 0
                    line, = ax.plot(summed.index[valid], values[valid], "o-", label=path)
                    if -1 in summed.index:
                        ax.scatter([-1], [values.loc[-1]], color=line.get_color(), marker="x")
                ax.set(xlabel=grouping + " (-1 = never arrived)", ylabel=f"{space} MSE")
                ax.set_xticks(summed.index)
                ax.legend()
            fig.suptitle(f"{map_name}: {model}, {condition}")
            save(root, f"reconstruction_{grouping}_{map_name}_{model}_{condition}", fig)


if __name__ == "__main__":
    main()
