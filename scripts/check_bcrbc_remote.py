"""Print persisted metrics for the September 2026 remote BCRBC experiment."""

import json
import math
from pathlib import Path


root = Path(__file__).resolve().parents[1]
for mode in ("pilot", "train"):
    runs = sorted(
        (root / "results" / "sacred").glob(
            f"bcrbc_8m_{mode}_20260905__8m__*/1"
        )
    )
    if not runs:
        continue

    run_dir = runs[-1]
    run = json.loads((run_dir / "run.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    selected_metrics = {}
    nonfinite_metrics = []
    for key, series in metrics.items():
        if any(not math.isfinite(value) for value in series["values"]):
            nonfinite_metrics.append(key)
        if key.startswith("loss/") or key in (
            "running/episode",
            "running/epsilon",
            "running/grad_norm",
            "running/battle_won_mean",
            "running/test_battle_won_mean",
            "running/test_ep_length_mean",
            "metric/return_mean",
            "metric/test_return_mean",
        ):
            selected_metrics[key] = list(
                zip(series["steps"][-5:], series["values"][-5:])
            )

    summary = {
        "mode": mode,
        "run_dir": str(run_dir),
        "status": run["status"],
        "start_time": run["start_time"],
        "seed": config["seed"],
        "t_max": config["t_max"],
        "batch_size": config["batch_size"],
        "nonfinite_metrics": nonfinite_metrics,
        "recent_metrics": selected_metrics,
    }
    print(json.dumps(summary, indent=2))
