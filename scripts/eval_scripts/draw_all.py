"""Draw study figures: draw_all.py STUDY [MODEL CONDITION EPISODE [AGENT]]."""
from pathlib import Path
import subprocess
import sys


def main():
    directory = Path(__file__).resolve().parent
    study = Path(sys.argv[1]).resolve()
    subprocess.run(
        [sys.executable, str(directory / "summarize_study.py"), str(study)],
        check=True,
    )
    for script in ("aggregate_study.py", "collect_training.py"):
        subprocess.run([sys.executable, str(directory / script), str(study)], check=True)
    figures = [
        "winrate_heatmap", "distribution_robustness", "reconstruction_quality",
        "policy_consistency", "error_action_relation", "dynamic_response",
        "efficiency", "training",
    ]
    for index, name in enumerate(figures, start=1):
        print(f"[{index}/{len(figures)}] Drawing {name}", flush=True)
        subprocess.run(
            [sys.executable, str(directory / "plots" / f"plot_{name}.py"), str(study)],
            check=True,
        )
    # Episode examples require an explicit selection, not cherry-picked defaults.
    if len(sys.argv) > 2:
        subprocess.run(
            [sys.executable, str(directory / "plots/plot_episode_case.py"),
             str(study), *sys.argv[2:]], check=True,
        )
    print(f"Figures saved to {study / 'figures'}", flush=True)


if __name__ == "__main__":
    main()
