"""Small shared I/O helpers; each figure's analysis lives in its own script."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load():
    root = Path(sys.argv[1])
    return root, pd.read_csv(root / "tables/summary.csv")


def save(root, name, figure):
    directory = root / "figures"
    directory.mkdir(exist_ok=True)
    figure.savefig(directory / (name + ".png"), dpi=180, bbox_inches="tight")
    figure.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    plt.close(figure)


def bucket_data(root, grouping):
    data = pd.read_csv(root / "tables/quality_buckets.csv")
    return data[data.grouping == grouping].copy()
