from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = [
    "node_id",
    "signal",
    "segment_id",
    "threshold_multiplier",
    "loss_probability",
    "seed",
    "freshness_deadline_minutes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired weighted-drift versus hold analysis")
    parser.add_argument("--input", default="results/raw/agriculture_experiments.csv")
    parser.add_argument("--output", default="results/summaries/paired_comparison.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    pivot = df.pivot_table(index=KEYS, columns="estimator", values="mse", aggfunc="first")
    pivot = pivot.dropna(subset=["hold", "weighted_drift"]).reset_index()
    pivot["mse_difference"] = pivot["weighted_drift"] - pivot["hold"]
    pivot["mse_improvement_percent"] = np.where(
        pivot["hold"] > 0,
        100 * (pivot["hold"] - pivot["weighted_drift"]) / pivot["hold"],
        np.nan,
    )

    group = ["signal", "threshold_multiplier", "loss_probability"]
    summary = (
        pivot.groupby(group)
        .agg(
            paired_runs=("mse_difference", "size"),
            mean_mse_difference=("mse_difference", "mean"),
            std_mse_difference=("mse_difference", "std"),
            mean_improvement_percent=("mse_improvement_percent", "mean"),
            improvement_fraction=("mse_difference", lambda x: float(np.mean(x < 0))),
        )
        .reset_index()
    )
    summary["difference_ci95_halfwidth"] = 1.96 * summary["std_mse_difference"] / np.sqrt(
        summary["paired_runs"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
