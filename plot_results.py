from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "hold": "Last-value hold",
    "full_drift": "Full drift",
    "weighted_drift": "Weighted drift",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot experiment summary")
    parser.add_argument(
        "--input", default="results/summaries/agriculture_summary.csv"
    )
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        raise SystemExit("The summary file contains no experiment rows")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for signal in sorted(df["signal"].unique()):
        subset = df[df["signal"] == signal]
        deadline = subset["freshness_deadline_minutes"].min()
        subset = subset[subset["freshness_deadline_minutes"] == deadline]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for estimator in ["hold", "full_drift", "weighted_drift"]:
            est = subset[subset["estimator"] == estimator]
            grouped = (
                est.groupby("loss_probability", as_index=False)
                .agg(mse=("mse_mean", "mean"), aoi=("mean_aoi_minutes", "mean"))
            )
            axes[0].plot(
                grouped["loss_probability"], grouped["mse"], "o-", label=LABELS[estimator]
            )
            axes[1].plot(
                grouped["loss_probability"], grouped["aoi"], "o-", label=LABELS[estimator]
            )
        axes[0].set_ylabel("Mean squared error")
        axes[1].set_ylabel("Mean AoI (minutes)")
        for axis in axes:
            axis.set_xlabel("Packet-loss probability")
            axis.grid(alpha=0.25)
        axes[0].legend()
        fig.suptitle(f"{signal}: accuracy and freshness under packet loss")
        fig.tight_layout()
        fig.savefig(out / f"{signal.lower()}_loss_comparison.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()

