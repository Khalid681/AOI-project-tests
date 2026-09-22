from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from aoi_project.baseline import run_baseline_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the loss-free base-paper benchmark")
    parser.add_argument("--slots", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    output = Path(args.output)
    raw_dir = output / "raw"
    figure_dir = output / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = run_baseline_sweep(list(range(1, 11)), args.slots, args.seed)
    results.to_csv(raw_dir / "baseline_validation.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [
        ("update_rate", "theoretical_update_rate", "Update rate"),
        ("mean_aoi", "theoretical_mean_aoi", "Mean AoI (slots)"),
        ("mse", "theoretical_mse", "MSE"),
    ]
    for axis, (simulated, theoretical, ylabel) in zip(axes, pairs):
        axis.plot(results["threshold"], results[theoretical], "k--", label="Analytical")
        axis.plot(results["threshold"], results[simulated], "o-", label="Simulation")
        axis.set_xlabel("Threshold T")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_validation.png", dpi=200)
    print(results.to_string(index=False))
    print(f"Saved baseline results to {raw_dir / 'baseline_validation.csv'}")


if __name__ == "__main__":
    main()

