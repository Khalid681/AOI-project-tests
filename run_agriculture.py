from __future__ import annotations

import argparse
import json
from pathlib import Path

from aoi_project.data import build_trace_splits, load_sensor_data
from aoi_project.experiments import run_agriculture_sweep, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agricultural trace experiments")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--output", default="results")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke experiment")
    parser.add_argument("--max-splits", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = load_sensor_data(config["data_path"])
    splits = build_trace_splits(
        data,
        signals=config["signals"],
        nodes=config["nodes"],
        gap_multiplier=config["split_gap_multiplier"],
        minimum_segment_records=config["minimum_segment_records"],
        train_fraction=config["train_fraction"],
        validation_fraction=config["validation_fraction"],
    )
    if args.max_splits is not None:
        splits = splits[: args.max_splits]

    threshold_multipliers = config["threshold_multipliers"]
    loss_probabilities = config["loss_probabilities"]
    seeds = config["seeds"]
    deadlines = config["freshness_deadline_multipliers"]
    beta_grid = config["beta_grid"]
    if args.quick:
        splits = splits[:2]
        threshold_multipliers = threshold_multipliers[-2:]
        loss_probabilities = loss_probabilities[1::2]
        seeds = seeds[:3]
        deadlines = deadlines[:1]
        beta_grid = beta_grid[::2]

    print(f"Running {len(splits)} node-signal segments")
    results, calibration = run_agriculture_sweep(
        splits=splits,
        threshold_multipliers=threshold_multipliers,
        loss_probabilities=loss_probabilities,
        beta_grid=beta_grid,
        seeds=seeds,
        freshness_deadline_multipliers=deadlines,
        feedback_model=config["feedback_model"],
        delivery_delay_minutes=config["delivery_delay_minutes"],
    )
    paths = save_results(results, calibration, args.output)
    print(f"Completed {len(results):,} estimator-level rows")
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
