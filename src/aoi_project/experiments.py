from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .channel import simulate_threshold_channel
from .data import TraceSplit, estimate_drift, increment_scale, trace_arrays
from .estimators import estimate_from_receptions
from .metrics import communication_metrics, estimation_metrics, freshness_metrics


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    threshold_multiplier: float
    drift: float
    beta: float
    validation_mse: float


def select_beta_on_validation(
    split: TraceSplit,
    threshold: float,
    drift: float,
    loss_probability: float,
    beta_grid: list[float],
    seeds: list[int],
    feedback_model: str,
    delivery_delay_minutes: float,
) -> tuple[float, float]:
    times, values = trace_arrays(split.validation)
    scores = {float(beta): [] for beta in beta_grid}

    for seed in seeds:
        reception = simulate_threshold_channel(
            times,
            values,
            threshold=threshold,
            loss_probability=loss_probability,
            seed=seed,
            feedback_model=feedback_model,
            delivery_delay_minutes=delivery_delay_minutes,
        )
        for beta in beta_grid:
            estimate = estimate_from_receptions(
                reception, "weighted_drift", drift=drift, beta=float(beta)
            )
            scores[float(beta)].append(estimation_metrics(values, estimate)["mse"])

    mean_scores = {beta: float(np.mean(values_)) for beta, values_ in scores.items()}
    best_beta = min(mean_scores, key=lambda beta: (mean_scores[beta], beta))
    return best_beta, mean_scores[best_beta]


def calibrate_split(
    split: TraceSplit,
    threshold_multiplier: float,
    loss_probability: float,
    beta_grid: list[float],
    calibration_seeds: list[int],
    feedback_model: str = "ack",
    delivery_delay_minutes: float = 0.0,
) -> CalibrationResult:
    drift = estimate_drift(split.train)
    threshold = float(threshold_multiplier * increment_scale(split.train))
    beta, validation_mse = select_beta_on_validation(
        split,
        threshold,
        drift,
        loss_probability,
        beta_grid,
        calibration_seeds,
        feedback_model,
        delivery_delay_minutes,
    )
    return CalibrationResult(
        threshold=threshold,
        threshold_multiplier=float(threshold_multiplier),
        drift=drift,
        beta=beta,
        validation_mse=validation_mse,
    )


def evaluate_split(
    split: TraceSplit,
    calibration: CalibrationResult,
    loss_probability: float,
    seeds: list[int],
    freshness_deadline_multipliers: list[float],
    feedback_model: str = "ack",
    delivery_delay_minutes: float = 0.0,
) -> list[dict[str, float | int | str]]:
    times, values = trace_arrays(split.test)
    duration = float(times[-1] - times[0])
    if duration <= 0:
        return []

    estimators = (
        ("hold", 0.0),
        ("full_drift", 1.0),
        ("weighted_drift", calibration.beta),
    )
    rows: list[dict[str, float | int | str]] = []

    for seed in seeds:
        reception = simulate_threshold_channel(
            times,
            values,
            threshold=calibration.threshold,
            loss_probability=loss_probability,
            seed=seed,
            feedback_model=feedback_model,
            delivery_delay_minutes=delivery_delay_minutes,
        )
        comm = communication_metrics(reception)

        for estimator, beta in estimators:
            estimate = estimate_from_receptions(
                reception,
                estimator,
                drift=calibration.drift,
                beta=beta,
            )
            accuracy = estimation_metrics(values, estimate)

            for deadline_multiplier in freshness_deadline_multipliers:
                deadline = deadline_multiplier * split.nominal_interval_minutes
                freshness = freshness_metrics(reception, deadline)
                rows.append(
                    {
                        "dataset": "Manzano2022COMPAG",
                        "node_id": split.node_id,
                        "signal": split.signal,
                        "segment_id": split.segment_id,
                        "split": "test",
                        "records": len(values),
                        "duration_minutes": duration,
                        "nominal_interval_minutes": split.nominal_interval_minutes,
                        "threshold": calibration.threshold,
                        "threshold_multiplier": calibration.threshold_multiplier,
                        "loss_probability": loss_probability,
                        "seed": seed,
                        "feedback_model": feedback_model,
                        "estimator": estimator,
                        "drift_per_minute": calibration.drift,
                        "beta": beta,
                        "selected_beta": calibration.beta,
                        "validation_mse": calibration.validation_mse,
                        **accuracy,
                        **comm,
                        **freshness,
                    }
                )
    return rows


def run_agriculture_sweep(
    splits: list[TraceSplit],
    threshold_multipliers: list[float],
    loss_probabilities: list[float],
    beta_grid: list[float],
    seeds: list[int],
    freshness_deadline_multipliers: list[float],
    feedback_model: str = "ack",
    delivery_delay_minutes: float = 0.0,
    calibration_seed_count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    calibration_rows: list[dict[str, float | int | str]] = []
    calibration_seeds = seeds[: max(1, calibration_seed_count)]

    for split in splits:
        for multiplier in threshold_multipliers:
            for loss_probability in loss_probabilities:
                calibration = calibrate_split(
                    split,
                    threshold_multiplier=multiplier,
                    loss_probability=loss_probability,
                    beta_grid=beta_grid,
                    calibration_seeds=calibration_seeds,
                    feedback_model=feedback_model,
                    delivery_delay_minutes=delivery_delay_minutes,
                )
                calibration_rows.append(
                    {
                        "node_id": split.node_id,
                        "signal": split.signal,
                        "segment_id": split.segment_id,
                        "loss_probability": loss_probability,
                        **asdict(calibration),
                    }
                )
                rows.extend(
                    evaluate_split(
                        split,
                        calibration,
                        loss_probability,
                        seeds,
                        freshness_deadline_multipliers,
                        feedback_model,
                        delivery_delay_minutes,
                    )
                )

    return pd.DataFrame(rows), pd.DataFrame(calibration_rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    group_columns = [
        "signal",
        "threshold_multiplier",
        "loss_probability",
        "estimator",
        "freshness_deadline_minutes",
    ]
    summary = (
        results.groupby(group_columns, dropna=False)
        .agg(
            experiments=("mse", "size"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            mean_aoi_minutes=("mean_aoi_minutes", "mean"),
            peak_aoi_minutes=("peak_aoi_minutes", "mean"),
            freshness_violation_rate=("freshness_violation_rate", "mean"),
            attempted_update_rate=("attempted_update_rate_per_observation", "mean"),
            delivered_update_rate=("delivered_update_rate_per_observation", "mean"),
        )
        .reset_index()
    )
    summary["mse_ci95_halfwidth"] = 1.96 * summary["mse_std"] / np.sqrt(
        summary["experiments"]
    )
    return summary


def save_results(
    results: pd.DataFrame,
    calibration: pd.DataFrame,
    output_directory: str | Path,
) -> tuple[Path, Path, Path]:
    output = Path(output_directory)
    raw_dir = output / "raw"
    summary_dir = output / "summaries"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "agriculture_experiments.csv"
    calibration_path = raw_dir / "calibration_results.csv"
    summary_path = summary_dir / "agriculture_summary.csv"
    results.to_csv(raw_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    summarize_results(results).to_csv(summary_path, index=False)
    return raw_path, calibration_path, summary_path
