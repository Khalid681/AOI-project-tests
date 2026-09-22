from __future__ import annotations

import numpy as np

from .channel import ReceptionTrace


def _safe_duration(times: np.ndarray) -> float:
    duration = float(times[-1] - times[0])
    if duration <= 0:
        raise ValueError("Experiment duration must be positive")
    return duration


def estimation_metrics(actual: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if actual.shape != estimate.shape:
        raise ValueError("actual and estimate must have identical shapes")
    error = actual - estimate
    mse = float(np.mean(error**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(estimate - actual)),
    }


def time_weighted_mean(times: np.ndarray, values: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(times) != len(values):
        raise ValueError("times and values must have equal length")
    duration = _safe_duration(times)
    widths = np.diff(times)
    return float(np.sum(values[:-1] * widths) / duration)


def communication_metrics(trace: ReceptionTrace) -> dict[str, float | int]:
    duration = _safe_duration(trace.times)
    attempted = int(trace.attempted.sum())
    delivered = int(trace.delivered.sum())
    return {
        "attempted_packets": attempted,
        "delivered_packets": delivered,
        "attempted_update_rate_per_minute": attempted / duration,
        "delivered_update_rate_per_minute": delivered / duration,
        "attempted_update_rate_per_observation": attempted / (len(trace.times) - 1),
        "delivered_update_rate_per_observation": delivered / (len(trace.times) - 1),
        "packet_delivery_ratio": delivered / attempted if attempted else 1.0,
    }


def freshness_metrics(
    trace: ReceptionTrace,
    deadline_minutes: float,
) -> dict[str, float]:
    if deadline_minutes <= 0:
        raise ValueError("deadline_minutes must be positive")
    age = np.maximum(trace.times - trace.latest_generation_time, 0.0)
    return {
        "mean_aoi_minutes": time_weighted_mean(trace.times, age),
        "sample_mean_aoi_minutes": float(np.mean(age)),
        "peak_aoi_minutes": float(np.max(age)),
        "aoi_p95_minutes": float(np.percentile(age, 95)),
        "freshness_deadline_minutes": float(deadline_minutes),
        "freshness_violation_rate": time_weighted_mean(
            trace.times, (age > deadline_minutes).astype(float)
        ),
    }
