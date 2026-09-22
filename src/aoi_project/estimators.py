from __future__ import annotations

import numpy as np

from .channel import ReceptionTrace


def estimate_from_receptions(
    trace: ReceptionTrace,
    estimator: str,
    drift: float = 0.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Reconstruct the source at observation times using a receiver estimator."""
    if estimator not in {"hold", "full_drift", "weighted_drift"}:
        raise ValueError(f"Unknown estimator: {estimator}")
    if not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")

    elapsed = np.maximum(trace.times - trace.latest_generation_time, 0.0)
    if estimator == "hold":
        weight = 0.0
    elif estimator == "full_drift":
        weight = 1.0
    else:
        weight = beta
    return trace.latest_value + weight * drift * elapsed

