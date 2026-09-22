from __future__ import annotations

import numpy as np
import pandas as pd


def generate_random_walk(
    n_slots: int,
    p: float,
    q: float,
    seed: int,
    initial_state: int = 0,
) -> np.ndarray:
    if n_slots < 2:
        raise ValueError("n_slots must be at least two")
    if p <= 0 or q <= 0 or p + q > 1:
        raise ValueError("Require p > 0, q > 0 and p + q <= 1")
    rng = np.random.default_rng(seed)
    increments = rng.choice(
        np.array([1, -1, 0]), size=n_slots - 1, p=[p, q, 1 - p - q]
    )
    return np.concatenate([[initial_state], initial_state + np.cumsum(increments)])


def simulate_loss_free_hold(states: np.ndarray, threshold: int) -> dict[str, float]:
    """Base-paper threshold scheme with an error-free, one-slot channel.

    The receiver resets to the transmitted state at each threshold crossing.
    AoI follows the paper's discrete convention and resets to one slot.
    """
    states = np.asarray(states, dtype=float)
    if threshold < 1:
        raise ValueError("threshold must be at least one")

    reference = states[0]
    estimate = states[0]
    age = 1.0
    updates = 0
    squared_errors: list[float] = []
    ages: list[float] = []

    for state in states[1:]:
        if abs(state - reference) >= threshold:
            reference = state
            estimate = state
            age = 1.0
            updates += 1
        else:
            age += 1.0
        squared_errors.append(float((state - estimate) ** 2))
        ages.append(age)

    return {
        "update_rate": updates / (len(states) - 1),
        "mean_aoi": float(np.mean(ages)),
        "mse": float(np.mean(squared_errors)),
    }


def theoretical_symmetric_simple_walk(threshold: int) -> dict[str, float]:
    """Closed forms for p=q=0.5 used as a baseline unit test."""
    t = float(threshold)
    return {
        "theoretical_update_rate": 1.0 / t**2,
        "theoretical_mean_aoi": (5.0 * t**2 + 1.0) / 6.0,
        "theoretical_mse": (t**2 - 1.0) / 6.0,
    }


def run_baseline_sweep(
    thresholds: list[int],
    n_slots: int = 1_000_000,
    seed: int = 2026,
) -> pd.DataFrame:
    states = generate_random_walk(n_slots=n_slots, p=0.5, q=0.5, seed=seed)
    rows = []
    for threshold in thresholds:
        simulated = simulate_loss_free_hold(states, threshold)
        theoretical = theoretical_symmetric_simple_walk(threshold)
        row = {"threshold": threshold, **simulated, **theoretical}
        row["update_rate_abs_error"] = abs(
            row["update_rate"] - row["theoretical_update_rate"]
        )
        row["mean_aoi_abs_error"] = abs(
            row["mean_aoi"] - row["theoretical_mean_aoi"]
        )
        row["mse_abs_error"] = abs(row["mse"] - row["theoretical_mse"])
        rows.append(row)
    return pd.DataFrame(rows)

