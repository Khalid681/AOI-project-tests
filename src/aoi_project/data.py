from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"time stamp", "Node_ID", "Temp", "Moist"}


@dataclass(frozen=True)
class TraceSplit:
    node_id: int
    signal: str
    segment_id: int
    nominal_interval_minutes: float
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_sensor_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the cleaned Manzano sensor trace."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["time stamp"], errors="raise")
    df["Node_ID"] = pd.to_numeric(df["Node_ID"], errors="raise").astype(int)
    for signal in ("Temp", "Moist"):
        df[signal] = pd.to_numeric(df[signal], errors="coerce")

    if df.duplicated(["Node_ID", "timestamp"]).any():
        raise ValueError("Duplicate timestamp records exist within at least one node")
    if ((df["Temp"] < -50) | (df["Temp"] > 80)).any():
        raise ValueError("Temperature values fall outside the configured plausibility range")
    if ((df["Moist"] < 0) | (df["Moist"] > 100)).any():
        raise ValueError("Moisture values must be between 0 and 100")

    return df.sort_values(["Node_ID", "timestamp"]).reset_index(drop=True)


def infer_nominal_interval_minutes(node_df: pd.DataFrame) -> float:
    """Infer a node's nominal sampling interval from its most frequent interval."""
    delta = node_df["timestamp"].sort_values().diff().dt.total_seconds().div(60)
    delta = delta[(delta > 0) & delta.notna()].round(3)
    if delta.empty:
        raise ValueError("At least two timestamps are needed to infer an interval")
    modes = delta.mode()
    return float(modes.iloc[0] if not modes.empty else delta.median())


def add_continuous_segments(
    node_df: pd.DataFrame,
    nominal_interval_minutes: float,
    gap_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Start a new segment when elapsed time exceeds gap_multiplier times nominal."""
    if gap_multiplier <= 1:
        raise ValueError("gap_multiplier must be greater than one")
    out = node_df.sort_values("timestamp").copy()
    elapsed = out["timestamp"].diff().dt.total_seconds().div(60)
    out["segment_id"] = (elapsed > gap_multiplier * nominal_interval_minutes).cumsum()
    return out


def chronological_split(
    segment_df: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("A non-empty test fraction is required")

    ordered = segment_df.sort_values("timestamp").reset_index(drop=True)
    n = len(ordered)
    i = max(1, int(np.floor(n * train_fraction)))
    j = max(i + 1, int(np.floor(n * (train_fraction + validation_fraction))))
    j = min(j, n - 1)
    return ordered.iloc[:i].copy(), ordered.iloc[i:j].copy(), ordered.iloc[j:].copy()


def build_trace_splits(
    df: pd.DataFrame,
    signals: list[str] | tuple[str, ...] = ("Temp", "Moist"),
    nodes: list[int] | None = None,
    gap_multiplier: float = 3.0,
    minimum_segment_records: int = 50,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> list[TraceSplit]:
    """Create leakage-safe chronological splits for each node, signal and segment."""
    selected_nodes = sorted(df["Node_ID"].unique()) if nodes is None else nodes
    output: list[TraceSplit] = []

    for node_id in selected_nodes:
        node_df = df[df["Node_ID"] == node_id].copy()
        if node_df.empty:
            continue
        nominal = infer_nominal_interval_minutes(node_df)
        node_df = add_continuous_segments(node_df, nominal, gap_multiplier)

        for segment_id, segment in node_df.groupby("segment_id", sort=True):
            if len(segment) < minimum_segment_records:
                continue
            for signal in signals:
                clean = segment[["timestamp", "Node_ID", signal]].dropna().copy()
                if len(clean) < minimum_segment_records:
                    continue
                clean = clean.rename(columns={signal: "value"})
                train, validation, test = chronological_split(
                    clean, train_fraction, validation_fraction
                )
                output.append(
                    TraceSplit(
                        node_id=int(node_id),
                        signal=signal,
                        segment_id=int(segment_id),
                        nominal_interval_minutes=nominal,
                        train=train,
                        validation=validation,
                        test=test,
                    )
                )
    return output


def trace_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return elapsed minutes from the first sample and numeric values."""
    ordered = frame.sort_values("timestamp")
    t = (ordered["timestamp"] - ordered["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 60
    x = ordered["value"].to_numpy(dtype=float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("Trace timestamps must be strictly increasing")
    return t.astype(float), x


def estimate_drift(frame: pd.DataFrame) -> float:
    """Estimate a training-only linear trend in signal units per minute.

    Adjacent agricultural readings are often quantized and unchanged, so the
    median adjacent slope collapses to zero. A centered least-squares slope
    captures the longer trend while beta calibration controls overprediction.
    """
    t, x = trace_arrays(frame)
    centered_t = t - np.mean(t)
    denominator = float(np.dot(centered_t, centered_t))
    if denominator <= 0:
        return 0.0
    return float(np.dot(centered_t, x - np.mean(x)) / denominator)


def increment_scale(frame: pd.DataFrame) -> float:
    """Robust scale of consecutive value changes for threshold construction."""
    _, x = trace_arrays(frame)
    changes = np.diff(x)
    median = np.median(changes)
    mad = np.median(np.abs(changes - median))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 1e-12:
        robust_sigma = float(np.std(changes, ddof=1)) if changes.size > 1 else 0.0
    return max(robust_sigma, 1e-6)
