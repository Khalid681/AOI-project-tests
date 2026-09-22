from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReceptionTrace:
    times: np.ndarray
    values: np.ndarray
    latest_value: np.ndarray
    latest_generation_time: np.ndarray
    attempted: np.ndarray
    delivered: np.ndarray


def simulate_threshold_channel(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
    loss_probability: float,
    seed: int,
    feedback_model: str = "ack",
    delivery_delay_minutes: float = 0.0,
) -> ReceptionTrace:
    """Simulate threshold triggering, packet erasure and receiver state.

    The first observation initializes sender and receiver without counting as an
    attempted packet. All estimators can consume the returned receiver trace,
    ensuring a paired comparison under identical deliveries.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(times) != len(values) or len(times) < 2:
        raise ValueError("times and values must have equal length of at least two")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if not 0 <= loss_probability < 1:
        raise ValueError("loss_probability must be in [0, 1)")
    if feedback_model not in {"ack", "no_ack"}:
        raise ValueError("feedback_model must be 'ack' or 'no_ack'")
    if delivery_delay_minutes < 0:
        raise ValueError("delivery_delay_minutes cannot be negative")

    rng = np.random.default_rng(seed)
    n = len(times)
    attempted = np.zeros(n, dtype=bool)
    delivered = np.zeros(n, dtype=bool)
    latest_value = np.empty(n, dtype=float)
    latest_generation = np.empty(n, dtype=float)

    sender_reference = float(values[0])
    receiver_value = float(values[0])
    receiver_generation = float(times[0])
    pending: list[tuple[float, float, float]] = []

    for i, (time, value) in enumerate(zip(times, values)):
        # Process packets whose reception and ACK occur by this observation time.
        due = [event for event in pending if event[0] <= time]
        pending = [event for event in pending if event[0] > time]
        for _, generation_time, packet_value in sorted(due, key=lambda e: e[0]):
            receiver_value = packet_value
            receiver_generation = generation_time
            if feedback_model == "ack":
                sender_reference = packet_value

        if i > 0 and abs(value - sender_reference) >= threshold:
            attempted[i] = True
            if feedback_model == "no_ack":
                sender_reference = float(value)

            if rng.random() >= loss_probability:
                delivered[i] = True
                receive_time = float(time + delivery_delay_minutes)
                if receive_time <= time:
                    receiver_value = float(value)
                    receiver_generation = float(time)
                    if feedback_model == "ack":
                        sender_reference = float(value)
                else:
                    pending.append((receive_time, float(time), float(value)))

        latest_value[i] = receiver_value
        latest_generation[i] = receiver_generation

    return ReceptionTrace(
        times=times,
        values=values,
        latest_value=latest_value,
        latest_generation_time=latest_generation,
        attempted=attempted,
        delivered=delivered,
    )


def packet_event_rows(trace: ReceptionTrace) -> list[dict[str, float | bool | int]]:
    return [
        {
            "sample_index": int(i),
            "time_minutes": float(trace.times[i]),
            "value": float(trace.values[i]),
            "attempted": bool(trace.attempted[i]),
            "delivered": bool(trace.delivered[i]),
        }
        for i in range(len(trace.times))
        if trace.attempted[i]
    ]

