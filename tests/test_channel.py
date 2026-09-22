import numpy as np

from aoi_project.channel import simulate_threshold_channel
from aoi_project.estimators import estimate_from_receptions


def test_zero_loss_delivers_every_attempt() -> None:
    times = np.arange(6, dtype=float)
    values = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    trace = simulate_threshold_channel(times, values, threshold=2, loss_probability=0, seed=1)
    assert trace.attempted.sum() == 2
    assert np.array_equal(trace.attempted, trace.delivered)
    assert np.allclose(trace.latest_value, [0, 0, 2, 2, 4, 4])


def test_estimators_use_same_reception_trace() -> None:
    times = np.arange(5, dtype=float)
    values = np.array([0, 0, 0, 0, 0], dtype=float)
    trace = simulate_threshold_channel(times, values, threshold=1, loss_probability=0, seed=1)
    hold = estimate_from_receptions(trace, "hold", drift=2)
    full = estimate_from_receptions(trace, "full_drift", drift=2)
    weighted = estimate_from_receptions(trace, "weighted_drift", drift=2, beta=0.5)
    assert np.allclose(hold, 0)
    assert np.allclose(full, [0, 2, 4, 6, 8])
    assert np.allclose(weighted, [0, 1, 2, 3, 4])


def test_no_ack_resets_reference_on_failed_attempt() -> None:
    times = np.arange(5, dtype=float)
    values = np.array([0, 2, 2, 4, 4], dtype=float)
    trace = simulate_threshold_channel(
        times,
        values,
        threshold=2,
        loss_probability=0.999999,
        seed=1,
        feedback_model="no_ack",
    )
    assert np.array_equal(trace.attempted, [False, True, False, True, False])

