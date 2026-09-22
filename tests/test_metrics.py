import numpy as np

from aoi_project.channel import ReceptionTrace
from aoi_project.metrics import freshness_metrics, time_weighted_mean


def test_time_weighted_mean_with_irregular_times() -> None:
    times = np.array([0.0, 1.0, 4.0])
    values = np.array([2.0, 4.0, 99.0])
    assert time_weighted_mean(times, values) == 3.5


def test_freshness_metrics() -> None:
    trace = ReceptionTrace(
        times=np.array([0.0, 10.0, 20.0, 30.0]),
        values=np.zeros(4),
        latest_value=np.zeros(4),
        latest_generation_time=np.array([0.0, 0.0, 20.0, 20.0]),
        attempted=np.zeros(4, dtype=bool),
        delivered=np.zeros(4, dtype=bool),
    )
    metrics = freshness_metrics(trace, deadline_minutes=5)
    assert metrics["peak_aoi_minutes"] == 10
    assert np.isclose(metrics["mean_aoi_minutes"], 10 / 3)
    assert np.isclose(metrics["freshness_violation_rate"], 1 / 3)
