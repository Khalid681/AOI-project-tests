import pandas as pd

from aoi_project.data import add_continuous_segments, chronological_split


def test_gap_creates_new_segment() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2022-01-01 00:00", "2022-01-01 00:20", "2022-01-01 02:00"]
            ),
            "value": [1.0, 2.0, 3.0],
        }
    )
    result = add_continuous_segments(frame, nominal_interval_minutes=20, gap_multiplier=3)
    assert result["segment_id"].tolist() == [0, 0, 1]


def test_chronological_split_preserves_order() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2022-01-01", periods=10, freq="20min"), "value": range(10)}
    )
    train, validation, test = chronological_split(frame, 0.6, 0.2)
    assert train["value"].tolist() == list(range(6))
    assert validation["value"].tolist() == [6, 7]
    assert test["value"].tolist() == [8, 9]
