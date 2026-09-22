import numpy as np

from aoi_project.baseline import (
    generate_random_walk,
    simulate_loss_free_hold,
    theoretical_symmetric_simple_walk,
)


def test_symmetric_baseline_matches_closed_form() -> None:
    states = generate_random_walk(600_000, p=0.5, q=0.5, seed=7)
    simulated = simulate_loss_free_hold(states, threshold=4)
    theoretical = theoretical_symmetric_simple_walk(4)
    assert abs(simulated["update_rate"] - theoretical["theoretical_update_rate"]) < 0.002
    assert abs(simulated["mean_aoi"] - theoretical["theoretical_mean_aoi"]) < 0.35
    assert abs(simulated["mse"] - theoretical["theoretical_mse"]) < 0.12


def test_random_walk_increments_are_valid() -> None:
    states = generate_random_walk(10_000, p=0.3, q=0.4, seed=11)
    assert set(np.unique(np.diff(states))).issubset({-1, 0, 1})

