import numpy as np

from scripts import p3_lagged_accumulation as lagged


def synthetic_rows(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for composition in range(8):
        for index in range(30):
            current = rng.normal()
            slope = rng.normal()
            ambiguity = rng.normal()
            rows.append({"composition": f"c{composition}", "performance_id": f"p{composition}",
                         "current_gap": current, "past_gap_slope": slope, "ambiguity": ambiguity,
                         "future_gap": .7 * current + .3 * slope + .8 * ambiguity + rng.normal(scale=.15)})
    return rows


def test_p3_l_model_recovers_added_information():
    result = lagged.evaluate(synthetic_rows())
    assert result["mae_relative_improvement"] > .5
    assert result["positive_coefficient_share"] == 1.0


def test_shift_preserves_each_performance_values():
    rows = synthetic_rows()
    shifted = lagged.shifted_ambiguity(rows, np.random.default_rng(1))
    original = np.asarray([row["ambiguity"] for row in rows])
    for composition in range(8):
        index = [i for i, row in enumerate(rows) if row["performance_id"] == f"p{composition}"]
        assert np.allclose(np.sort(shifted[index]), np.sort(original[index]))
