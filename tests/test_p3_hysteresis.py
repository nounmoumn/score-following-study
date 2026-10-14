import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_hysteresis.py"
SPEC = importlib.util.spec_from_file_location("p3_hysteresis", SCRIPT)
hysteresis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hysteresis)


def synthetic_rows(seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for composition in range(20):
        for performance in range(2):
            for landmark in range(25):
                initial = rng.normal(scale=.2)
                slope = rng.normal(scale=.1)
                current_ambiguity = rng.normal()
                exposure = rng.normal()
                target = .6 * initial + .2 * slope + .3 * current_ambiguity + .8 * exposure + rng.normal(scale=.15)
                rows.append({"composition": f"c{composition:02d}",
                             "performance_id": f"c{composition:02d}p{performance}",
                             "initial_gap": initial, "initial_gap_slope": slope,
                             "current_ambiguity": current_ambiguity,
                             "recent_exposure": exposure, "current_gap": target})
    return rows


def test_hysteresis_model_recovers_added_information():
    result = hysteresis.evaluate(synthetic_rows())
    assert result["mae_relative_improvement"] > .5
    assert result["positive_coefficient_share"] == 1.0


def test_shift_preserves_each_performance_values():
    rows = synthetic_rows()
    shifted = hysteresis.shifted_exposure(rows, np.random.default_rng(1))
    for performance in sorted({r["performance_id"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["performance_id"] == performance]
        original = sorted(rows[i]["recent_exposure"] for i in indices)
        assert np.allclose(sorted(shifted[indices]), original)

