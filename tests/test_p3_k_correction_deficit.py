import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p3_k_correction_deficit.py"
SPEC = importlib.util.spec_from_file_location("p3_k_correction_deficit", SCRIPT)
correction = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(correction)


def test_auc_direction_and_shift_preserve_values():
    rows = []
    for track in ("a", "b"):
        for index in range(5):
            rows.append({"track": track, "deficit": float(index), "event": index >= 3})
    assert correction.rank_auc_safe([3, 4], [0, 1, 2]) == 1.0
    rng = np.random.default_rng(1)
    shifted = correction.shifted_auc(rows, rng)
    assert 0 <= shifted <= 1

