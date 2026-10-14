import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_detects_harmful_observation(monkeypatch):
    path = ROOT / "scripts/p3_cp_wtc_observation_harm_diagnostic.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    trace = np.array([[.9, .1], [.1, .9]])
    cost = np.array([[0., 1.], [1., 0.]])
    truth = np.array([0., 0.])
    monkeypatch.setattr(module.balance, "predictive_prior", lambda posterior, weights: posterior)
    result = module.diagnose(trace, cost, truth, 1., (1.,), 1, 2)
    assert result["observation_correction_sum"] < 0
    assert result["negative_correction_share"] == 1
    assert result["mean_truth_cost_rank"] == 1
