import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def test_median_state_uses_probability_not_mean():
    path=ROOT/"scripts/p3_fc_fresh_certificate.py";spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.median_state(np.array([.6,0,0,0,.4]))==0
