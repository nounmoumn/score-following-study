import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def test_robust_velocity_tracks_posterior_mean():
    path=ROOT/"scripts/p3_fv_velocity_diagnostic.py";spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    trace=np.eye(8)
    assert module.robust_velocity(trace,0,8)==1.0
