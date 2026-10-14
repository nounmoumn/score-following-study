import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def module():
    path=ROOT/"scripts/p3_fu_vienna_development.py";spec=importlib.util.spec_from_file_location(path.stem,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded

def test_forecast_center_uses_recent_posterior_velocity():
    trace=np.eye(30)[5:25]
    center,velocity=module().forecast_center(trace,20,history=20,future=21)
    assert velocity==1
    assert center==45
