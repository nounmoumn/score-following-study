import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def test_lipschitz_help_bound():
    path=ROOT/"scripts/p3_fb_forecast_bound.py";spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(3);z=rng.normal(size=20);z-=z.mean();states=np.arange(20)
    predicted,true,bound,violation=module.help_bound(z,states,8.2,11.7,.1)
    assert abs(predicted-true)<=bound+1e-12
    assert violation<=1e-12
