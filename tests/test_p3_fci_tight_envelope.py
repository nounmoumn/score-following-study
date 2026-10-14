import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def test_piecewise_envelope_matches_dense_grid():
    path=ROOT/"scripts/p3_fci_tight_envelope_diagnostic.py";spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(8);z=rng.normal(size=30);z-=z.mean();lo,hi=module.help_envelope(z,14.,.8,.1,cap=.5)
    states=np.arange(30);grid=np.linspace(6,22,20001);values=np.array([-np.minimum(np.abs(states-g)*.1,.5)@z for g in grid])
    assert abs(lo-values.min())<1e-8
    assert abs(hi-values.max())<1e-8
