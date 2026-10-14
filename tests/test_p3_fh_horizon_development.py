import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'scripts/p3_fh_horizon_development.py';spec=importlib.util.spec_from_file_location(path.stem,path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_grid_is_fixed_and_finite():
    assert mod.WINDOWS==(1.,2.,4.)
    assert mod.HORIZONS==(1.,2.,4.,8.)
    assert mod.CAP==5.
    assert mod.higher([1,2,3],95)==3.

def test_median_state():
    assert mod.median_state(np.array([.1,.5,.4]))==1
