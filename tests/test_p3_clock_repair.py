import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def module():
    path=ROOT/"scripts/p3_clock_repair_preflight.py";spec=importlib.util.spec_from_file_location(path.stem,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded

def test_pava_is_strict_and_minimally_repairs_inversion():
    fixed=module().pava([0.,2.,1.,3.])
    assert np.all(np.diff(fixed)>0)
    np.testing.assert_allclose(fixed[1:3],[1.5,1.5+1e-9],atol=2e-9)
