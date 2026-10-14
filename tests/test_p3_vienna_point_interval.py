import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_higher_quantile_is_conservative():
    path=ROOT/"scripts/p3_vienna_point_interval_diagnostic.py";spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.higher([0,1,2,3],75)==3
