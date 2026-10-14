import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_alpha_grid_is_fixed_and_bounded():
    path=ROOT/"scripts/p3_fu_blend_development.py";spec=importlib.util.spec_from_file_location(path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.ALPHAS==(0.,.25,.5,.75,1.)
