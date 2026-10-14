import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'scripts/p3_fhd_help_drop_development.py';s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_temporal_formula_parameters_are_frozen():
 assert (m.WINDOW_SEC,m.HORIZON_SEC,m.BASELINE_SEC,m.STEP_SEC,m.CAP)==(12.,.5,20.,.5,5.)
 assert m.RADIUS==.5211881394162992
