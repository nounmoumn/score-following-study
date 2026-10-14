import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'scripts/p3_fhvb6_final_validation.py';s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_final_signal_and_multiplicity_are_frozen():
 assert (m.LAG_SEC,m.HORIZON_SEC,m.WINDOW_SEC,m.STEP_SEC)==(.5,.5,12.,.5)
 assert m.PERMUTATIONS==2000 and m.RADIUS==.5211881394162992
