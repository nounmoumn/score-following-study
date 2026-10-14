import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'scripts/p3_fhg_help_gradient_development.py';s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_fixed_gradient_family():
 assert m.LAGS==(.5,1.,2.,4.) and m.PERMUTATIONS==2000
 assert (m.WINDOW_SEC,m.HORIZON_SEC,m.STEP_SEC)==(12.,.5,.5)
def test_auc_direction():
 rows=[{'event':True,'deficit':2},{'event':False,'deficit':1}]
 assert m.auc(rows)==1.
