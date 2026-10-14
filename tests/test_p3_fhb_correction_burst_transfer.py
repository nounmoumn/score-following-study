import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'scripts/p3_fhb_correction_burst_transfer.py';s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_frozen_burst_is_single_direction_single_lag():
 assert (m.LAG_SEC,m.HORIZON_SEC,m.WINDOW_SEC,m.STEP_SEC)==(.5,.5,12.,.5)
 assert m.PERMUTATIONS==2000 and m.RADIUS==.5211881394162992
def test_fast_auc_matches_pairwise_ties():
 assert m.fast_auc([2,1],[1,0])==.875
