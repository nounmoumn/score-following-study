import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def module(name):
    path=ROOT/"scripts"/name;spec=importlib.util.spec_from_file_location(path.stem,path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded

def test_vienna_match_parser_is_monotone():
    m=module("p3_vienna_forecast_preflight.py")
    score,performance=m.load_clock(ROOT/"data/vienna4x22_rematched/match/Chopin_op10_no3_p01.match")
    assert len(score)>100
    assert np.all(np.diff(score)>0)
    assert np.all(np.diff(performance)>0)

def test_predictive_quantiles_and_conservative_percentile():
    m=module("p3_vienna_forecast_validation.py")
    p=np.array([.1,.2,.4,.2,.1])
    assert m.quantile_state(p,.005)==0
    assert m.quantile_state(p,.5)==2
    assert m.quantile_state(p,.995)==4
    assert m.higher_percentile([0,1,2,3],75)==3
