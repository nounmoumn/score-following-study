import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_invariant_trigger_transfers_but_policy_fails():
    result=json.loads((ROOT/'results/p3_r_invariant_transfer/result.json').read_text())
    assert result['coverage']=={'composers':30,'interventions':120}
    assert result['incidence']['events']==16 and result['incidence']['affected_composers']==9
    assert result['gates']['trigger'] and result['gates']['oracle']
    assert not result['gates']['incidence']
    assert not result['gates']['selector']
    assert not result['gates']['safety']
    assert not result['gates']['all']
    assert result['decision']=='stop_before_preserved_validation'

