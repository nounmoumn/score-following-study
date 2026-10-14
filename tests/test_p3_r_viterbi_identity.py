import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_viterbi_identity_negative_result_is_preserved():
    result=json.loads((ROOT/'results/p3_r_viterbi_identity/result.json').read_text())
    assert result['events']==16 and result['affected_composers']==9
    assert result['actions']==12
    assert result['helpful_actions']==0 and result['harmful_actions']==8
    assert result['safety_actions']==117
    assert not result['gates']['recovery']
    assert not result['gates']['safety']
    assert not result['gates']['all']
    assert result['decision']=='stop_candidate_identity_family'

