import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_preoutage_anchor_negative_result_is_preserved():
    result=json.loads((ROOT/'results/p3_ra_preoutage_anchor/result.json').read_text())
    assert result['combined_incidence']=={'events':42,'affected_groups':21,'gate':True}
    for corpus in ('ASAP','PianoCoRe'):
        section=result['corpora'][corpus]
        assert section['gates']['no_intact_false_outage']
        assert not section['gates']['recovery']
        assert not section['gates']['anchor_recall']
    assert not result['all_gate_pass']
    assert result['decision']=='preoutage_anchor_not_supported'

