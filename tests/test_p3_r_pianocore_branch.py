import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_pianocore_split_acquisition_and_branch_negative_result():
    manifest=json.loads((ROOT/'results/p3_pianocore_p3r_manifest.json').read_text())
    assert manifest['counts']['development']==30
    assert manifest['counts']['validation']==17
    assert manifest['counts']['composer_overlap']==0
    acquisition=json.loads((ROOT/'results/p3_pianocore_p3r_acquisition.json').read_text())
    assert acquisition['acquired']==30
    assert acquisition['preserved_validation_requested']==0
    invariant=json.loads((ROOT/'results/p3_r_invariant_transfer/result.json').read_text())
    assert invariant['gates']['trigger'] and invariant['gates']['oracle']
    assert not invariant['gates']['all']
    branch=json.loads((ROOT/'results/p3_r_branch_policy/result.json').read_text())
    assert branch['branch_records']==2400 and branch['beneficial_branches']==31
    assert branch['actions']==1 and branch['helpful_actions']==0 and branch['harmful_actions']==1
    assert not branch['gates']['all']
    preserved_ids={row['id'] for row in manifest['preserved_validation']}
    acquired_ids={row['id'] for row in acquisition['rows']}
    assert preserved_ids.isdisjoint(acquired_ids)

