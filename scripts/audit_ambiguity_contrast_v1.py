"""Independent regrouping and saved-output checks; no model rerun."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ambiguity_contrast_v1'


def main():
    result=json.loads((OUT/'result.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted(OUT.glob('c[0-9][0-9][0-9].json'))]
    assert len(records)==40
    works=sorted({r['work'] for r in records})
    rows=[row for r in records for row in r['rows']]
    assert len(works)==23 and len(rows)==3146
    for record in records:
        old=json.loads((ROOT/'results/ambiguity_rival_support'/(record['case']+'.json')).read_text())
        assert len(old)==len(record['rows'])
        for before,row in zip(old,record['rows']):
            assert (before['time'],before['truth'],before['correct'])==(row['time'],row['truth'],row['base'])
            candidates=set([before['map_state']]+before['candidates']['score']+before['candidates']['posterior'])
            assert len(row['conditions'])==16
            for k,j in row['predictions'].items():
                assert j in candidates
                assert row['conditions'][k]==(abs((j+1)*.1-row['truth'])<=.5+1e-12)
    for k,v in result['accuracy_equal_work'].items():
        get=lambda r:r[k] if k in ('base','pitch_full') else r['conditions'][k]
        got=np.mean([np.mean([np.mean([get(row) for row in r['rows']]) for r in records if r['work']==w]) for w in works])
        assert abs(got-v)<1e-12
    for p,h in json.loads((OUT/'freeze.json').read_text()).items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    report={'status':'PASS','cases':40,'works':23,'rows':3146,'conditions_per_row':16,
            'saved_predictions_match_candidates':True,'hash_drift':[]}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
