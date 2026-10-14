"""Conditional monotone bounds between frozen anchors, not certified GT intervals."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/p3_external_metadata'

def bounds(times,positions,t):
    if t<times[0] or t>times[-1]: return None
    j=int(np.searchsorted(times,t))
    if j<len(times) and t==times[j]: return float(positions[j]),float(positions[j])
    return float(positions[j-1]),float(positions[j])

def main():
    assert bounds(np.array([0.,2.]),np.array([1.,4.]),1.)==(1.,4.)
    assert bounds(np.array([0.,2.]),np.array([1.,4.]),2.)==(4.,4.)
    assert bounds(np.array([0.,2.]),np.array([1.,4.]),-1.) is None
    source=BASE/'onset_sequence_comparison.json';data=json.loads(source.read_text())
    out={'assumption':'Both surrounding anchors correct, and true score position monotone. These are conditional bounds, not confidence intervals.',
         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':[]}
    for case in data['cases']:
        path=BASE/f"{case['name']}_sequence_mapping.npz"
        mapping=np.load(path);times=mapping['performance_onset_sec'];pos=mapping['mapped_score_onset_sec']
        assert np.all(np.diff(times)>0) and np.all(np.diff(pos)>0)
        rows=[]
        for row in case['beat_rows']:
            lo,hi=bounds(times,pos,row['performance_sec'])
            truth=row['true_score_sec']
            distance=max(lo-truth,truth-hi,0.)
            rows.append({**row,'lower':lo,'upper':hi,'width':hi-lo,
                         'distance_outside':distance,'inside_numeric_tolerance':distance<=1e-6})
        widths=np.array([r['width'] for r in rows])
        summary={'name':case['name'],'n':len(rows),
                 'inside':sum(r['inside_numeric_tolerance'] for r in rows),
                 'outside_gt_0_1':sum(r['distance_outside']>.1 for r in rows),
                 'outside_gt_0_5':sum(r['distance_outside']>.5 for r in rows),
                 'width_median':float(np.median(widths)),'width_p95':float(np.percentile(widths,95)),
                 'width_gt_1_score_sec':int(sum(widths>1)),
                 'remaining_large_errors':[r for r in rows if abs(r['sequence_error'])>.5],
                 'all_rows':rows,'mapping_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        out['cases'].append(summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('all_rows','mapping_sha256')},indent=2))
    out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (BASE/'anchor_interval_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__': main()
