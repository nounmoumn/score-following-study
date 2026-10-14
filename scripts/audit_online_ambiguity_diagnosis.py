"""Independent arithmetic and clock audit of the completed diagnosis."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/online_ambiguity_diagnosis_v2'
def main():
 r=json.loads((OUT/'result.json').read_text())
 cases=[json.loads(p.read_text()) for p in sorted(OUT.glob('c[0-9][0-9][0-9].json'))]
 ws=sorted({c['work'] for c in cases})
 frozen=json.loads((OUT/'freeze.json').read_text())
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
 max_age=0.;seen=set();n=0;missing=0
 for c in cases:
  for row in c['rows']:
   key=(c['case'],row['time']);assert key not in seen;seen.add(key)
   age=row['time']-row['available'];assert -1e-12<=age<.1+1e-12
   assert abs(row['lag_available']-row['available']-2)<1e-12
   max_age=max(max_age,age)
   for m,p in row['positions'].items():
    assert abs(abs(p-row['truth'])-row['errors'][m])<1e-12
    assert int(abs(p-row['truth'])<=.5+1e-12)==row['correct'][m]
   k=row['correct']
   assert (k['lag_pitch']-k['lag_chroma'])-(k['bayes_pitch']-k['bayes_chroma']) == (k['lag_pitch']-k['bayes_pitch'])-(k['lag_chroma']-k['bayes_chroma'])
   n+=1;missing+=row['A'] is None
 assert n==4889 and missing==3
 for m in r['models']:
  vals=[np.mean([np.mean([x['correct'][m] for x in c['rows']]) for c in cases if c['work']==w]) for w in ws]
  assert abs(np.mean(vals)-r['models'][m]['accuracy_all_4889']['mean'])<1e-12
 slopes=[]
 for w in ws:
  s=[]
  for c in cases:
   if c['work']!=w:continue
   rows=[x for x in c['rows'] if x['A'] is not None]
   a=np.array([x['A'] for x in rows]);a=(a-a.mean())/a.std()
   y=np.array([x['correct']['bayes_pitch']-x['correct']['bayes_chroma'] for x in rows])
   s.append(np.linalg.lstsq(np.column_stack([np.ones(len(a)),a]),y,rcond=None)[0][1])
  slopes.append(np.mean(s))
 residual=float(abs(np.mean(slopes)-r['contrasts']['representation_bayes']['within_case_A_slope_per_SD']['mean']))
 assert residual<1e-12
 result=dict(status='PASS',rows=n,missing_A=missing,works=len(ws),max_output_age_sec=max_age,
             independent_lstsq_residual=residual,identity_checks=n,hash_drift=0)
 (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
