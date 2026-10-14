"""Post-hoc bridge audit: isolate sample-range from online/full-hindsight changes."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/online_ambiguity_diagnosis_v2'
target=OUT/'matched_old_subset_audit.json'
if target.exists():raise SystemExit('Refuse overwrite')
values={};n=0;hashes={}
for p in sorted(OUT.glob('c[0-9][0-9][0-9].json')):
 oldp=ROOT/'results/ambiguity_contrast_v1'/p.name
 hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 hashes[str(oldp.relative_to(ROOT))]=hashlib.sha256(oldp.read_bytes()).hexdigest()
 c=json.loads(p.read_text());old=json.loads(oldp.read_text())
 lookup={round(r['time'],6):r for r in c['rows']}
 a=[];ys={k:[] for k in ['full_hindsight','online','lag2']}
 for r in old['rows']:
  s=lookup[round(r['time'],6)];assert abs(r['truth']-s['truth'])<1e-9
  assert s['A'] is not None
  a.append(s['A']);k=s['correct']
  ys['full_hindsight'].append(int(r['pitch_full'])-int(r['base']))
  ys['online'].append(k['bayes_pitch']-k['bayes_chroma'])
  ys['lag2'].append(k['lag_pitch']-k['lag_chroma']);n+=1
 x=np.array(a);x-=x.mean()
 for k,y in ys.items():
  y=np.array(y);v={'gain':float(y.mean()),'raw_slope':float(x@y/(x@x)),'SD_slope':float(np.mean(x/x.std()*y))}
  values.setdefault(c['work'],{}).setdefault(k,[]).append(v)
assert n==3146
sample=np.random.default_rng(20260923).integers(0,23,(5000,23))
result={'scope':'post-hoc sample-equivalence check, no policy tuning','rows':n,'works':23,'conditions':{}}
for k in ys:
 result['conditions'][k]={}
 for metric in ['gain','raw_slope','SD_slope']:
  vals=np.array([np.mean([v[metric] for v in w[k]]) for w in values.values()])
  result['conditions'][k][metric]={'mean':float(vals.mean()),'bootstrap95':np.quantile(vals[sample].mean(1),[.025,.975]).tolist()}
previous=json.loads((ROOT/'results/diagnostic_map_pilot/result.json').read_text())
assert abs(result['conditions']['full_hindsight']['raw_slope']['mean']-previous['A_within_case_gain_slope']['mean'])<1e-12
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
result['input_hashes']=hashes;result['audit']='PASS: complete 3146 time/truth/A join; previous slope reproduced; hash drift0'
target.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'},indent=2))
