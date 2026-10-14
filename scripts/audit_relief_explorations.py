"""Independent pair reconstruction and provenance checks, both window conventions."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.contrast_probe import piano_roll,normalize
meta={r['case_id']:r for r in csv.DictReader((ROOT/'output/score_priority_plan/cohort_manifest.csv').open(encoding='utf-8-sig'))}
cache={};report={}
for folder,trailing in [('rival_relief_exploration',False),('past_rival_exploration',True)]:
 out=ROOT/'results'/folder;frozen=json.loads((out/'freeze.json').read_text())
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
 maximum=0.;checked=0;n=0
 for path in sorted(out.glob('c[0-9][0-9][0-9].json')):
  c=json.loads(path.read_text());score=meta[c['case']]['score']
  if score not in cache:
   notes=load_notes(ROOT/'asap_dataset'/score);duration=max(x[1] for x in notes)
   cache[score]=notes_to_chroma_stream(notes,duration,.1),normalize(piano_roll(notes,duration))
  chroma,pitch=cache[score]
  n+=len(c['rows'])
  # Deterministic first/middle/last; no outcome-based sampling.
  for row in [c['rows'][0],c['rows'][len(c['rows'])//2],c['rows'][-1]]:
   i=int(round(row['truth']/.1))-1-(19 if trailing else 0)
   vals=[]
   for j,saved in zip(row['rivals'],row['similarities']):
    actual=[]
    for feat in [chroma,pitch]:
     x=feat[i:i+20].flatten();y=feat[j:j+20].flatten()
     actual.append(float(np.sum(x*y)/(np.sqrt(np.sum(x*x))*np.sqrt(np.sum(y*y)))))
    maximum=max(maximum,float(np.max(np.abs(np.array(actual)-saved))))
    vals.append(actual[0]-actual[1]);checked+=1
   assert abs(row['R1']-vals[0])<1e-10 and abs(row['R5']-np.mean(vals))<1e-10
 assert maximum<1e-10
 report[folder]=dict(rows=n,reconstructed_pairs=checked,max_similarity_residual=maximum,hash_drift=0)
out=ROOT/'results/continuous_error_exploration'
frozen=json.loads((out/'freeze.json').read_text())
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen.items())
report['status']='PASS'
(ROOT/'results/relief_explorations_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
