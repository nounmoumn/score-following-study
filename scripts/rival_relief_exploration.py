"""Adaptive opened-data research; preserves all six exploratory comparisons."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.contrast_probe import piano_roll,normalize

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/'results/rival_relief_exploration'
 if out.exists():raise SystemExit('Refuse overwrite')
 src=ROOT/'results/online_ambiguity_diagnosis_v2'
 casefiles=sorted(src.glob('c[0-9][0-9][0-9].json'))
 cases=[json.loads(p.read_text()) for p in casefiles]
 manifest=ROOT/'output/score_priority_plan/cohort_manifest.csv'
 meta={r['case_id']:r for r in csv.DictReader(manifest.open(encoding='utf-8-sig'))}
 files=casefiles+[manifest,Path(__file__),ROOT/'RIVAL_RELIEF_EXPLORATION_PROTOCOL.md',ROOT/'scripts/analyze_rival_relief_exploration.py',ROOT/'src/score_following/features.py',ROOT/'src/score_following/contrast_probe.py']
 files+=list({ROOT/'asap_dataset'/meta[c['case']]['score'] for c in cases})
 hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
 out.mkdir();write(out/'freeze.json',hashes);write(out/'state.json',{'status':'RUNNING'})
 scorepaths=sorted({meta[c['case']]['score'] for c in cases});cache={};maxdiff=0
 with threadpool_limits(limits=2):
  for scorepath in scorepaths:
   notes=load_notes(ROOT/'asap_dataset'/scorepath);duration=max(n[1] for n in notes)
   chroma=notes_to_chroma_stream(notes,duration,.1);pitch=normalize(piano_roll(notes,duration))
   cf=np.array([chroma[i:i+20].ravel() for i in range(len(chroma)-20)])
   cf=normalize(cf)
   indices=sorted({int(round(r['truth']/.1))-1 for c in cases if meta[c['case']]['score']==scorepath for r in c['rows'] if r['A'] is not None})
   data={}
   for block in range(0,len(indices),128):
    ids=indices[block:block+128];sims=cf[ids]@cf.T
    for idx,vals in zip(ids,sims):
     vals[max(0,idx-20):idx+21]=-np.inf
     chosen=[];pairs=[];q=normalize(pitch[idx:idx+20].ravel())
     for _ in range(5):
      j=int(vals.argmax());assert np.isfinite(vals[j])
      s=float(vals[j]);p=float(q@normalize(pitch[j:j+20].ravel()))
      chosen.append(j);pairs.append([s,p]);vals[max(0,j-20):j+21]=-np.inf
     d=np.array(pairs);data[idx]=dict(rivals=chosen,similarities=pairs,R1=float(d[0,0]-d[0,1]),R5=float(np.mean(d[:,0]-d[:,1])))
   cache[scorepath]=data;print(scorepath,len(indices),flush=True)
  n=0
  for c in cases:
   rows=[]
   for r in c['rows']:
    if r['A'] is None:continue
    idx=int(round(r['truth']/.1))-1;v=cache[meta[c['case']]['score']][idx]
    diff=abs(v['similarities'][0][0]-r['A']);maxdiff=max(maxdiff,diff);assert diff<1e-10
    assert all(abs(idx-j)>20 for j in v['rivals'])
    assert all(abs(a-b)>20 for k,a in enumerate(v['rivals']) for b in v['rivals'][k+1:])
    rows.append({**r,**v});n+=1
   write(out/(c['case']+'.json'),{**c,'rows':rows})
 assert n==4886
 assert all(sha(ROOT/p)==h for p,h in hashes.items())
 write(out/'audit.json',dict(rows=n,max_A_difference=maxdiff,hash_drift=0,status='PASS'))
 write(out/'state.json',{'status':'FEATURES_COMPLETE'})
if __name__=='__main__':main()
