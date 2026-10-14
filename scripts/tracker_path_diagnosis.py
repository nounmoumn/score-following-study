import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.contrast_probe import piano_roll,normalize
from score_following.p3_extension import cosine_cost
from score_following.tracker_path_diagnosis import path_features
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
 out=ROOT/'results/tracker_path_diagnosis';src=ROOT/'results/tracker_candidate_diagnosis'
 if out.exists():raise SystemExit('Refuse overwrite')
 paths=sorted(src.glob('c[0-9][0-9][0-9].json'))
 meta={r['case_id']:r for r in csv.DictReader((ROOT/'output/score_priority_plan/cohort_manifest.csv').open(encoding='utf-8-sig'))}
 files=paths+[Path(__file__),ROOT/'scripts/analyze_tracker_path_diagnosis.py',ROOT/'TRACKER_PATH_DIAGNOSIS_PROTOCOL.md',ROOT/'tests/test_tracker_path_diagnosis.py']+list((ROOT/'src/score_following').glob('*.py'))
 files+=[ROOT/'asap_dataset'/meta[p.stem][k] for p in paths for k in ['score','performance']]
 h={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 out.mkdir();write(out/'freeze.json',h);cache={};n=0
 with threadpool_limits(limits=2):
  for path in paths:
   old=json.loads(path.read_text());m=meta[path.stem]
   if m['score'] not in cache:
    sn=load_notes(ROOT/'asap_dataset'/m['score']);d=max(x[1] for x in sn)
    cache[m['score']]=notes_to_chroma_stream(sn,d,.1),normalize(piano_roll(sn,d))
   sc,sp=cache[m['score']];pn=load_notes(ROOT/'asap_dataset'/m['performance'])
   nf=int(np.floor(min(90.,max(x[1] for x in pn))/.1))
   cost=cosine_cost(notes_to_chroma_stream(pn,nf*.1,.1)[:nf],sc)
   candidates={r['frame']:r['candidate'] for r in old['rows']}
   f=path_features(cost,sc,sp,candidates);cut=min(420,nf-1)
   small=path_features(cost[:cut+1],sc,sp,{t:v for t,v in candidates.items() if t<=cut})
   assert all(f[t]==small[t] for t in small)
   rows=[]
   for r in old['rows']:
    v=f[r['frame']];assert np.isfinite(v['score']).all()
    rows.append({**r,'candidate':{**r['candidate'],'score':v['score']},'paths':v['paths']});n+=1
   write(out/path.name,{**old,'rows':rows});print(path.stem,flush=True)
 assert n==4889
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in h.items())
 write(out/'audit.json',dict(status='PASS',rows=n,real_prefix_cases=40,hash_drift=0))
if __name__=='__main__':main()
