import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.contrast_probe import piano_roll,normalize
from score_following.p3_extension import cosine_cost
from score_following.tracker_candidate_diagnosis import trace
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
 out=ROOT/'results/tracker_candidate_diagnosis'
 if out.exists():raise SystemExit('Refuse overwrite')
 source=ROOT/'results/online_ambiguity_diagnosis_v2'
 meta={r['case_id']:r for r in csv.DictReader((ROOT/'output/score_priority_plan/cohort_manifest.csv').open(encoding='utf-8-sig'))}
 paths=sorted(source.glob('c[0-9][0-9][0-9].json'))
 files=paths+[Path(__file__),ROOT/'scripts/analyze_tracker_candidate_diagnosis.py',ROOT/'TRACKER_CANDIDATE_DIAGNOSIS_PROTOCOL.md',ROOT/'tests/test_tracker_candidate_diagnosis.py']+list((ROOT/'src/score_following').glob('*.py'))
 files+=[ROOT/'asap_dataset'/meta[p.stem][k] for p in paths for k in ['score','performance']]
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 out.mkdir();write(out/'freeze.json',hashes);write(out/'state.json',{'status':'RUNNING'})
 cache={};n=0;invalid=0
 with threadpool_limits(limits=2):
  for path in paths:
   old=json.loads(path.read_text());m=meta[path.stem]
   if m['score'] not in cache:
    sn=load_notes(ROOT/'asap_dataset'/m['score']);d=max(v[1] for v in sn)
    cache[m['score']]=(notes_to_chroma_stream(sn,d,.1),normalize(piano_roll(sn,d)))
   sc,sp=cache[m['score']]
   pn=load_notes(ROOT/'asap_dataset'/m['performance']);duration=min(90.,max(v[1] for v in pn));nf=int(np.floor(duration/.1))
   pc=notes_to_chroma_stream(pn,nf*.1,.1)[:nf];cost=cosine_cost(pc,sc)
   frames=[r['frame'] for r in old['rows']]
   maps,features=trace(cost,sc,sp,frames)
   cut=min(420,nf-1);pm,pf=trace(cost[:cut+1],sc,sp,[f for f in frames if f<=cut])
   np.testing.assert_array_equal(maps[:cut+1],pm)
   for f in pf:
    assert features[f]==pf[f]
   rows=[]
   for r in old['rows']:
    f=features[r['frame']]
    assert abs((f['map']+1)*.1-r['positions']['bayes_chroma'])<1e-10
    assert np.isfinite(f['base']+f['score']).all()
    rows.append({**r,'candidate':f});invalid+=not f['valid'];n+=1
   write(out/path.name,{**old,'rows':rows});print(path.stem,len(rows),flush=True)
 assert n==4889
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 write(out/'audit.json',{'status':'PASS','rows':n,'invalid_pair_kept_with_flag':invalid,'hash_drift':0,'real_prefix_cases':40})
 write(out/'state.json',{'status':'FEATURES_COMPLETE'})
if __name__=='__main__':main()
