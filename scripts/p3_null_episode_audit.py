"""Exhaustive exact-null episode inventory, separate from fixed-landmark prediction."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import yaml
from p3_temporal import load_case
from score_following.p3_extension import cosine_cost
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/p3_onset_development';SOURCE=ROOT/'results/p3_temporal_20260918_000058'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
def main():
    assert runs([False,True,True,False,True])==[(1,3),(4,5)]
    cfg=yaml.safe_load((SOURCE/'config.yaml').read_text(encoding='utf-8'))
    ap=ROOT/cfg['asap_root']/'asap_annotations.json';annot=json.loads(ap.read_text(encoding='utf-8'))
    rows=[];file_hashes={}
    for c in read(SOURCE/'manifest.csv'):
        u,v,hv,t,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v)
        null=np.ptp(cost,axis=1)==0
        assert np.array_equal(null,np.linalg.norm(u,axis=1)==0)
        pathfile=SOURCE/'cases'/(c['case_id']+'.npz')
        with np.load(pathfile) as z:path=z['intact_bayes']
        error=abs(path-truth)*hv
        for start,end in runs(null):
            start,end=int(start),int(end)
            healthy=start>=20 and bool(np.all(error[start-20:start]<=.5))
            # Different descriptive endpoints retained separately, never select by success.
            complete=start+24<=len(error)
            future=[j for j in range(start,start+20) if np.all(error[j:j+5]>.5)] if complete else []
            during=[j for j in range(start,min(end,len(error)-4)) if np.all(error[j:j+5]>.5)]
            rows.append(dict(case_id=c['case_id'],composition=c['composition'],start_frame=int(start),end_frame_exclusive=int(end),
                start_sec=start*.1,first_available_sec=(start+1)*.1,end_sec=end*.1,duration_sec=(end-start)*.1,
                left_censored=start==0,right_censored=end==len(error),entry_history_complete=start>=20,
                entry_healthy=healthy,two_sec_followup_complete=complete,
                future2_onset=None if not future else future[0]*.1,
                during_null_onset=None if not during else during[0]*.1,
                error_before=None if start==0 else float(error[start-1]),error_at_end=float(error[end-1]),
                error_first_nonnull=None if end==len(error) else float(error[end]),
                first_nonnull_cost_flat=None if end==len(error) else bool(null[end]),
                error_five_frames_after=None if end+4>=len(error) else float(error[end+4]),
                sustained_after_return=None if end+5>len(error) else bool(np.all(error[end:end+5]>.5)),
                nonlocal_during=bool(np.any((np.diff(path[max(0,start-1):end])<0)|(np.diff(path[max(0,start-1):end])>3)))))
        file_hashes[str(pathfile.relative_to(ROOT))]=hashlib.sha256(pathfile.read_bytes()).hexdigest()
    eligible=[r for r in rows if r['entry_healthy'] and r['two_sec_followup_complete']]
    summary=dict(scope='All exact-null episodes in original40development performances. Episode-selected diagnostic; no causal comparison, no new predictor gate.',
        total_episodes=len(rows),healthy_complete_episodes=len(eligible),
        healthy_future2_events=sum(r['future2_onset'] is not None for r in eligible),
        healthy_onsets_during_null=sum(r['during_null_onset'] is not None for r in eligible),
        healthy_during_null_event_rows=[r for r in eligible if r['during_null_onset'] is not None],
        left_censored=sum(r['left_censored'] for r in rows),right_censored=sum(r['right_censored'] for r in rows))
    out=BASE/'null_numeric'
    with (out/'episodes.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    file_hashes.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),SOURCE/'manifest.csv',SOURCE/'config.yaml',ap)})
    summary['provenance']=file_hashes
    (out/'episode_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='provenance'},indent=2))
if __name__=='__main__':main()
