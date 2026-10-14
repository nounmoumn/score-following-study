"""Paired transition-hold intervention across all frozen eligible null episodes."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml
from p3_temporal import load_case
from score_following.p3_extension import cosine_cost
from score_following.observation_evidence import predict_prior
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/p3_onset_development/null_numeric';SOURCE=ROOT/'results/p3_temporal_20260918_000058'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def update(p,c,b):
    p=p*np.exp(-b['emission_beta']*(c-c.min()));return p/p.sum()
def main():
    cfg=yaml.safe_load((SOURCE/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes'];ap=ROOT/cfg['asap_root']/'asap_annotations.json';annot=json.loads(ap.read_text(encoding='utf-8'))
    cases={r['case_id']:r for r in read(SOURCE/'manifest.csv')};raw=read(BASE/'episodes.csv')
    targets=[r for r in raw if r['entry_healthy']=='True' and r['two_sec_followup_complete']=='True' and round(float(r['duration_sec'])*10)>=5]
    by=defaultdict(list)
    for r in targets:by[r['case_id']].append(r)
    rows=[];hashes={}
    for cid,tt in sorted(by.items()):
        c=cases[cid];u,v,hv,_,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v);pathfile=SOURCE/'cases'/(cid+'.npz')
        with np.load(pathfile) as z:saved=z['intact_bayes']
        p=np.zeros(len(v));p[0]=1;starts={int(r['start_frame']):r for r in tt};snap={}
        for i,cc in enumerate(cost):
            if i in starts:snap[i]=p.copy()
            if i:p=predict_prior(p,b['transition'],b['restart_probability'])
            p=update(p,cc,b)
            assert int(p.argmax())==saved[i]
        for s,r in starts.items():
            length=round(float(r['duration_sec'])*10);original=snap[s].copy();hold=snap[s].copy();opath=[];hpath=[]
            for i in range(s,s+length+5):
                if i<s+length:
                    original=predict_prior(original,b['transition'],b['restart_probability']);original=update(original,cost[i],b)
                    # Exact null: no transition and no likelihood change in hold.
                    assert np.ptp(cost[i])==0
                else:
                    original=predict_prior(original,b['transition'],b['restart_probability']);original=update(original,cost[i],b)
                    hold=predict_prior(hold,b['transition'],b['restart_probability']);hold=update(hold,cost[i],b)
                opath.append(int(original.argmax()));hpath.append(int(hold.argmax()))
            np.testing.assert_array_equal(opath,saved[s:s+length+5])
            oe=abs(np.asarray(opath)-truth[s:s+length+5])*hv;he=abs(np.asarray(hpath)-truth[s:s+length+5])*hv
            rows.append(dict(case_id=cid,composition=c['composition'],start_sec=s*.1,length_frames=length,
                original_end_error=float(oe[length-1]),hold_end_error=float(he[length-1]),hold_minus_original_end=float(he[length-1]-oe[length-1]),
                original_after5_error=float(oe[length+4]),hold_after5_error=float(he[length+4]),hold_minus_original_after5=float(he[length+4]-oe[length+4]),
                original_null_event=bool(any(np.all(oe[j:j+5]>.5) for j in range(length))),
                hold_null_event=bool(any(np.all(he[j:j+5]>.5) for j in range(length))),
                original_persist_after=bool(np.all(oe[length:length+5]>.5)),hold_persist_after=bool(np.all(he[length:length+5]>.5))))
        hashes[str(pathfile.relative_to(ROOT))]=hashlib.sha256(pathfile.read_bytes()).hexdigest()
    case_means=[]
    for cid in sorted({r['case_id'] for r in rows}):
        rr=[r for r in rows if r['case_id']==cid];case_means.append(dict(case_id=cid,n=len(rr),
            end=float(np.mean([r['hold_minus_original_end'] for r in rr])),after5=float(np.mean([r['hold_minus_original_after5'] for r in rr]))))
    out=dict(scope='Development paired algorithmic intervention, not natural P3 validation',episodes=len(rows),cases=len(case_means),
        median_hold_minus_original_end=float(np.median([r['hold_minus_original_end'] for r in rows])),
        median_hold_minus_original_after5=float(np.median([r['hold_minus_original_after5'] for r in rows])),
        improved_end=sum(r['hold_minus_original_end']<0 for r in rows),improved_after5=sum(r['hold_minus_original_after5']<0 for r in rows),
        original_null_events=sum(r['original_null_event'] for r in rows),hold_null_events=sum(r['hold_null_event'] for r in rows),
        original_persist_after=sum(r['original_persist_after'] for r in rows),hold_persist_after=sum(r['hold_persist_after'] for r in rows),
        positive_case_mean_end=sum(r['end']>0 for r in case_means),negative_case_mean_end=sum(r['end']<0 for r in case_means),case_means=case_means)
    with (BASE/'dynamics_pairs.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out['provenance']={**hashes,**{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'P3_NULL_DYNAMICS_PROTOCOL.md',BASE/'episodes.csv',SOURCE/'config.yaml',SOURCE/'manifest.csv',ap)}}
    (BASE/'dynamics_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='provenance'},indent=2))
if __name__=='__main__':main()
