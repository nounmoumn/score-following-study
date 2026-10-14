"""Frozen within-case matching of natural exact-null episodes."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from p3_temporal import load_case
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/p3_onset_development/null_numeric'
SOURCE=ROOT/'results/p3_temporal_20260918_000058'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def event(error,start,stop):
    x=[j for j in range(start,stop) if len(error[j:j+5])==5 and np.all(error[j:j+5]>.5)]
    return x[0] if x else None
def main():
    cfg=yaml.safe_load((SOURCE/'config.yaml').read_text(encoding='utf-8'))
    ap=ROOT/cfg['asap_root']/'asap_annotations.json';annot=json.loads(ap.read_text(encoding='utf-8'))
    cases={r['case_id']:r for r in read(SOURCE/'manifest.csv')}
    raw=read(BASE/'episodes.csv')
    targets=[r for r in raw if r['entry_healthy']=='True' and r['two_sec_followup_complete']=='True' and round(float(r['duration_sec'])*10)>=5]
    by=defaultdict(list)
    for r in targets:by[(r['case_id'],round(float(r['duration_sec'])*10))].append(r)
    pairs=[];unmatched=[];hashes={}
    for (case_id,length),tt in sorted(by.items()):
        c=cases[case_id];u,v,hv,time,truth=load_case(c,cfg,annot)
        pathfile=SOURCE/'cases'/(case_id+'.npz')
        with np.load(pathfile) as z:path=z['intact_bayes']
        error=abs(path-truth)*hv;null=np.linalg.norm(u,axis=1)==0
        candidates=[s for s in range(20,len(error)-length-4)
                    if np.all(error[s-20:s]<=.5) and not np.any(null[s:s+length])]
        # Candidate windows may overlap each other; start indices themselves cannot be reused.
        if not candidates:
            unmatched.extend(tt);continue
        cost=np.empty((len(tt),len(candidates)))
        tstats=[]
        for i,r in enumerate(tt):
            s=int(r['start_frame']);progress=float((truth[s+length-1]-truth[s])*hv);pre=float(error[s-1])
            tstats.append((s,pre,progress))
            for j,q in enumerate(candidates):
                qprogress=float((truth[q+length-1]-truth[q])*hv)
                cost[i,j]=abs(pre-error[q-1])/.1+abs(progress-qprogress)/.25+abs(s-q)/900
        ri,ci=linear_sum_assignment(cost)
        matched=set(ri)
        unmatched.extend(tt[i] for i in range(len(tt)) if i not in matched)
        for i,j in zip(ri,ci):
            r=tt[int(i)];s,pre,progress=tstats[int(i)];q=candidates[int(j)]
            te=event(error,s,s+length);ce=event(error,q,q+length)
            pairs.append(dict(case_id=case_id,composition=c['composition'],length_frames=length,
                target_start_sec=s*.1,control_start_sec=q*.1,match_cost=float(cost[i,j]),
                target_pre_error=pre,control_pre_error=float(error[q-1]),
                target_gt_progress=progress,control_gt_progress=float((truth[q+length-1]-truth[q])*hv),
                target_error_change=float(error[s+length-1]-pre),control_error_change=float(error[q+length-1]-error[q-1]),
                paired_error_change_difference=float((error[s+length-1]-pre)-(error[q+length-1]-error[q-1])),
                target_event=te is not None,control_event=ce is not None,
                target_persists_after=bool(np.all(error[s+length:s+length+5]>.5)),
                control_persists_after=bool(np.all(error[q+length:q+length+5]>.5))))
        hashes[str(pathfile.relative_to(ROOT))]=hashlib.sha256(pathfile.read_bytes()).hexdigest()
    case_diffs=[]
    for cid in sorted({r['case_id'] for r in pairs}):
        rr=[r for r in pairs if r['case_id']==cid]
        case_diffs.append(dict(case_id=cid,n=len(rr),mean_difference=float(np.mean([r['paired_error_change_difference'] for r in rr]))))
    out=dict(scope='Frozen exploratory development comparison; not independent P3 validation.',
        eligible_targets=len(targets),matched=len(pairs),unmatched=len(unmatched),cases=len(case_diffs),
        median_target_pre_error=float(np.median([r['target_pre_error'] for r in pairs])),
        median_control_pre_error=float(np.median([r['control_pre_error'] for r in pairs])),
        median_abs_pre_error_difference=float(np.median([abs(r['target_pre_error']-r['control_pre_error']) for r in pairs])),
        median_abs_gt_progress_difference=float(np.median([abs(r['target_gt_progress']-r['control_gt_progress']) for r in pairs])),
        target_events=sum(r['target_event'] for r in pairs),control_events=sum(r['control_event'] for r in pairs),
        target_persist_after=sum(r['target_persists_after'] for r in pairs),control_persist_after=sum(r['control_persists_after'] for r in pairs),
        median_paired_error_change_difference=float(np.median([r['paired_error_change_difference'] for r in pairs])),
        positive_case_mean_differences=sum(r['mean_difference']>0 for r in case_diffs),case_differences=case_diffs,
        caveat='Matching excludes outcomes but candidate windows may overlap. Multiple episodes are correlated; no inferential p-value.')
    with (BASE/'matched_pairs.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    out['provenance']={**hashes,**{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'P3_NULL_MATCHED_PROTOCOL.md',BASE/'episodes.csv',SOURCE/'manifest.csv',SOURCE/'config.yaml',ap)}}
    (BASE/'matched_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='provenance'},indent=2))
if __name__=='__main__':main()
