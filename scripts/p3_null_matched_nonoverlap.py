"""Post-primary implementation sensitivity: controls share no frames."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml
from p3_temporal import load_case
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/p3_onset_development/null_numeric';SOURCE=ROOT/'results/p3_temporal_20260918_000058'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def event(e,s,stop):return any(np.all(e[j:j+5]>.5) for j in range(s,stop))
def overlap(a,b):return max(a[0],b[0])<min(a[1],b[1])
def main():
    cfg=yaml.safe_load((SOURCE/'config.yaml').read_text(encoding='utf-8'));ap=ROOT/cfg['asap_root']/'asap_annotations.json';annot=json.loads(ap.read_text(encoding='utf-8'))
    cases={r['case_id']:r for r in read(SOURCE/'manifest.csv')};raw=read(BASE/'episodes.csv')
    targets=[r for r in raw if r['entry_healthy']=='True' and r['two_sec_followup_complete']=='True' and round(float(r['duration_sec'])*10)>=5]
    by=defaultdict(list)
    for r in targets:by[r['case_id']].append(r)
    rows=[];miss=[];hashes={}
    for cid,tt in sorted(by.items()):
        c=cases[cid];u,v,hv,_,truth=load_case(c,cfg,annot);pathfile=SOURCE/'cases'/(cid+'.npz')
        with np.load(pathfile) as z:path=z['intact_bayes']
        err=abs(path-truth)*hv;null=np.linalg.norm(u,axis=1)==0;used=[]
        for r in sorted(tt,key=lambda x:(-round(float(x['duration_sec'])*10),int(x['start_frame']))):
            s=int(r['start_frame']);length=round(float(r['duration_sec'])*10);pre=float(err[s-1]);prog=float((truth[s+length-1]-truth[s])*hv)
            choices=[]
            for q in range(20,len(err)-length-4):
                interval=(q,q+length)
                if np.any(null[q:q+length]) or np.any(err[q-20:q]>.5) or any(overlap(interval,z) for z in used):continue
                qprog=float((truth[q+length-1]-truth[q])*hv)
                cost=abs(pre-err[q-1])/.1+abs(prog-qprog)/.25+abs(s-q)/900
                choices.append((cost,q,qprog))
            if not choices:miss.append(r);continue
            cost,q,qprog=min(choices);used.append((q,q+length))
            rows.append(dict(case_id=cid,length_frames=length,target_start_sec=s*.1,control_start_sec=q*.1,match_cost=float(cost),
                target_pre_error=pre,control_pre_error=float(err[q-1]),target_gt_progress=prog,control_gt_progress=qprog,
                paired_error_change_difference=float((err[s+length-1]-pre)-(err[q+length-1]-err[q-1])),
                target_event=event(err,s,s+length),control_event=event(err,q,q+length),
                target_persists_after=bool(np.all(err[s+length:s+length+5]>.5)),control_persists_after=bool(np.all(err[q+length:q+length+5]>.5))))
        assert all(not overlap(a,b) for i,a in enumerate(used) for b in used[i+1:])
        hashes[str(pathfile.relative_to(ROOT))]=hashlib.sha256(pathfile.read_bytes()).hexdigest()
    cd=[]
    for cid in sorted({r['case_id'] for r in rows}):
        rr=[r for r in rows if r['case_id']==cid];cd.append(np.mean([r['paired_error_change_difference'] for r in rr]))
    out=dict(scope='Post-primary nonoverlap sensitivity, development only',eligible_targets=len(targets),matched=len(rows),unmatched=len(miss),
        cases=len(cd),target_events=int(sum(r['target_event'] for r in rows)),control_events=int(sum(r['control_event'] for r in rows)),
        target_persist_after=int(sum(r['target_persists_after'] for r in rows)),control_persist_after=int(sum(r['control_persists_after'] for r in rows)),
        median_paired_error_change_difference=float(np.median([r['paired_error_change_difference'] for r in rows])),
        positive_case_mean_differences=int(sum(x>0 for x in cd)),
        median_abs_pre_error_difference=float(np.median([abs(r['target_pre_error']-r['control_pre_error']) for r in rows])),
        median_abs_gt_progress_difference=float(np.median([abs(r['target_gt_progress']-r['control_gt_progress']) for r in rows])))
    with (BASE/'matched_nonoverlap_pairs.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out['provenance']={**hashes,**{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'P3_NULL_MATCHED_PROTOCOL.md',BASE/'episodes.csv')}}
    (BASE/'matched_nonoverlap_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k!='provenance'},indent=2))
if __name__=='__main__':main()
