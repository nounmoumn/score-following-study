"""Fill missing pre-onset evidence for two existing local-compatible development cases."""
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import load_case
from score_following.p3_extension import cosine_cost
from score_following.observation_evidence import predict_prior
from score_following.current_evidence import current_evidence
BASE=ROOT/'results/p3_onset_development';SOURCE=ROOT/'results/p3_temporal_20260918_000058'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    route=BASE/'claim_route/summary.json'
    candidates=json.loads(route.read_text())['post_exposure']['local_compatible']
    assert [(r['case_id'],r['onset_frame']) for r in candidates]==[('n010',534),('n019',540)]
    manifest={r['case_id']:r for r in read(SOURCE/'manifest.csv')}
    cfg=yaml.safe_load((SOURCE/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes']
    ap=ROOT/cfg['asap_root']/'asap_annotations.json';annot=json.loads(ap.read_text(encoding='utf-8'))
    rows=[];summaries=[]
    with threadpool_limits(limits=1):
        for candidate in candidates:
            c=manifest[candidate['case_id']];j=candidate['onset_frame']
            u,v,hv,time,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v);ix=np.arange(len(v))
            with np.load(SOURCE/'cases'/(c['case_id']+'.npz')) as z:original=z['intact_bayes']
            p=np.zeros(len(v));p[0]=1;local=[]
            for i,row in enumerate(cost):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                if j-20<=i<j+5:
                    p,m=current_evidence(p,row,abs(ix-truth[i])*hv<=.5,b['emission_beta'])
                    trace=dict(case_id=c['case_id'],frame=i,available_sec=(i+1)*.1,
                        onset_sec=j*.1,strictly_available_before_onset=(i+1)<j,
                        coincides_with_onset_boundary=(i+1)==j,
                        observation_norm=float(np.linalg.norm(u[i])),cost_range=float(np.ptp(row)),**m)
                    rows.append(trace);local.append(trace)
                else:p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum()
                assert int(np.argmax(p))==original[i],(c['case_id'],i)
            strict=[r for r in local if r['strictly_available_before_onset']]
            null_start=j
            while null_start>0 and np.ptp(cost[null_start-1])==0:null_start-=1
            summary=dict(case_id=c['case_id'],composition=c['composition'],onset_sec=j*.1,
                strict_pre_frames=len(strict),strict_B_positive=sum(r['B']>0 for r in strict),
                strict_B_mean=float(np.mean([r['B'] for r in strict])),strict_B_min=min(r['B'] for r in strict),strict_B_max=max(r['B'] for r in strict),
                boundary_B=next(r['B'] for r in local if r['coincides_with_onset_boundary']),
                at_onset_frame_B=next(r['B'] for r in local if r['frame']==j),
                strict_flat_cost_frames=sum(r['cost_range']==0 for r in strict),
                strict_zero_observation_frames=sum(r['observation_norm']==0 for r in strict),
                strict_B_above_numerical_1e_12=sum(r['B']>1e-12 for r in strict),
                preceding_exact_flat_run_start_sec=null_start*.1 if null_start<j else None,
                preceding_exact_flat_run_first_available_sec=(null_start+1)*.1 if null_start<j else None,
                max_identity_residual=max(abs(r['identity_residual']) for r in local),full_path_replay_equal=True)
            summaries.append(summary)
    out=BASE/'claim_route'
    with (out/'local_evidence.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(cases=summaries,scope='Retrospective selected cases, oracle GT regions; B>0 means wrong-region observation preference, not a validated loss-of-evidence onset.',
        provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),route,SOURCE/'config.yaml',SOURCE/'manifest.csv',ap,ROOT/'src/score_following/current_evidence.py')})
    (out/'local_evidence_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
