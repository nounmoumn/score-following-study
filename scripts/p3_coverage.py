import sys,json,shutil,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import read_csv,write_csv,load_case
from score_following.multicandidate import select_regions
from score_following.partition_lineage import partition_prior,advance
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import cosine_cost


def main():
    sys.stdout.reconfigure(encoding='utf-8');base=ROOT/'results/p3_onset_development';source=ROOT/'results/p3_temporal_20260918_000058';out=base/'coverage';out.mkdir(exist_ok=True);(out/'source_snapshot').mkdir(exist_ok=True)
    events=read_csv(base/'new_failures.csv');cand=read_csv(base/'multicandidate/candidates.csv');cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes'];cases=read_csv(source/'manifest.csv')
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'));rows=[];residual=0.;odds_residual=0.
    with threadpool_limits(limits=1):
        for case in cases:
            ee=[r for r in events if r['case_id']==case['case_id']]
            if not ee:continue
            u,v,hv,time,truth=load_case(case,cfg,annot);cost=cosine_cost(u,v);priors={}
            starts={int(round(float(r['time_sec'])*10)) for r in ee};p=np.zeros(len(v));p[0]=1
            with np.load(source/'cases'/(case['case_id']+'.npz')) as saved:path=saved['intact_bayes']
            for i,row in enumerate(cost):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                if i in starts:priors[i]=p.copy()
                p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum();assert p.argmax()==path[i]
            for event in ee:
                start=int(round(float(event['time_sec'])*10));onset=int(event['failure_frame']);p=priors[start].copy()
                regions,mass,centers=select_regions(p,int(round(truth[start-1]))+1,max(1,int(round(.5/hv))),max(1,int(round(2/hv))))
                components=partition_prior(p,regions);origins=components.copy()
                for i in range(start,onset+1):
                    components=advance(components,cost[i],b['transition'],b['emission_beta'],b['restart_probability'],first=i==start)
                    origins=advance(origins,cost[i],b['transition'],b['emission_beta'],b['restart_probability'],first=i==start,origin=True)
                    if i>start:p=predict_prior(p,b['transition'],b['restart_probability'])
                    p*=np.exp(-b['emission_beta']*(cost[i]-cost[i].min()));p/=p.sum()
                    for values in [components,origins]:
                        residual=max(residual,float(np.max(abs(values.sum(axis=0)-p))));np.testing.assert_allclose(values.sum(axis=0),p,atol=1e-12)
                    winner=int(p.argmax());assert winner==path[i]
                    if i not in [start+19,onset]:continue
                    phase='exposure_end' if i==start+19 else 'onset'
                    expected=sorted([r for r in cand if r['case_id']==case['case_id'] and float(r['time_sec'])==float(event['time_sec'])],key=lambda r:int(r['prior_rank']))
                    logodds=np.log(origins[1:6].sum(axis=1)/origins[0].sum())
                    delta=float(np.max(abs(logodds-np.array([float(r['end_odds' if phase=='exposure_end' else 'later_odds']) for r in expected]))))
                    odds_residual=max(odds_residual,delta);assert delta<1e-8
                    shares=np.array([components[0,winner],components[1:6,winner].sum(),components[6,winner],components[7,winner]])/p[winner]
                    names=['correct_initial','selected_wrong','unselected','new_restart'];dominant=names[int(shares.argmax())]
                    item=dict(case_id=case['case_id'],composition=case['composition'],time_sec=float(event['time_sec']),phase=phase,frame=i,
                        winner_frame=winner,error_sec=float(abs(winner-truth[i])*hv),dominant=dominant,dominant_share=float(shares.max()),
                        origin_wrong_overtake=bool(logodds.max()>0),origin_wrong_max_odds=float(logodds.max()),
                        initial_selected_wrong_mass=float(mass[1:].sum()))
                    for name,value in zip(names,shares):item[name+'_winner_share']=float(value)
                    rows.append(item)
            print(case['case_id'],flush=True)
    write_csv(out/'events.csv',rows);onsets=[r for r in rows if r['phase']=='onset']
    metrics=dict(events=len(onsets),dominant_counts=dict(Counter(r['dominant'] for r in onsets)),
        selected_wrong_majority=sum(r['selected_wrong_winner_share']>.5 for r in onsets),unselected_majority=sum(r['unselected_winner_share']>.5 for r in onsets),
        new_restart_majority=sum(r['new_restart_winner_share']>.5 for r in onsets),correct_initial_majority=sum(r['correct_initial_winner_share']>.5 for r in onsets),
        selected_wrong_majority_without_origin_overtake=sum(r['selected_wrong_winner_share']>.5 and not r['origin_wrong_overtake'] for r in onsets),
        max_posterior_residual=residual,max_origin_odds_residual=odds_residual)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    files=[Path(__file__),ROOT/'src/score_following/partition_lineage.py',ROOT/'src/score_following/multicandidate.py',ROOT/'P3_COVERAGE_PROTOCOL.md',ROOT/'tests/test_partition_lineage.py',base/'new_failures.csv',base/'multicandidate/candidates.csv',source/'provenance.json']
    for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
    (out/'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
