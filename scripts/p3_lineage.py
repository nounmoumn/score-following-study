"""Post-outcome mechanism audit of existing development failures."""
import sys,json,shutil,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import read_csv,write_csv,load_case
from score_following.lineage import lineage_step,crossing_type
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import cosine_cost


def main():
    sys.stdout.reconfigure(encoding='utf-8');base=ROOT/'results/p3_onset_development';source=ROOT/'results/p3_temporal_20260918_000058'
    out=base/'lineage';out.mkdir(exist_ok=True);(out/'source_snapshot').mkdir(exist_ok=True)
    events=read_csv(base/'new_failures.csv');ablations={(r['case_id'],float(r['time_sec'])):r for r in read_csv(base/'restart_ablation.csv')}
    cases=read_csv(source/'manifest.csv');cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes']
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'));summaries=[];traces=[];max_residual=0.
    with threadpool_limits(limits=1):
        for c in cases:
            ee=[r for r in events if r['case_id']==c['case_id']]
            if not ee:continue
            u,v,hv,time,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v)
            with np.load(source/'cases'/(c['case_id']+'.npz')) as z:baseline=z['intact_bayes']
            queries={int(round(float(r['time_sec'])*10))+19 for r in ee};states={};p=np.zeros(len(v));p[0]=1
            for i,row in enumerate(cost):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum()
                assert int(p.argmax())==baseline[i]
                if i in queries:states[i]=p.copy()
            for r in ee:
                end=int(round(float(r['time_sec'])*10))+19;onset=int(r['failure_frame']);p=states[end].copy();old=p.copy();new=np.zeros_like(p)
                for i in range(end+1,onset+5):
                    row=cost[i];old,new,prior=lineage_step(old,new,row,b['transition'],b['emission_beta'],b['restart_probability'])
                    reference=predict_prior(p,b['transition'],b['restart_probability'])*np.exp(-b['emission_beta']*(row-row.min()));reference/=reference.sum()
                    p=old+new;res=float(np.max(abs(p-reference)));max_residual=max(max_residual,res)
                    np.testing.assert_allclose(p,reference,atol=1e-13,rtol=1e-10)
                    winner=int(p.argmax());assert winner==baseline[i]
                    center=int(np.clip(round(truth[i]),0,len(p)-1));radius=max(1,int(round(.5/hv)))
                    correct=float(p[max(0,center-radius):min(len(p),center+radius+1)].sum());share=float(new[winner]/p[winner])
                    traces.append(dict(case_id=c['case_id'],landmark_sec=r['time_sec'],frame=i,time_sec=i*.1,map_frame=winner,map_mass=float(p[winner]),new_share_at_MAP=share,
                        true_region_mass=correct,error_sec=float(abs(winner-truth[i])*hv)))
                    if i==onset:
                        floor=np.finfo(float).tiny
                        prior_odds=float(np.log(max(prior[winner],floor))-np.log(max(prior[center],floor)))
                        bf=float(-b['emission_beta']*(row[winner]-row[center]));post_odds=float(np.log(max(p[winner],floor))-np.log(max(p[center],floor)))
                        ab=ablations[c['case_id'],float(r['time_sec'])]
                        summaries.append(dict(case_id=c['case_id'],composition=c['composition'],landmark_sec=r['time_sec'],onset_sec=i*.1,
                            kind=crossing_type(baseline[i-1],winner,hv),map_step_sec=float((winner-baseline[i-1])*hv),error_sec=float(abs(winner-truth[i])*hv),
                            new_share_at_MAP=share,existing_share_at_MAP=1-share,true_region_mass=correct,
                            winner_point_prior_log_odds=prior_odds,winner_point_observation_log_BF=bf,winner_point_posterior_log_odds=post_odds,
                            odds_identity_residual=post_odds-prior_odds-bf,
                            restart_off_failure=ab['no_restart_failure'],original_weakness=float(r['weakness'])))
            print(c['case_id'],flush=True)
    write_csv(out/'events.csv',summaries);write_csv(out/'traces.csv',traces)
    metrics=dict(events=len(summaries),kinds=dict(Counter(r['kind'] for r in summaries)),max_posterior_residual=max_residual,
        max_odds_identity_residual=max(abs(r['odds_identity_residual']) for r in summaries),
        new_lineage_majority=sum(r['new_share_at_MAP']>.5 for r in summaries),existing_lineage_majority=sum(r['new_share_at_MAP']<=.5 for r in summaries),
        median_new_share=float(np.median([r['new_share_at_MAP'] for r in summaries])),groups={})
    for kind in metrics['kinds']:
        rr=[r for r in summaries if r['kind']==kind];metrics['groups'][kind]=dict(n=len(rr),restart_off_prevented=sum(r['restart_off_failure']=='False' for r in rr),new_majority=sum(r['new_share_at_MAP']>.5 for r in rr),median_new_share=float(np.median([r['new_share_at_MAP'] for r in rr])))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    files=[Path(__file__),ROOT/'src/score_following/lineage.py',ROOT/'src/score_following/observation_evidence.py',ROOT/'P3_LINEAGE_PROTOCOL.md',ROOT/'tests/test_lineage.py',base/'new_failures.csv',base/'restart_ablation.csv',source/'provenance.json']
    for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
    (out/'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
