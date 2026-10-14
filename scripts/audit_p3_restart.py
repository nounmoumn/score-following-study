"""Exploratory matched-state restart ablation for every clean exposure window."""
import sys,json
from pathlib import Path
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import read_csv,write_csv,load_case
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import cosine_cost


def first_failure(err):
    return next((j for j in range(len(err)-4) if np.all(err[j:j+5]>.5)),None)


def main():
    sys.stdout.reconfigure(encoding='utf-8');out=ROOT/'results/p3_onset_development';source=ROOT/'results/p3_temporal_20260918_000058'
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes'];landmarks=read_csv(out/'landmarks.csv');cases=read_csv(source/'manifest.csv')
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'));rows=[]
    with threadpool_limits(limits=1):
        for c in cases:
            queries={int(round(float(r['time_sec'])*10))+19:r for r in landmarks if r['case_id']==c['case_id'] and r['status'] in ['new_failure','no_failure']}
            if not queries:continue
            u,v,hv,time,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v)
            with np.load(source/'cases'/(c['case_id']+'.npz')) as z:baseline=z['intact_bayes']
            p=np.zeros(len(v));p[0]=1
            for i,row in enumerate(cost):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum()
                if i not in queries:continue
                original=queries[i];paths={}
                for restart in [b['restart_probability'],0.]:
                    q=p.copy();path=[]
                    for future in cost[i+1:i+51]:
                        q=predict_prior(q,b['transition'],restart);q*=np.exp(-b['emission_beta']*(future-future.min()));q/=q.sum();path.append(int(q.argmax()))
                    paths[restart]=np.array(path)
                np.testing.assert_array_equal(paths[b['restart_probability']],baseline[i+1:i+51])
                f=first_failure(abs(paths[b['restart_probability']]-truth[i+1:i+51])*hv)
                no=first_failure(abs(paths[0.]-truth[i+1:i+51])*hv)
                assert (f is not None)==(original['status']=='new_failure')
                rows.append(dict(case_id=c['case_id'],composition=c['composition'],time_sec=original['time_sec'],original_failure=f is not None,no_restart_failure=no is not None,
                    original_onset_frame=None if f is None else i+1+f,no_restart_onset_frame=None if no is None else i+1+no,
                    path_identical=bool(np.array_equal(paths[0.],paths[b['restart_probability']]))))
            print(c['case_id'],flush=True)
    write_csv(out/'restart_ablation.csv',rows)
    summary=dict(clean_windows=len(rows),original_failures=sum(r['original_failure'] for r in rows),no_restart_failures=sum(r['no_restart_failure'] for r in rows),
        prevented=sum(r['original_failure'] and not r['no_restart_failure'] for r in rows),induced=sum(not r['original_failure'] and r['no_restart_failure'] for r in rows),
        persisted=sum(r['original_failure'] and r['no_restart_failure'] for r in rows),identical_paths=sum(r['path_identical'] for r in rows),
        interpretation='Exploratory, all clean windows; restart removed only after exposure, existing remote posterior mass remains. No new P3 significance test.')
    (out/'restart_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
