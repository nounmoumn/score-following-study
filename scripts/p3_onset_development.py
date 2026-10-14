"""Audit new failures on development data; no held-out outcomes or p-values."""
import sys,json,hashlib,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import load_case,read_csv,write_csv
from score_following.onset import onset_outcome,work_family
from score_following.frozen_evidence import freeze_regions
from score_following.path_evidence import path_evidence
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import cosine_cost


def main():
    sys.stdout.reconfigure(encoding='utf-8');source=ROOT/'results/p3_temporal_20260918_000058'
    out=ROOT/'results/p3_onset_development';out.mkdir(exist_ok=True);(out/'source_snapshot').mkdir(exist_ok=True)
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));cases=read_csv(source/'manifest.csv');b=cfg['bayes']
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'));rows=[]
    files=[Path(__file__),ROOT/'P3_ONSET_DEVELOPMENT_PROTOCOL.md',ROOT/'src/score_following/onset.py',ROOT/'src/score_following/path_evidence.py',ROOT/'src/score_following/frozen_evidence.py',source/'manifest.csv',source/'config.yaml']
    for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
    (out/'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
    with threadpool_limits(limits=1):
        for k,c in enumerate(cases):
            u,v,hv,time,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v)
            with np.load(source/'cases'/(c['case_id']+'.npz')) as z:path=z['intact_bayes'];error=abs(path-truth)*hv
            p=np.zeros(len(v));p[0]=1;replay=[]
            for i,row in enumerate(cost):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                if i in range(200,751,50):
                    outcome=onset_outcome(error,i)
                    regions,mass,wrong=freeze_regions(p,int(round(truth[i-1]))+1,max(1,int(round(.5/hv))),max(1,int(round(2/hv))))
                    w=path_evidence(cost[i:i+20],regions,b['transition'],b['emission_beta'],b['restart_probability'])
                    rows.append(dict(**c,work_family=work_family(str(Path(c['score']).parent).replace('\\','/')),time_sec=float(time[i]),
                        **outcome,weakness=w,risk=mass[1]/sum(mass),pre_error_sec=float(error[i-1]),end_error_sec=float(error[i+19]),
                        future_error5_sec=float(error[i+69]),wrong_center=wrong,correct_mass=mass[0],wrong_mass=mass[1]))
                p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum();replay.append(p.argmax())
            np.testing.assert_array_equal(replay,path);write_csv(out/'landmarks.csv',rows)
            print(f'{k+1}/40 {c["composition"]}',flush=True)
    failures=[r for r in rows if r['status']=='new_failure'];summary=dict(status_counts=dict(Counter(r['status'] for r in rows)),
        unique_onsets=len({(r['case_id'],r['failure_frame']) for r in failures}),failure_scores=len({r['score'] for r in failures}),
        failure_families=len({r['work_family'] for r in failures}),groups={})
    for status in sorted({r['status'] for r in rows}):
        rr=[r for r in rows if r['status']==status]
        summary['groups'][status]=dict(n=len(rr),median_W=float(np.median([r['weakness'] for r in rr])),W_positive=sum(r['weakness']>0 for r in rr),risk_ge_1percent=sum(r['risk']>=.01 for r in rr))
    write_csv(out/'new_failures.csv',failures)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
