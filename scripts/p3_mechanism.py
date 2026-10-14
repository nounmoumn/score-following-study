"""Previously used data: measurement gates and paired factorial intervention."""
import sys,json,hashlib,shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import load_case,read_csv,write_csv,pair_summary
from score_following.observation_evidence import predict_prior
from score_following.frozen_evidence import freeze_regions,fixed_path_evidence
from score_following.path_evidence import path_evidence
from score_following.temporal import bayes_instrumented
from score_following.p3_extension import cosine_cost,holm_adjust


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    source=ROOT/'results/p3_temporal_20260918_000058';cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'))
    cfg.update(seed=20260921,n_bootstrap=10000,n_sign_flips=19999)
    cases=read_csv(source/'manifest.csv');annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'))
    out=ROOT/'results'/f'p3_mechanism_{datetime.now():%Y%m%d_%H%M%S}';out.mkdir();(out/'cases').mkdir();(out/'source_snapshot').mkdir()
    write_csv(out/'manifest.csv',cases);(out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    files=[Path(__file__),ROOT/'P3_MECHANISM_PROTOCOL.md',ROOT/'scripts/p3_temporal.py']+list((ROOT/'src/score_following').glob('*.py'))
    for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
    inputs=[ROOT/cfg['asap_root']/c[k] for c in cases for k in ['score','performance']]+[source/'manifest.csv',ROOT/cfg['asap_root']/'asap_annotations.json']
    (out/'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files+inputs},indent=2),encoding='utf-8')
    rows=[];evidence=[];effects=[];b=cfg['bayes'];print(out,flush=True)
    with threadpool_limits(limits=1):
        for k,c in enumerate(cases):
            u,v,hv,time,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v);erased=cost.copy();erased[300:400]=cost[300:400].mean(axis=1,keepdims=True)
            traces={};lookup={}
            for shift in [0,-5,5]:
                for erase,obs in [(False,cost),(True,erased)]:
                    pos,_,_=bayes_instrumented(obs,truth,max(1,int(round(.5/hv))),onset=300,shift=shift,transition=b['transition'],beta=b['emission_beta'],restart=b['restart_probability'])
                    if shift==0 and not erase:
                        baseline=pos.copy()
                        with np.load(source/'cases'/(c['case_id']+'.npz')) as saved:np.testing.assert_array_equal(pos,saved['intact_bayes'])
                    np.testing.assert_array_equal(pos[:300],baseline[:300])
                    err=abs(pos-truth)*hv
                    r=dict(**c,shift_frames=shift,shift_score_sec=shift*hv,erased=erase,mean_error_during_sec=float(err[300:400].mean()),error5_sec=float(err[349]),mean_error_recovery_sec=float(err[400:500].mean()))
                    rows.append(r);lookup[shift,erase]=r;traces[f'path_{shift}_{erase}']=pos
            eff={s:lookup[s,True]['mean_error_during_sec']-lookup[s,False]['mean_error_during_sec'] for s in [0,-5,5]}
            effects.append(dict(**c,erasure=eff[0],interaction=(eff[-5]+eff[5])/2-eff[0],negative_shift_erasure=eff[-5],positive_shift_erasure=eff[5]))
            p=np.zeros(len(v));p[0]=1
            for i,row in enumerate(cost[:300]):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum()
            prior=predict_prior(p,b['transition'],b['restart_probability'])
            regions,mass,wrong=freeze_regions(prior,int(round(truth[299]))+1,max(1,int(round(.5/hv))),max(1,int(round(2/hv))))
            null=path_evidence(erased[300:320],regions,b['transition'],b['emission_beta'],b['restart_probability'])
            assert abs(null)<1e-10
            evidence.append(dict(**c,correct_mass=mass[0],wrong_mass=mass[1],wrong_center=wrong,fixed_W=fixed_path_evidence(cost[300:320],regions,b['emission_beta']),path_W=path_evidence(cost[300:320],regions,b['transition'],b['emission_beta'],b['restart_probability']),erased_path_W=null))
            np.savez_compressed(out/'cases'/(c['case_id']+'.npz'),truth_frames=truth,hop_score_sec=hv,**traces)
            print(f'{k+1}/40 {c["composition"]}',flush=True)
    write_csv(out/'conditions.csv',rows);write_csv(out/'effects.csv',effects);write_csv(out/'evidence.csv',evidence)
    tests=[dict(name=key,**pair_summary([r[key] for r in effects],cfg)) for key in ['erasure','interaction']]
    for t,p in zip(tests,holm_adjust([t['p'] for t in tests])):t.update(holm_p=float(p),passed=bool(t['ci95'][0]>0 and p<=.05))
    metrics=dict(primary=tests,null_max_abs=max(abs(r['erased_path_W']) for r in evidence),conditions=len(rows),scores=len(cases),fixed_path_median=float(np.median([r['fixed_W'] for r in evidence])),all_path_median=float(np.median([r['path_W'] for r in evidence])),sign_disagreements=sum((r['fixed_W']>0)!=(r['path_W']>0) for r in evidence))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
