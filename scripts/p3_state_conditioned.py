"""Frozen, score-held-out state/observation prediction experiment."""
import json,sys,shutil,hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import load_case,read_csv,write_csv,pair_summary
from score_following.features import load_notes
from score_following.p3_extension import cosine_cost,bayes_from_cost,holm_adjust
from score_following.observation_evidence import predict_prior
from score_following.frozen_evidence import freeze_regions,fixed_path_evidence
from score_following.prospective import ridge_predict


def select(cfg,annot):
    old=[r for f in cfg['excluded_manifests'] for r in read_csv(ROOT/f)]
    scores={r['score'] for r in old};names={r['composition'] for r in old}
    buckets=defaultdict(list)
    for r in read_csv(ROOT/cfg['asap_root']/'metadata.csv'):buckets[r['composer']].append(r)
    ordered=[b[i] for i in range(max(map(len,buckets.values()))) for b in buckets.values() if i<len(b)]
    selected=[];excluded=[];root=ROOT/cfg['asap_root']
    for r in ordered:
        label=r['composer']+'/'+r['title']
        if label in names or r['midi_score'] in scores:continue
        a=annot.get(r['midi_performance'],{});beats=a.get('performance_beats',[])
        reason=None
        if not a.get('score_and_performance_aligned'):reason='not_aligned'
        elif not beats or min(beats)>15 or max(beats)<90:reason='annotation_coverage'
        elif not all((root/r[k]).exists() for k in ['midi_performance','midi_score']):reason='missing_midi'
        else:
            notes=load_notes(root/r['midi_performance'])
            if not notes or max(n[1] for n in notes)<90:reason='short_performance'
        if reason:
            excluded.append(dict(composition=label,performance=r['midi_performance'],reason=reason));continue
        selected.append(dict(case_id=f'h{len(selected):03}',composition=label,performance=r['midi_performance'],score=r['midi_score']))
        names.add(label);scores.add(r['midi_score'])
        if len(selected)==cfg['new_scores']:break
    if len(selected)!=cfg['new_scores']:raise ValueError(f'Only {len(selected)} unused eligible scores; no reduced analysis')
    return selected,excluded


def extract(case,split,cfg,annot,out):
    u,v,hv,time,truth=load_case(case,cfg,annot);cost=cosine_cost(u,v)
    b=cfg['bayes'];hop=cfg['hop_sec'];nexp=int(round(cfg['exposure_sec']/hop))
    starts={int(round(t/hop)):t for t in cfg['landmarks_sec']}
    p=np.zeros(len(v));p[0]=1;path=[];frozen=[]
    for i,row in enumerate(cost):
        if i:p=predict_prior(p,b['transition'],b['restart_probability'])
        if i in starts:
            center=int(round(truth[i-1]))+1
            regions,mass,wrong=freeze_regions(p,center,max(1,int(round(cfg['region_halfwidth_sec']/hv))),max(1,int(round(cfg['exclude_sec']/hv))))
            w=fixed_path_evidence(cost[i:i+nexp],regions,b['emission_beta'])
            frozen.append(dict(start=i,time_sec=starts[i],correct_center=center,wrong_center=wrong,
                correct_prior_mass=mass[0],wrong_prior_mass=mass[1],risk=mass[1]/sum(mass),
                prior_log_odds=float(np.log(mass[0])-np.log(mass[1])),log_wrong_prior=float(np.log(mass[1])),
                prior_entropy=float(-np.sum(p[p>0]*np.log(p[p>0]))),weakness=w))
        p=p*np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum();path.append(int(np.argmax(p)))
    path=np.array(path);expected=bayes_from_cost(cost,b['transition'],b['emission_beta'],b['restart_probability'])
    np.testing.assert_array_equal(path,expected)
    if split=='train':
        with np.load((ROOT/cfg['training_manifest']).parent/'cases'/(case['case_id']+'.npz')) as saved:
            np.testing.assert_array_equal(path,saved['intact_bayes'])
    error=abs(path-truth)*hv;rows=[]
    for r in frozen:
        i=r['start'];pre=i-1;end=i+nexp-1
        item=dict(**case,split=split,**r,exposure_last_time_sec=float(time[end]),
            pre_log_error=float(np.log1p(error[pre])),past2_log_error=float(np.log1p(error[pre-20])),
            past5_log_error=float(np.log1p(error[pre-50])),past_tempo=float((truth[pre]-truth[pre-20])*hv/2),
            end_log_error=float(np.log1p(error[end])),pre_healthy=bool(np.max(error[pre-19:pre+1])<=.5))
        for h in cfg['horizons_sec']:
            j=end+int(round(h/hop));assert j<len(error)
            item[f'y_{h:g}s']=float(np.log1p(error[j])-np.log1p(error[end]))
        rows.append(item)
    np.savez_compressed(out/'cases'/(split+'_'+case['case_id']+'.npz'),time_sec=time,truth_frames=truth,hop_score_sec=hv,path=path,error_sec=error)
    return rows


def design(rows):
    keys=['pre_log_error','past2_log_error','past5_log_error','time_sec','past_tempo','prior_log_odds','log_wrong_prior','prior_entropy']
    x=np.array([[r[k] for k in keys] for r in rows]);w=np.array([r['weakness'] for r in rows]);risk=np.array([r['risk'] for r in rows]);end=np.array([r['end_log_error'] for r in rows])
    return dict(M0=x,M1=np.column_stack([x,w]),M2=np.column_stack([x,w,w*risk]),M3=np.column_stack([x,end]),M4=np.column_stack([x,end,w,w*risk]))


def evaluate(rows,cfg):
    train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='test']
    a,z=design(train),design(test);names=sorted({r['composition'] for r in test})
    weights=np.array([1/sum(s['composition']==r['composition'] for s in train) for r in train])
    predictions=[];losses=[];comparisons=[]
    for h in cfg['horizons_sec']:
        y=np.array([r[f'y_{h:g}s'] for r in train]);target=np.array([r[f'y_{h:g}s'] for r in test])
        pred={m:ridge_predict(a[m],y,z[m],weights,cfg['ridge_alpha']) for m in a}
        for k,r in enumerate(test):predictions.append(dict(case_id=r['case_id'],composition=r['composition'],time_sec=r['time_sec'],horizon_sec=h,target=target[k],**{m:float(p[k]) for m,p in pred.items()}))
        grouped={m:[] for m in a}
        for name in names:
            ix=np.array([r['composition']==name for r in test]);rr=dict(composition=name,horizon_sec=h)
            for m in a:rr[m]=float(np.mean(abs(target[ix]-pred[m][ix])));grouped[m].append(rr[m])
            losses.append(rr)
        for base,full in [('M0','M1'),('M0','M2'),('M3','M4'),('M1','M2')]:
            effect=np.array(grouped[base])-np.array(grouped[full]);composers=sorted({n.split('/')[0] for n in names})
            by_composer=[float(np.mean([v for n,v in zip(names,effect) if n.split('/')[0]==c])) for c in composers]
            comparisons.append(dict(name=f'{full}_vs_{base}',horizon_sec=h,primary=h==5 and full!='M2' or h==5 and base=='M0',
                baseline_mae=float(np.mean(grouped[base])),full_mae=float(np.mean(grouped[full])),
                improvement_percent=float(100*effect.mean()/np.mean(grouped[base])),**pair_summary(effect,cfg),
                composer_balanced=pair_summary(by_composer,cfg),
                leave_composer_out={c:float(np.mean([v for n,v in zip(names,effect) if n.split('/')[0]!=c])) for c in composers}))
    primary=[c for c in comparisons if c['primary']]
    assert len(primary)==3
    for c,p in zip(primary,holm_adjust([c['p'] for c in primary])):
        c['holm_p']=float(p);c['passed']=bool(c['mean']>0 and c['ci95'][0]>0 and p<=.05)
    desc={}
    for split,rr in [('train',train),('test',test)]:
        desc[split]=dict(scores=len({r['score'] for r in rr}),landmarks=len(rr),risk_ge_1percent=sum(r['risk']>=.01 for r in rr),pre_healthy=sum(r['pre_healthy'] for r in rr),weakness_favors_wrong=sum(r['weakness']>0 for r in rr))
    return dict(comparisons=comparisons,description=desc),predictions,losses


def main():
    sys.stdout.reconfigure(encoding='utf-8');cfgpath=ROOT/'configs/p3_state_conditioned.yaml';cfg=yaml.safe_load(cfgpath.read_text(encoding='utf-8'))
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'))
    test,excluded=select(cfg,annot);train=read_csv(ROOT/cfg['training_manifest'])
    out=ROOT/'results'/f"{cfg['experiment']}_{datetime.now():%Y%m%d_%H%M%S}";out.mkdir();(out/'cases').mkdir();(out/'source_snapshot').mkdir()
    write_csv(out/'manifest.csv',test);write_csv(out/'training_manifest.csv',train);write_csv(out/'eligibility_exclusions.csv',excluded)
    sources=[cfgpath,Path(__file__),ROOT/'P3_STATE_PROTOCOL.md',ROOT/'scripts/p3_temporal.py']+list((ROOT/'src/score_following').glob('*.py'))
    inputs=[ROOT/cfg['asap_root']/c[k] for c in train+test for k in ['score','performance']]+[ROOT/cfg['asap_root']/'asap_annotations.json',ROOT/cfg['asap_root']/'metadata.csv']+[ROOT/f for f in cfg['excluded_manifests']]
    for f in sources:shutil.copy2(f,out/'source_snapshot'/f.name)
    shutil.copy2(cfgpath,out/'config.yaml')
    (out/'provenance.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources+inputs}),indent=2),encoding='utf-8')
    print(f'Frozen 40 train + 40 test: {out}',flush=True);rows=[]
    with threadpool_limits(limits=1):
        for split,cases in [('train',train),('test',test)]:
            for j,c in enumerate(cases):
                rows.extend(extract(c,split,cfg,annot,out));write_csv(out/'landmarks.csv',rows)
                print(f'{split} {j+1}/{len(cases)} {c["composition"]}',flush=True)
        metrics,pred,loss=evaluate(rows,cfg)
    write_csv(out/'predictions.csv',pred);write_csv(out/'score_losses.csv',loss)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps([c for c in metrics['comparisons'] if c['primary']],indent=2),flush=True)


if __name__=='__main__':main()
