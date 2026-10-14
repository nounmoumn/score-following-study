import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import read_csv,write_csv,load_case
from score_following.current_evidence import current_evidence
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import cosine_cost
from score_following.multicandidate import logistic_predict


def evaluate(rows):
    keys=['pre_log_error','end_log_error','past_error_change','time_sec','end_wrong','end_entropy','mean_wrong','mean_entropy']
    x=np.array([[r[k] for k in keys] for r in rows]);designs=dict(S=x,B=np.column_stack([x,[r['mean_B'] for r in rows]]),D=np.column_stack([x,[r['mean_D'] for r in rows]]))
    groups=np.array([r['work_family'] for r in rows]);y=np.array([r['failure'] for r in rows],float);names=np.unique(groups);weights=np.array([1/np.sum(groups==g) for g in groups]);pred={m:np.zeros(len(rows)) for m in ['C','S','B','D']}
    for g in names:
        test=groups==g;train=~test;pred['C'][test]=np.average(y[train],weights=weights[train])
        for m,xx in designs.items():pred[m][test]=logistic_predict(xx[train],y[train],xx[test],weights[train])
    losses=[]
    for g in names:
        ix=groups==g;r=dict(work_family=g,n=int(ix.sum()),events=int(y[ix].sum()))
        for m,p in pred.items():
            pp=np.clip(p[ix],1e-12,1-1e-12);r[m+'_brier']=float(np.mean((p[ix]-y[ix])**2));r[m+'_logloss']=float(np.mean(-y[ix]*np.log(pp)-(1-y[ix])*np.log1p(-pp)))
        losses.append(r)
    models={m:{loss:float(np.mean([r[m+'_'+loss] for r in losses])) for loss in ['brier','logloss']} for m in pred}
    comparisons={}
    for base,full in [('C','B'),('S','B'),('S','D'),('C','D')]:
        comparisons[full+'_vs_'+base]={loss+'_improvement_percent':100*(models[base][loss]-models[full][loss])/models[base][loss] for loss in ['brier','logloss']}
        comparisons[full+'_vs_'+base]['improved_works']=sum(r[full+'_brier']<r[base+'_brier'] for r in losses)
    gate=all(comparisons['B_vs_'+base][loss+'_improvement_percent']>=5 for base in ['C','S'] for loss in ['brier','logloss']) and comparisons['B_vs_S']['improved_works']>len(names)/2
    metrics=dict(works=len(names),windows=len(rows),events=int(y.sum()),models=models,comparisons=comparisons,independent_validation_gate=gate)
    for label,value in [('failure',True),('no_failure',False)]:
        rr=[r for r in rows if r['failure']==value];metrics[label]=dict(n=len(rr),median_B=float(np.median([r['mean_B'] for r in rr])),median_D=float(np.median([r['mean_D'] for r in rr])),positive_mean_B=sum(r['mean_B']>0 for r in rr))
    preds=[dict(case_id=r['case_id'],work_family=r['work_family'],time_sec=r['time_sec'],failure=r['failure'],**{m:float(p[i]) for m,p in pred.items()}) for i,r in enumerate(rows)]
    return metrics,losses,preds


def main():
    sys.stdout.reconfigure(encoding='utf-8');source=ROOT/'results/p3_temporal_20260918_000058';base=ROOT/'results/p3_onset_development';out=base/'current_position';out.mkdir(exist_ok=True);(out/'source_snapshot').mkdir(exist_ok=True)
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes'];cases=read_csv(source/'manifest.csv');landmarks=read_csv(base/'landmarks.csv');annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'))
    files=[Path(__file__),ROOT/'P3_CURRENT_POSITION_PROTOCOL.md',ROOT/'src/score_following/current_evidence.py',ROOT/'src/score_following/multicandidate.py',ROOT/'tests/test_current_evidence.py',base/'landmarks.csv',source/'provenance.json']
    for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
    (out/'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8');rows=[];frames=[];max_res=0.
    with threadpool_limits(limits=1):
        for c in cases:
            queries={int(round(float(r['time_sec'])*10)):r for r in landmarks if r['case_id']==c['case_id'] and r['status'] in ['new_failure','no_failure']}
            u,v,hv,time,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v);indices=np.arange(len(v));p=np.zeros(len(v));p[0]=1;trace={}
            with np.load(source/'cases'/(c['case_id']+'.npz')) as saved:path=saved['intact_bayes'];error=abs(path-truth)*hv
            needed={i for s in queries for i in range(s,s+20)}
            for i,row in enumerate(cost):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                if i in needed:
                    correct=abs(indices-truth[i])*hv<=.5;p,metrics=current_evidence(p,row,correct,b['emission_beta']);trace[i]=metrics
                    max_res=max(max_res,abs(metrics['identity_residual']));frames.append(dict(case_id=c['case_id'],frame=i,**metrics))
                else:p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum()
                assert int(p.argmax())==path[i]
            for start,r in queries.items():
                window=[trace[i] for i in range(start,start+20)];last=window[-1]
                rows.append(dict(case_id=c['case_id'],work_family=r['work_family'],time_sec=float(r['time_sec']),failure=r['status']=='new_failure',pre_log_error=float(np.log1p(error[start-1])),end_log_error=float(np.log1p(error[start+19])),past_error_change=float(error[start-1]-error[start-21]),
                    end_wrong=last['posterior_wrong'],end_entropy=last['entropy'],mean_wrong=float(np.mean([w['posterior_wrong'] for w in window])),mean_entropy=float(np.mean([w['entropy'] for w in window])),mean_B=float(np.mean([w['B'] for w in window])),mean_D=float(np.mean([w['D'] for w in window])) ))
            print(c['case_id'],flush=True)
        assert len(rows)==371
        metrics,losses,preds=evaluate(rows)
    metrics['max_identity_residual']=max_res
    for name,data in [('features',rows),('frames',frames),('work_losses',losses),('predictions',preds)]:write_csv(out/(name+'.csv'),data)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
