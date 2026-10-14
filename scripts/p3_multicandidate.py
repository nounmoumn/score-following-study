import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import load_case,read_csv,write_csv
from score_following.multicandidate import select_regions,likelihood_trace,logistic_predict
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import cosine_cost


def evaluate(rows):
    keys=['pre_log_error','end_log_error','past_error_change','time_sec','correct_mass','top1_mass','entropy']
    base=np.array([[r[k] for k in keys] for r in rows]);single=np.column_stack([base,[r['W1'] for r in rows]])
    multi=np.column_stack([single,[r['max_W'] for r in rows],[r['max_end_odds'] for r in rows]])
    designs=dict(B=base,S=single,M=multi);y=np.array([r['failure'] for r in rows],float);groups=np.array([r['work_family'] for r in rows]);names=np.unique(groups)
    weights=np.array([1/np.sum(groups==g) for g in groups]);pred={m:np.zeros(len(rows)) for m in ['constant','B','S','M']}
    for g in names:
        test=groups==g;train=~test;pred['constant'][test]=np.average(y[train],weights=weights[train])
        for m,x in designs.items():pred[m][test]=logistic_predict(x[train],y[train],x[test],weights[train])
    losses=[]
    for g in names:
        ix=groups==g;item=dict(work_family=g,n=int(ix.sum()),events=int(y[ix].sum()))
        for m,p in pred.items():
            prob=np.clip(p[ix],1e-12,1-1e-12);item[m+'_brier']=float(np.mean((p[ix]-y[ix])**2));item[m+'_logloss']=float(np.mean(-y[ix]*np.log(prob)-(1-y[ix])*np.log1p(-prob)))
        losses.append(item)
    metrics=dict(works=len(names),windows=len(rows),events=int(y.sum()),models={},comparisons={})
    for m in pred:metrics['models'][m]={loss:float(np.mean([r[m+'_'+loss] for r in losses])) for loss in ['brier','logloss']}
    for a,b in [('B','S'),('B','M'),('S','M')]:
        delta=np.array([r[a+'_brier']-r[b+'_brier'] for r in losses]);metrics['comparisons'][b+'_vs_'+a]=dict(brier_improvement=float(delta.mean()),relative_percent=float(100*delta.mean()/metrics['models'][a]['brier']),improved_works=int((delta>0).sum()))
    predictions=[dict(case_id=r['case_id'],work_family=r['work_family'],time_sec=r['time_sec'],failure=r['failure'],**{m:float(p[i]) for m,p in pred.items()}) for i,r in enumerate(rows)]
    return metrics,losses,predictions


def main():
    sys.stdout.reconfigure(encoding='utf-8');source=ROOT/'results/p3_temporal_20260918_000058';development=ROOT/'results/p3_onset_development';out=development/'multicandidate';out.mkdir(exist_ok=True);(out/'source_snapshot').mkdir(exist_ok=True)
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes'];cases=read_csv(source/'manifest.csv');landmarks=read_csv(development/'landmarks.csv')
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'));rows=[];candidates=[]
    files=[Path(__file__),ROOT/'src/score_following/multicandidate.py',ROOT/'P3_MULTICANDIDATE_PROTOCOL.md',ROOT/'tests/test_multicandidate.py',development/'landmarks.csv',source/'provenance.json']
    for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
    (out/'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
    with threadpool_limits(limits=1):
        for c in cases:
            queries={int(round(float(r['time_sec'])*10)):r for r in landmarks if r['case_id']==c['case_id'] and r['status'] in ['new_failure','no_failure']}
            if not queries:continue
            u,v,hv,time,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v)
            with np.load(source/'cases'/(c['case_id']+'.npz')) as z:path=z['intact_bayes'];error=abs(path-truth)*hv
            p=np.zeros(len(v));p[0]=1
            for i,row in enumerate(cost):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                if i in queries:
                    r=queries[i];regions,mass,centers=select_regions(p,int(round(truth[i-1]))+1,max(1,int(round(.5/hv))),max(1,int(round(2/hv))))
                    failure=r['status']=='new_failure';stop=int(r['failure_frame']) if failure else i+69
                    ll=likelihood_trace(cost[i:stop+1],regions,b['transition'],b['emission_beta'],b['restart_probability'])
                    w=(ll[19,1:]-ll[19,0])/20;odds=np.log(mass[1:]/mass[0])+20*w
                    later=np.log(mass[1:]/mass[0])+ll[-1,1:]-ll[-1,0]
                    np.testing.assert_allclose(w[0],float(r['weakness']),atol=1e-10)
                    rows.append(dict(case_id=c['case_id'],work_family=r['work_family'],time_sec=float(r['time_sec']),failure=failure,
                        pre_log_error=float(np.log1p(error[i-1])),end_log_error=float(np.log1p(error[i+19])),past_error_change=float(error[i-1]-error[i-21]),
                        correct_mass=float(mass[0]),top1_mass=float(mass[1]),entropy=float(-np.sum(p[p>0]*np.log(p[p>0]))),W1=float(w[0]),max_W=float(w.max()),max_end_odds=float(odds.max()),
                        end_best_rank=int(odds.argmax()+1),later_best_rank=int(later.argmax()+1),end_overtake=bool(odds.max()>0),later_overtake=bool(later.max()>0)))
                    for k in range(5):candidates.append(dict(case_id=c['case_id'],time_sec=r['time_sec'],prior_rank=k+1,center=centers[k+1],prior_mass=float(mass[k+1]),W=float(w[k]),end_odds=float(odds[k]),later_odds=float(later[k]),diagnostic_end_frame=stop))
                p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum();assert int(p.argmax())==path[i]
            write_csv(out/'features.csv',rows);write_csv(out/'candidates.csv',candidates);print(c['case_id'],flush=True)
        assert len(rows)==371
        metrics,losses,pred=evaluate(rows)
    metrics['diagnostics']={}
    for label,rr in [('failure',[r for r in rows if r['failure']]),('no_failure',[r for r in rows if not r['failure']])]:
        metrics['diagnostics'][label]=dict(n=len(rr),end_best_not_top1=sum(r['end_best_rank']!=1 for r in rr),later_best_not_top1=sum(r['later_best_rank']!=1 for r in rr),end_overtake=sum(r['end_overtake'] for r in rr),later_overtake=sum(r['later_overtake'] for r in rr))
    write_csv(out/'work_losses.csv',losses);write_csv(out/'predictions.csv',pred)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
