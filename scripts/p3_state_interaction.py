"""One frozen exploratory alternative on cached development features."""
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from score_following.multicandidate import logistic_predict
def main():
    source=ROOT/'results/p3_onset_development/online_precursor/landmarks.csv'
    with source.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    y=np.array([r['sustained']=='True' for r in rows],float)
    g=np.array([r['work_family'] for r in rows]);groups=np.unique(g)
    assert len(rows)==396 and sum(y)==7 and len(groups)==35
    p=np.array([float(r['prior_margin_median']) for r in rows]);o=np.array([float(r['observation_margin_median']) for r in rows]);t=np.array([float(r['time_sec']) for r in rows])
    design={'S':np.column_stack([p,t]),'I':np.column_stack([p,t,o,p*o])}
    weights=np.array([1/sum(g==v) for v in g]);pred={m:np.zeros(len(rows)) for m in ('C','S','I')}
    for group in groups:
        test=g==group;train=~test
        pred['C'][test]=np.average(y[train],weights=weights[train])
        for model,x in design.items():pred[model][test]=logistic_predict(x[train],y[train],x[test],weights[train],alpha=1.)
    losses=[]
    for group in groups:
        ix=g==group;row={'work_family':group,'n':int(sum(ix)),'events':int(sum(y[ix]))}
        for m,pp in pred.items():
            q=np.clip(pp[ix],1e-12,1-1e-12)
            row[m+'_brier']=float(np.mean((pp[ix]-y[ix])**2))
            row[m+'_logloss']=float(np.mean(-y[ix]*np.log(q)-(1-y[ix])*np.log1p(-q)))
        losses.append(row)
    metrics={m:{l:float(np.mean([r[m+'_'+l] for r in losses])) for l in ('brier','logloss')} for m in pred}
    comparison={b:{l:100*(metrics[b][l]-metrics['I'][l])/metrics[b][l] for l in ('brier','logloss')} for b in ('C','S')}
    improved=sum(r['I_brier']<r['S_brier'] for r in losses)
    gate=all(v>=5 for c in comparison.values() for v in c.values()) and improved>len(groups)/2
    output=dict(scope='Exploratory reused development data, no independent P3 inference',metrics=metrics,relative_improvement_percent=comparison,improved_works_vs_S=improved,development_gate=gate,works=len(groups),events=int(sum(y)))
    out=ROOT/'results/p3_onset_development/state_interaction';out.mkdir(exist_ok=True)
    predictions=[dict(case_id=r['case_id'],time_sec=r['time_sec'],work_family=r['work_family'],original_event=bool(y[i]),**{m:float(v[i]) for m,v in pred.items()}) for i,r in enumerate(rows)]
    for name,data in [('predictions',predictions),('work_losses',losses)]:
        with (out/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    output['provenance']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (source,Path(__file__),ROOT/'P3_STATE_INTERACTION_PROTOCOL.md',ROOT/'src/score_following/multicandidate.py')}
    (out/'metrics.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
