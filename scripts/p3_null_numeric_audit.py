"""Paired exact-null correction audit with unchanged tracker and risk set."""
import csv,json,sys,hashlib,importlib.util
from pathlib import Path
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import load_case
from score_following.p3_extension import cosine_cost
from score_following.observation_evidence import predict_prior,regional_evidence
from score_following.current_evidence import current_evidence
BASE=ROOT/'results/p3_onset_development';SOURCE=ROOT/'results/p3_temporal_20260918_000058'
def legacy(name):
    path=BASE/'null_numeric/legacy_source'/(name+'.py')
    spec=importlib.util.spec_from_file_location('legacy_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def matrix(rows,key):
    tp=sum(r[key] and r['event'] for r in rows);fp=sum(r[key] and not r['event'] for r in rows)
    fn=sum(not r[key] and r['event'] for r in rows);tn=sum(not r[key] and not r['event'] for r in rows)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,fpr=fp/(fp+tn),sensitivity=tp/(tp+fn))
def main():
    original_current=legacy('current_evidence').current_evidence
    original_regional=legacy('observation_evidence').regional_evidence
    cases=read(SOURCE/'manifest.csv');refs=read(BASE/'short_signal/landmarks.csv')
    online={(r['case_id'],float(r['time_sec'])):r for r in read(BASE/'online_precursor/landmarks.csv')}
    cfg=yaml.safe_load((SOURCE/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes']
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'))
    rows=[];frames=[]
    with threadpool_limits(limits=1):
        for c in cases:
            rr=[r for r in refs if r['case_id']==c['case_id']]
            if not rr:continue
            u,v,hv,_,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v);ix=np.arange(len(v))
            with np.load(SOURCE/'cases'/(c['case_id']+'.npz')) as z:path=z['intact_bayes']
            needed={i for r in rr for i in range(round(float(r['time_sec'])*10)-20,round(float(r['time_sec'])*10))}
            p=np.zeros(len(v));p[0]=1;trace={}
            for i,cc in enumerate(cost):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                if i in needed:
                    args=(p,cc,abs(ix-truth[i])*hv<=.5,b['emission_beta'])
                    po,a=original_current(*args);fixed_po,fixed_a=current_evidence(*args)
                    rargs=(p,cc,int(p.argmax()),max(1,round(.5/hv)),max(1,round(2/hv)),b['emission_beta'])
                    pr,d=original_regional(*rargs);fixed_pr,fixed_d=regional_evidence(*rargs)
                    np.testing.assert_array_equal(po,pr)
                    flat=bool(np.ptp(cc)==0)
                    np.testing.assert_array_equal(fixed_po,po);np.testing.assert_array_equal(fixed_pr,pr)
                    assert fixed_a['B']==(0. if flat else a['B'])
                    assert fixed_d['observation_log_bayes_factor']==(0. if flat else d['observation_log_bayes_factor'])
                    trace[i]=dict(case_id=c['case_id'],frame=i,flat=flat,zero_input=bool(np.linalg.norm(u[i])==0),
                        oracle_B=a['B'],online_wrong_B=-d['observation_log_bayes_factor'])
                    frames.append(trace[i]);p=po
                else:p*=np.exp(-b['emission_beta']*(cc-cc.min()));p/=p.sum()
                assert int(p.argmax())==path[i]
            for r in rr:
                t=float(r['time_sec']);s=round(t*10);ff=[trace[i] for i in range(s-20,s)]
                row=dict(case_id=c['case_id'],time_sec=t,event=r['sustained']=='True',flat_frames=sum(f['flat'] for f in ff))
                for name,field in [('oracle','oracle_B'),('online','online_wrong_B')]:
                    for variant in ('raw','exact_null'):
                        arr=np.array([f[field]>0 and not(variant=='exact_null' and f['flat']) for f in ff])
                        row[name+'_'+variant+'_any']=bool(arr.any());row[name+'_'+variant+'_two']=bool(np.any(arr[:-1]&arr[1:]))
                for field,ref in [('oracle_raw_any',r['any_positive']),('oracle_raw_two',r['two_consecutive']),('online_raw_any',online[(c['case_id'],t)]['evidence_any']),('online_raw_two',online[(c['case_id'],t)]['evidence_two'])]:
                    assert row[field]==(ref=='True'),(c['case_id'],t,field)
                rows.append(row)
            print(c['case_id'],flush=True)
    assert len(rows)==396 and sum(r['event'] for r in rows)==7
    summary=dict(n=len(rows),unique_evaluated_frames=len(frames),flat_frames=sum(f['flat'] for f in frames),
        flat_positive_oracle=sum(f['flat'] and f['oracle_B']>0 for f in frames),
        flat_positive_online=sum(f['flat'] and f['online_wrong_B']>0 for f in frames),
        rules={k:matrix(rows,k) for k in rows[0] if '_raw_' in k or '_exact_null_' in k},
        changed_landmarks={name+'_'+run:sum(r[name+'_raw_'+run]!=r[name+'_exact_null_'+run] for r in rows) for name in ('oracle','online') for run in ('any','two')})
    out=BASE/'null_numeric';out.mkdir(exist_ok=True)
    for name,data in [('frames',frames),('landmarks',rows)]:
        with (out/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summary['fixed_implementation_verified']=True
    summary['provenance']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'P3_NULL_NUMERIC_PROTOCOL.md',ROOT/'src/score_following/current_evidence.py',ROOT/'src/score_following/observation_evidence.py',BASE/'null_numeric/legacy_source/current_evidence.py',BASE/'null_numeric/legacy_source/observation_evidence.py',SOURCE/'config.yaml',SOURCE/'manifest.csv',BASE/'short_signal/landmarks.csv',BASE/'online_precursor/landmarks.csv')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
