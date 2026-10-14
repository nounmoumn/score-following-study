"""Apply identical pre-landmark warning to every healthy development landmark."""
import json,sys,hashlib,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import load_case,read_csv,write_csv
from score_following.current_evidence import current_evidence
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import cosine_cost

def sustained_start(error, start, end, threshold=.5, dwell=5):
    """Onset must fall before end; confirmation may finish four frames later."""
    for i in range(start,end):
        if np.all(error[i:i+dwell]>threshold):return i
    return None

def summarize(rows,key):
    p=np.array([r[key] for r in rows],bool);y=np.array([r['sustained'] for r in rows],bool)
    tp=int(np.sum(p&y));fp=int(np.sum(p&~y));fn=int(np.sum(~p&y));tn=int(np.sum(~p&~y))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,sensitivity=tp/(tp+fn) if tp+fn else None,
        false_alarm_rate=fp/(fp+tn),positive_predictive_value=tp/(tp+fp) if tp+fp else None,
        alert_rate=(tp+fp)/len(rows))

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    root=ROOT/'results/p3_onset_development';source=ROOT/'results/p3_temporal_20260918_000058';out=root/'short_signal'
    out.mkdir(exist_ok=True);(out/'source_snapshot').mkdir(exist_ok=True)
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes'];cases=read_csv(source/'manifest.csv')
    marks=read_csv(root/'landmarks.csv');by_case={c['case_id']:[r for r in marks if r['case_id']==c['case_id'] and r['eligible']=='True'] for c in cases}
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'))
    rows=[];maximum=0.
    with threadpool_limits(limits=1):
        for c in cases:
            queries=by_case[c['case_id']]
            if not queries:continue
            u,v,hv,time,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v);ix=np.arange(len(v))
            with np.load(source/'cases'/(c['case_id']+'.npz')) as z:path=z['intact_bayes'];error=abs(path-truth)*hv
            needed={i for r in queries for i in range(int(round(float(r['time_sec'])/.1))-20,int(round(float(r['time_sec'])/.1)))}
            p=np.zeros(len(v));p[0]=1;signal={}
            for i,row in enumerate(cost):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                if i in needed:
                    p,m=current_evidence(p,row,abs(ix-truth[i])*hv<=.5,b['emission_beta']);signal[i]=m['B'];maximum=max(maximum,abs(m['identity_residual']))
                else:p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum()
                if int(p.argmax())!=path[i]:raise AssertionError(f'path mismatch {c["case_id"]} {i}')
            for r in queries:
                start=int(round(float(r['time_sec'])/.1));bs=np.array([signal[i] for i in range(start-20,start)])
                onset=sustained_start(error,start,start+20)
                rows.append(dict(case_id=c['case_id'],composition=c['composition'],work_family=r['work_family'],time_sec=float(r['time_sec']),
                    status=r['status'],sustained=onset is not None,onset_frame=onset,onset_sec=None if onset is None else onset*.1,
                    any_positive=bool(np.any(bs>0)),two_consecutive=bool(np.any((bs[:-1]>0)&(bs[1:]>0))),
                    positive_frames=int(np.sum(bs>0)),mean_B=float(bs.mean()),min_B=float(bs.min()),max_B=float(bs.max())))
            print(c['case_id'],flush=True)
    write_csv(out/'landmarks.csv',rows)
    assert len(rows)==396
    reference=[r for r in rows if r['status']=='exposure_crossing']
    assert len(reference)==25 and sum(r['sustained'] for r in reference)==7
    assert all(not r['sustained'] for r in rows if r['status'] in ('no_failure','new_failure'))
    summary=dict(healthy_landmarks=len(rows),sustained=sum(r['sustained'] for r in rows),
        event_works=len({r['work_family'] for r in rows if r['sustained']}),
        base_rate=sum(r['sustained'] for r in rows)/len(rows),
        unique_onsets=len({(r['case_id'],r['onset_frame']) for r in rows if r['sustained']}),
        status_counts=dict(Counter(r['status'] for r in rows)),max_identity_residual=maximum,
        any_positive=summarize(rows,'any_positive'),two_consecutive=summarize(rows,'two_consecutive'))
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    files=[Path(__file__),ROOT/'P3_SHORT_SIGNAL_PROTOCOL.md',ROOT/'src/score_following/current_evidence.py',source/'config.yaml',source/'manifest.csv',root/'landmarks.csv']
    for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
    (out/'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
