import sys,json,shutil,hashlib
from pathlib import Path
import numpy as np
import yaml
from scipy.stats import t,nct
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from p3_temporal import load_case,read_csv,write_csv
from score_following.current_evidence import current_evidence
from score_following.observation_evidence import predict_prior
from score_following.p3_extension import cosine_cost


def power(n,d,alpha=.05):return float(nct.sf(t.ppf(1-alpha,n-1),n-1,d*np.sqrt(n)))


def main():
    sys.stdout.reconfigure(encoding='utf-8');base=ROOT/'results/p3_onset_development';source=ROOT/'results/p3_temporal_20260918_000058';out=base/'timing_sample';out.mkdir(exist_ok=True);(out/'source_snapshot').mkdir(exist_ok=True)
    cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'));b=cfg['bayes'];cases=read_csv(source/'manifest.csv');landmarks=read_csv(base/'landmarks.csv');selected=[r for r in landmarks if r['status']=='exposure_crossing']
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'));rows=[];frames=[]
    with threadpool_limits(limits=1):
        for c in cases:
            queries=[r for r in selected if r['case_id']==c['case_id']]
            if not queries:continue
            u,v,hv,time,truth=load_case(c,cfg,annot);cost=cosine_cost(u,v);idx=np.arange(len(v))
            with np.load(source/'cases'/(c['case_id']+'.npz')) as z:path=z['intact_bayes'];error=abs(path-truth)*hv
            p=np.zeros(len(v));p[0]=1;trace={}
            need={i for r in queries for i in range(int(float(r['time_sec'])*10)-40,int(float(r['time_sec'])*10)+24)}
            for i,row in enumerate(cost):
                if i:p=predict_prior(p,b['transition'],b['restart_probability'])
                if i in need:
                    p,m=current_evidence(p,row,abs(idx-truth[i])*hv<=.5,b['emission_beta']);trace[i]=m
                    frames.append(dict(case_id=c['case_id'],frame=i,**m))
                else:p*=np.exp(-b['emission_beta']*(row-row.min()));p/=p.sum()
                assert int(p.argmax())==path[i]
            for r in queries:
                start=int(round(float(r['time_sec'])*10));first=next(i for i in range(start,start+20) if error[i]>.5)
                onset=next((i for i in range(start,start+20) if np.all(error[i:i+5]>.5)),None)
                rr=dict(case_id=c['case_id'],composition=c['composition'],work_family=r['work_family'],time_sec=float(r['time_sec']),first_cross_sec=first*.1,
                    sustained_onset_sec=None if onset is None else onset*.1,first_cross_delay_sec=(first-start)*.1,sustained=onset is not None)
                if onset is not None:
                    positives=[i for i in range(start-20,onset) if trace[i]['B']>0]
                    rr.update(onset_delay_sec=(onset-start)*.1,confirmed_sec=(onset+4)*.1,pre2_mean_B=float(np.mean([trace[i]['B'] for i in range(onset-20,onset)])),
                        onset_B=trace[onset]['B'],positive_B_before=bool(positives),first_positive_lead_sec=(onset-positives[0])*.1 if positives else None,
                        map_step_at_onset_sec=float((path[onset]-path[onset-1])*hv))
                else:rr.update(onset_delay_sec=None,confirmed_sec=None,pre2_mean_B=None,onset_B=None,positive_B_before=False,first_positive_lead_sec=None,map_step_at_onset_sec=None)
                rows.append(rr)
            print(c['case_id'],flush=True)
    write_csv(out/'timing.csv',rows);write_csv(out/'frames.csv',frames)
    sustained=[r for r in rows if r['sustained']]
    summary=dict(windows=len(rows),sustained_windows=len(sustained),transient_only=len(rows)-len(sustained),
        unique_sustained_onsets=len({(r['case_id'],r['sustained_onset_sec']) for r in sustained}),sustained_works=len({r['work_family'] for r in sustained}),
        median_onset_delay_sec=float(np.median([r['onset_delay_sec'] for r in sustained])),positive_B_before=sum(r['positive_B_before'] for r in sustained),
        positive_pre2_mean_B=sum(r['pre2_mean_B']>0 for r in sustained),positive_onset_B=sum(r['onset_B']>0 for r in sustained),
        confirmation_after_exposure=sum(r['confirmed_sec']>=r['time_sec']+2 for r in sustained))
    plans=[]
    for d in [.2,.35,.5]:
        for target in [.8,.9]:
            n=next(n for n in range(3,2001) if power(n,d)>=target)
            plans.append(dict(standardized_effect=d,target_power=target,works=n,achieved_power=power(n,d),power_with25=power(25,d),expected_events_rate025=n*.25,expected_events_rate05=n*.5,expected_events_rate1=n))
    # Numerical sanity: null power equals alpha; power increases with N and d.
    assert abs(power(25,0)-.05)<1e-10
    assert power(50,.2)>power(25,.2) and power(25,.5)>power(25,.2)
    write_csv(out/'sample_scenarios.csv',plans);summary['sample_scenarios']=plans
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    files=[Path(__file__),ROOT/'P3_TIMING_SAMPLE_PROTOCOL.md',ROOT/'src/score_following/current_evidence.py',base/'landmarks.csv',source/'provenance.json']
    for f in files:shutil.copy2(f,out/'source_snapshot'/f.name)
    (out/'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
