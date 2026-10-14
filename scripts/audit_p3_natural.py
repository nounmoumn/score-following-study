"""Diagnostic-only latency/selection/candidate audit of frozen natural P3 results.

No alternative P3 p-values, cutoffs, or healthy-cohort re-selection.
"""
import argparse
import csv
import json
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from p3_temporal import load_case
from score_following.p3_extension import cosine_cost,oltw_from_cost
from score_following.temporal import healthy_before


def read(path):
    with open(path,encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def write(path,rows):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def candidate(u,v,truth,i,window,exclude,radius):
    pos=np.rint(truth[i-window+1:i+1]).astype(int);center=pos[-1];offsets=pos-center
    ends=np.arange(max(-offsets.min(),center-radius),min(len(v)-offsets.max(),center+radius+1))
    templates=v[ends[:,None]+offsets[None,:]].reshape(len(ends),-1)
    query=u[i-window+1:i+1].reshape(-1)
    den=np.linalg.norm(templates,axis=1)*np.linalg.norm(query)
    sim=np.divide(templates@query,den,out=np.zeros(len(ends)),where=den>0)
    mask=abs(ends-center)>exclude;wrong=ends[mask][np.argmax(sim[mask])]
    return center,int(wrong),float(sim[mask].max()-sim[ends==center][0])


def bayes_audit(cost,candidates,truth,radius,settings):
    n=cost.shape[1];p=np.zeros(n);p[0]=1
    out={};path=[]
    for i in range(len(cost)):
        if i:
            prior=np.zeros(n)
            for step,weight in enumerate(settings['transition']):
                if step==0:prior+=weight*p
                else:
                    prior[step:]+=weight*p[:-step];prior[-1]+=weight*p[-step:].sum()
            p=(1-settings['restart_probability'])*prior+settings['restart_probability']/n
        p*=np.exp(-settings['emission_beta']*(cost[i]-cost[i].min()));p/=p.sum()
        path.append(int(np.argmax(p)))
        if candidates[i,0]>=0:
            correct,wrong=candidates[i]
            true_mass=p[max(0,correct-radius):min(n,correct+radius+1)].sum()
            wrong_mass=p[max(0,wrong-radius):min(n,wrong+radius+1)].sum()
            out[i]=dict(correct_mass=float(true_mass),wrong_mass=float(wrong_mass),
                        wrong_to_correct_mass_ratio=float(wrong_mass/max(true_mass,1e-300)))
    return np.array(path),out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);args=p.parse_args()
    source=ROOT/args.result;cfg=yaml.safe_load((source/'config.yaml').read_text(encoding='utf-8'))
    out=source/'natural_audit';out.mkdir(exist_ok=True)
    manifest=read(source/'manifest.csv');events=read(source/'natural_event_audit.csv');pairs=read(source/'natural_pairs.csv')
    evidence=read(source/'evidence.csv')
    annot=json.loads((ROOT/cfg['asap_root']/'asap_annotations.json').read_text(encoding='utf-8'))
    hop=cfg['hop_sec'];history=int(round(cfg['healthy_history_sec']/hop));band=int(round(cfg['band_sec']/hop))
    rows=[];candidate_rows=[];match_rows=[]
    with threadpool_limits(limits=1):
        for case in manifest:
            ev=[r for r in events if r['case_id']==case['case_id']]
            pp=[r for r in pairs if r['case_id']==case['case_id']]
            if not ev:continue
            with np.load(source/'cases'/(case['case_id']+'.npz')) as data:
                truth=data['truth_frames'];hv=float(data['hop_score_sec'])
                paths={tr:data['intact_'+tr].copy() for tr in ['oltw','bayes']}
            u,v,hv2,time,truth2=load_case(case,cfg,annot)
            np.testing.assert_allclose(truth,truth2);assert abs(hv-hv2)<1e-12
            queries=sorted({float(r['event_time_sec']) for r in ev}|{float(r['control_sec']) for r in pp})
            candidates=np.full((len(u),2),-1,dtype=np.int64);weakness={}
            for t in queries:
                i=int(round(t/hop))-1
                correct,wrong,w=candidate(u,v,truth,i,int(round(cfg['evidence_window_sec']/hop)),
                    max(1,int(round(cfg['candidate_exclude_score_sec']/hv))),max(1,int(round(cfg['candidate_radius_score_sec']/hv))))
                candidates[i]=[correct,wrong];weakness[i]=w
            costs=cosine_cost(u,v);values=np.full((len(u),3),np.nan)
            replay=oltw_from_cost(costs,band,audit_candidates=candidates,audit_values=values)
            np.testing.assert_array_equal(replay,paths['oltw'])
            replay_b,posterior=bayes_audit(costs,candidates,truth,max(1,int(round(.5/hv))),cfg['bayes'])
            np.testing.assert_array_equal(replay_b,paths['bayes'])
            for t in queries:
                i=int(round(t/hop))-1;correct,wrong=candidates[i];frontier=int(values[i,2])
                cc,wc=values[i,:2]
                role='event' if any(float(r['event_time_sec'])==t for r in ev) else 'control'
                candidate_rows.append(dict(case_id=case['case_id'],composition=case['composition'],time_sec=t,role=role,
                    weakness=weakness[i],correct_frame=int(correct),wrong_frame=int(wrong),wrong_offset_score_sec=float((wrong-correct)*hv),
                    oltw_frontier=frontier,oltw_correct_accessible=bool(np.isfinite(cc)),oltw_wrong_accessible=bool(np.isfinite(wc)),
                    oltw_correct_norm_cost=float(cc) if np.isfinite(cc) else None,
                    oltw_wrong_norm_cost=float(wc) if np.isfinite(wc) else None,
                    oltw_wrong_minus_correct_cost=float(wc-cc) if np.isfinite(wc) and np.isfinite(cc) else None,
                    **{'bayes_'+k:val for k,val in posterior[i].items()}))
            for event in ev:
                tr=event['tracker'];t=float(event['event_time_sec']);confirm=int(round(t/hop))-1
                first=confirm-int(round(1/hop));pre_first=first-1
                error=abs(paths[tr]-truth)*hv
                hpre=healthy_before(error,pre_first,history,cfg['healthy_error_sec'])
                hfirst=healthy_before(error,first,history,cfg['healthy_error_sec'])
                hconfirm=healthy_before(error,confirm,history,cfg['healthy_error_sec'])
                if hconfirm!=(event['status']!='unhealthy'):raise AssertionError('health reconstruction mismatch')
                # Detection transition lies between the sample before first weak and first weak.
                prev_sample=first-int(round(1/hop))
                hprev=healthy_before(error,prev_sample,history,cfg['healthy_error_sec'])
                rows.append(dict(case_id=case['case_id'],composition=case['composition'],tracker=tr,
                    confirmation_sec=t,first_weak_sample_sec=t-1,status=event['status'],
                    healthy_at_previous_sample=hprev,healthy_before_first_weak=hpre,
                    healthy_at_first_weak=hfirst,healthy_at_confirmation=hconfirm,
                    lost_during_confirmation=bool(hfirst and not hconfirm),
                    lost_across_sampling_and_confirmation=bool(hprev and not hconfirm),
                    error_previous=float(error[prev_sample]),error_first=float(error[first]),error_confirm=float(error[confirm]),
                    max_error_since_first=float(error[first:confirm+1].max())))
                if hconfirm:
                    grid=[r for r in evidence if r['case_id']==case['case_id'] and r['tracker']==tr]
                    times=np.array([float(r['time_sec']) for r in grid]);ww=np.array([float(r['weakness']) for r in grid])
                    hs=np.array([r['healthy']=='True' for r in grid])
                    ee=np.array([float(r['current_error_sec']) for r in grid]);dd=np.array([float(r['past_error_change2']) for r in grid]);tt=np.array([float(r['past_tempo']) for r in grid])
                    ix=int(np.flatnonzero(abs(times-t)<1e-8)[0]);ec=cfg['event']
                    strong=np.zeros(len(grid),bool);strong[1:]=(ww[1:]<=ec['strong_threshold'])&(ww[:-1]<=ec['strong_threshold'])
                    healthy=strong&hs;separated=healthy&(abs(times-t)>=ec['control_separation_sec'])
                    caliper=separated&(abs(ee-ee[ix])<=ec['error_caliper_sec'])&(abs(dd-dd[ix])<=ec['trend_caliper_sec'])&(abs(tt-tt[ix])<=ec['tempo_caliper'])
                    match_rows.append(dict(case_id=case['case_id'],tracker=tr,time_sec=t,status=event['status'],
                        strong_candidates=int(strong.sum()),healthy_candidates=int(healthy.sum()),
                        separated_candidates=int(separated.sum()),caliper_candidates_before_reuse=int(caliper.sum())))
            print(f"audited {case['case_id']} {case['composition']}",flush=True)
    write(out/'latency_events.csv',rows);write(out/'candidate_competition.csv',candidate_rows);write(out/'matching_funnel.csv',match_rows)
    metrics={'latency':{},'candidates':{},'matching':{},'scope':'Post-hoc measurement audit; original P3 tests unchanged.'}
    for tr in ['oltw','bayes']:
        rr=[r for r in rows if r['tracker']==tr]
        metrics['latency'][tr]={k:sum(bool(r[k]) for r in rr) for k in
            ['healthy_at_previous_sample','healthy_before_first_weak','healthy_at_first_weak','healthy_at_confirmation',
             'lost_during_confirmation','lost_across_sampling_and_confirmation']}
        metrics['latency'][tr].update(events=len(rr),unhealthy=sum(r['status']=='unhealthy' for r in rr))
        mm=[r for r in match_rows if r['tracker']==tr and r['status']=='unmatched']
        metrics['matching'][tr]=dict(unmatched=len(mm),no_strong=sum(r['strong_candidates']==0 for r in mm),
            no_healthy=sum(r['healthy_candidates']==0 for r in mm),no_separated=sum(r['separated_candidates']==0 for r in mm),
            no_caliper=sum(r['caliper_candidates_before_reuse']==0 for r in mm),
            blocked_by_reuse_or_overlap=sum(r['caliper_candidates_before_reuse']>0 for r in mm))
    for role in ['event','control']:
        rr=[r for r in candidate_rows if r['role']==role]
        comparable=[r for r in rr if r['oltw_wrong_minus_correct_cost'] is not None]
        metrics['candidates'][role]=dict(n=len(rr),
            wrong_ahead_of_frontier=sum(r['wrong_frame']>r['oltw_frontier'] for r in rr),
            wrong_accessible=sum(r['oltw_wrong_accessible'] for r in rr),
            correct_accessible=sum(r['oltw_correct_accessible'] for r in rr),
            both_accessible=len(comparable),wrong_accum_cost_at_least_as_good=sum(r['oltw_wrong_minus_correct_cost']<=0 for r in comparable),
            bayes_wrong_mass_median=float(np.median([r['bayes_wrong_mass'] for r in rr])),
            bayes_correct_mass_median=float(np.median([r['bayes_correct_mass'] for r in rr])),
            bayes_wrong_mass_ge_correct=sum(r['bayes_wrong_mass']>=r['bayes_correct_mass'] for r in rr))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2))
    sources=[Path(__file__),ROOT/'src/score_following/p3_extension.py',source/'config.yaml',source/'manifest.csv',source/'natural_event_audit.csv',source/'evidence.csv']
    (out/'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__),out/'audit_source.py')


if __name__=='__main__':main()
