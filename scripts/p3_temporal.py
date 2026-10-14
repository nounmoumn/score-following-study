"""Original P3: pre-healthy evidence interventions and past-defined natural events."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.ground_truth import piecewise_linear_gt
from score_following.p3_extension import cosine_cost,oltw_from_cost,bayes_from_cost,holm_adjust,sign_flip_p
from score_following.prospective import bootstrap_mean
from score_following.temporal import (bayes_instrumented,local_weakness,healthy_before,
                                      match_events,first_sustained_failure)


def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    if not rows:return
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def select_cases(cfg):
    root=ROOT/cfg['asap_root'];meta=read_csv(root/'metadata.csv')
    previous=read_csv(ROOT/cfg['previous_manifest'])
    used={r['composition'] for r in previous};used_scores={r['score'] for r in previous}
    annot=json.loads((root/'asap_annotations.json').read_text(encoding='utf-8'))
    buckets=defaultdict(list)
    for r in meta:buckets[r['composer']].append(r)
    ordered=[b[i] for i in range(max(map(len,buckets.values()))) for b in buckets.values() if i<len(b)]
    cases,excluded=[],[]
    for r in ordered:
        label=r['composer']+'/'+r['title']
        if label in used or r['midi_score'] in used_scores:continue
        a=annot.get(r['midi_performance'],{});beats=a.get('performance_beats',[])
        reason=None
        if not a.get('score_and_performance_aligned'):reason='not_aligned'
        elif not beats or min(beats)>15 or max(beats)<90:reason='annotation_coverage'
        elif not (root/r['midi_performance']).exists() or not (root/r['midi_score']).exists():reason='missing_midi'
        else:
            notes=load_notes(root/r['midi_performance'])
            if not notes or max(n[1] for n in notes)<90:reason='short_performance'
        if reason:
            excluded.append(dict(composition=label,performance=r['midi_performance'],reason=reason));continue
        cases.append(dict(case_id=f"n{len(cases):03}",composition=label,performance=r['midi_performance'],score=r['midi_score']))
        used.add(label);used_scores.add(r['midi_score'])
        if len(cases)==cfg['new_scores']:break
    if len(cases)!=cfg['new_scores']:raise ValueError('insufficient new scores')
    return cases,excluded,annot,previous[:cfg['calibration']['old_performances']]


def load_case(case,cfg,annot):
    root=ROOT/cfg['asap_root'];hop=cfg['hop_sec']
    pn,sn=load_notes(root/case['performance']),load_notes(root/case['score'])
    pd,sd=max(n[1] for n in pn),max(n[1] for n in sn)
    hv=hop*sd/pd
    u=notes_to_chroma_stream(pn,cfg['duration_sec'],hop)
    v=notes_to_chroma_stream(sn,sd,hv)
    a=annot[case['performance']];gt,lo,hi=piecewise_linear_gt(a['performance_beats'],a['midi_score_beats'])
    time=np.arange(len(u))*hop
    truth=gt(time)/hv
    return u,v,hv,time,truth


def pair_summary(values,cfg):
    x=np.asarray(values,float)
    if len(x)<2:
        return dict(n=len(x),mean=float(x.mean()) if len(x) else None,ci95=None,p=1.,positive=int((x>0).sum()))
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),
                ci95=bootstrap_mean(x,cfg['seed'],cfg['n_bootstrap']),
                p=sign_flip_p(x,cfg['seed'],cfg['n_sign_flips']),positive=int((x>0).sum()))


def calibrate(cases,cfg,annot,out):
    hop=cfg['hop_sec'];onset=int(round(cfg['onset_sec']/hop));end=int(round(cfg['erasure_end_sec']/hop))
    b=cfg['bayes'];rows=[];trace={}
    for case in cases:
        u,v,hv,time,truth=load_case(case,cfg,annot);cost=cosine_cost(u,v)
        original=bayes_from_cost(cost,b['transition'],b['emission_beta'],b['restart_probability'])
        for erased in [False,True]:
            obs=cost.copy()
            if erased:obs[onset:end]=obs[onset:end].mean(axis=1,keepdims=True)
            for shift in cfg['calibration']['shift_frames']:
                pos,entropy,mass=bayes_instrumented(obs,truth,max(1,int(round(.5/hv))),onset=onset,shift=shift,
                    transition=b['transition'],beta=b['emission_beta'],restart=b['restart_probability'])
                if not np.array_equal(pos[:onset],original[:onset]):raise AssertionError('calibration changed prefix')
                if shift==0 and not erased and not np.array_equal(pos,original):raise AssertionError('zero shift mismatch')
                error=abs(pos-truth)*hv
                rows.append(dict(composition=case['composition'],performance=case['performance'],erased=erased,
                    shift_frames=shift,shift_score_sec=shift*hv,pre_error_sec=float(error[onset-1]),
                    error_after5_sec=float(error[onset+49]),mean_error_during_sec=float(error[onset:end].mean()),
                    mean_entropy_during=float(entropy[onset:end].mean()),
                    mean_true_mass_during=float(mass[onset:end].mean())))
                key=case['case_id']+f'_{int(erased)}_{shift}'
                trace[key+'_error']=error;trace[key+'_entropy']=entropy;trace[key+'_mass']=mass
    write_csv(out/'calibration.csv',rows)
    np.savez_compressed(out/'calibration_traces.npz',**trace)
    return rows


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/p3_temporal.yaml');args=p.parse_args()
    cfg_path=ROOT/args.config;cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    out=ROOT/'results'/f"{cfg['experiment']}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True);(out/'cases').mkdir();(out/'source_snapshot').mkdir()
    lf=open(out/'run.log','w',encoding='utf-8')
    def log(s):print(s,flush=True);lf.write(s+'\n');lf.flush()
    log(f'output: {out}');shutil.copy2(cfg_path,out/'config.yaml')
    cases,excluded,annot,calibration_cases=select_cases(cfg)
    write_csv(out/'manifest.csv',cases);write_csv(out/'eligibility_exclusions.csv',excluded)
    write_csv(out/'calibration_manifest.csv',calibration_cases)
    sources=[cfg_path,Path(__file__),ROOT/'P3_TEMPORAL_PROTOCOL.md',ROOT/'src/score_following/temporal.py',
        ROOT/'src/score_following/p3_extension.py',ROOT/'src/score_following/prospective.py',
        ROOT/'src/score_following/features.py',ROOT/'src/score_following/ground_truth.py']
    files=sources+[ROOT/cfg['asap_root']/c[k] for c in cases+calibration_cases for k in ['score','performance']]
    files += [ROOT/cfg['asap_root']/'metadata.csv',ROOT/cfg['asap_root']/'asap_annotations.json',ROOT/cfg['previous_manifest']]
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    for f in sources:shutil.copy2(f,out/'source_snapshot'/f.name)
    (out/'provenance.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,sha256=hashes),indent=2),encoding='utf-8')
    log(f'frozen {len(cases)} previously unused scores')
    intervention,natural,event_audit,evidence_rows=[],[],[],[]
    hop=cfg['hop_sec'];onset=int(round(cfg['onset_sec']/hop));end=int(round(cfg['erasure_end_sec']/hop))
    history=int(round(cfg['healthy_history_sec']/hop));window=int(round(cfg['evidence_window_sec']/hop))
    grid=np.arange(cfg['event']['grid_start_sec'],cfg['event']['grid_stop_sec']+.001,1.)
    frames=np.rint(grid/hop).astype(int)-1
    b=cfg['bayes']
    with threadpool_limits(limits=1):
        calibration=calibrate(calibration_cases,cfg,annot,out)
        log('calibration: 8 old performances x 6 conditions complete; not a P3 test')
        for case_no,case in enumerate(cases):
            u,v,hv,time,truth=load_case(case,cfg,annot);cost=cosine_cost(u,v)
            erased=cost.copy();erased[onset:end]=cost[onset:end].mean(axis=1,keepdims=True)
            estimates={}
            for name,obs in [('intact',cost),('erased',erased)]:
                estimates[name+'_oltw']=oltw_from_cost(obs,int(round(cfg['band_sec']/hop)))
                estimates[name+'_bayes']=bayes_from_cost(obs,b['transition'],b['emission_beta'],b['restart_probability'])
            weakness=np.array([local_weakness(u,v,truth,i,window,
                max(1,int(round(cfg['candidate_exclude_score_sec']/hv))),
                max(1,int(round(cfg['candidate_radius_score_sec']/hv)))) for i in frames])
            tempo=np.array([(truth[i]-truth[i-int(round(2/hop))])*hv/2 for i in frames])
            for tracker in ['oltw','bayes']:
                pos0,pos1=estimates['intact_'+tracker],estimates['erased_'+tracker]
                if not np.array_equal(pos0[:onset],pos1[:onset]):raise AssertionError('treatment altered prefix')
                err0,err1=abs(pos0-truth)*hv,abs(pos1-truth)*hv
                eligible=healthy_before(err0,onset-1,history,cfg['healthy_error_sec'])
                row=dict(**case,tracker=tracker,eligible=eligible,
                    pre_max_error_sec=float(err0[onset-history:onset].max()),pre_error_sec=float(err0[onset-1]))
                for h in cfg['horizons_sec']:
                    i=onset+int(round(h/hop))-1
                    row[f'effect_{h:g}s']=float(err1[i]-err0[i])
                    row[f'intact_{h:g}s']=float(err0[i]);row[f'erased_{h:g}s']=float(err1[i])
                for name,err in [('intact',err0),('erased',err1)]:
                    failure=first_sustained_failure(err,onset,end,cfg['healthy_error_sec'],int(round(.5/hop)))
                    row[name+'_failed10']=failure is not None
                    row[name+'_failure_onset_sec']=float((failure-onset)*hop) if failure is not None else None
                intervention.append(row)
                currents=err0[frames]
                trends=np.array([err0[i]-err0[i-int(round(2/hop))] for i in frames])
                health=np.array([healthy_before(err0,i,history,cfg['healthy_error_sec']) for i in frames])
                pairs,audit=match_events(grid,weakness,currents,trends,tempo,health,cfg['event'])
                for rr in audit:event_audit.append(dict(**case,tracker=tracker,**rr))
                for e,c in pairs:
                    ie,ic=frames[e],frames[c]
                    rr=dict(**case,tracker=tracker,event_sec=float(grid[e]),control_sec=float(grid[c]),
                        event_weakness=float(weakness[e]),control_weakness=float(weakness[c]),
                        current_error_difference=float(currents[e]-currents[c]),
                        pretrend_difference=float(trends[e]-trends[c]),tempo_difference=float(tempo[e]-tempo[c]))
                    for h in cfg['horizons_sec']:
                        offset=int(round(h/hop))
                        rr[f'effect_{h:g}s']=float((err0[ie+offset]-err0[ie])-(err0[ic+offset]-err0[ic]))
                    for label,i in [('event',ie),('control',ic)]:
                        failure=first_sustained_failure(err0,i+1,i+1+int(round(10/hop)),cfg['healthy_error_sec'],int(round(.5/hop)))
                        rr[label+'_failed10']=failure is not None
                        rr[label+'_failure_onset_sec']=float((failure-i)*hop) if failure is not None else None
                    natural.append(rr)
                for k,i in enumerate(frames):
                    evidence_rows.append(dict(**case,tracker=tracker,time_sec=float(grid[k]),weakness=float(weakness[k]),
                        current_error_sec=float(currents[k]),past_error_change2=float(trends[k]),past_tempo=float(tempo[k]),healthy=bool(health[k])))
            np.savez_compressed(out/'cases'/(case['case_id']+'.npz'),time_sec=time,truth_frames=truth,hop_score_sec=hv,**estimates)
            write_csv(out/'intervention.csv',intervention);write_csv(out/'natural_pairs.csv',natural)
            write_csv(out/'natural_event_audit.csv',event_audit);write_csv(out/'evidence.csv',evidence_rows)
            log(f"{case_no+1}/{len(cases)} {case['composition']}: complete")
    metrics=dict(intervention={},natural={},primary_tests=[],scope='Original P3 temporal checks; oracle diagnostic; no P1/P2 re-test')
    for tracker in ['oltw','bayes']:
        rr=[r for r in intervention if r['tracker']==tracker];ok=[r for r in rr if r['eligible']]
        desc=dict(eligible_scores=len(ok),all_scores=len(rr),horizons={},all_score_horizons={})
        for h in cfg['horizons_sec']:
            desc['horizons'][f'{h:g}s']=pair_summary([r[f'effect_{h:g}s'] for r in ok],cfg)
            desc['all_score_horizons'][f'{h:g}s']=pair_summary([r[f'effect_{h:g}s'] for r in rr],cfg)
        desc['intact_failed10']=sum(r['intact_failed10'] for r in ok)
        desc['erased_failed10']=sum(r['erased_failed10'] for r in ok)
        metrics['intervention'][tracker]=desc
        metrics['primary_tests'].append(dict(name='intervention_'+tracker,**desc['horizons']['5s']))
        paired=[r for r in natural if r['tracker']==tracker]
        groups=sorted({r['composition'] for r in paired})
        audits=[r for r in event_audit if r['tracker']==tracker]
        desc=dict(detected_events=len(audits),healthy_events=sum(r['status']!='unhealthy' for r in audits),
            matched_events=len(paired),contributing_scores=len(groups),horizons={})
        for h in cfg['horizons_sec']:
            values=[np.mean([r[f'effect_{h:g}s'] for r in paired if r['composition']==g]) for g in groups]
            desc['horizons'][f'{h:g}s']=pair_summary(values,cfg)
        desc['event_failed10']=sum(r['event_failed10'] for r in paired)
        desc['control_failed10']=sum(r['control_failed10'] for r in paired)
        desc['mean_pretrend_difference']=float(np.mean([r['pretrend_difference'] for r in paired])) if paired else None
        metrics['natural'][tracker]=desc
        metrics['primary_tests'].append(dict(name='natural_'+tracker,**desc['horizons']['5s']))
    # Inadequate sample never enters multiple-testing family as a favorable small-n test.
    for t in metrics['primary_tests']:
        if t['n']<cfg['minimum_scores']:t['p']=1.
    adjusted=holm_adjust([t['p'] for t in metrics['primary_tests']])
    for t,adj in zip(metrics['primary_tests'],adjusted):
        t['holm_p']=float(adj)
        t['passed']=bool(t['n']>=cfg['minimum_scores'] and t['ci95'] is not None and t['ci95'][0]>0 and adj<=.05)
        t['status']='insufficient_scores' if t['n']<cfg['minimum_scores'] else ('pass' if t['passed'] else 'not_supported')
    metrics['calibration']=[]
    for erased in [False,True]:
        for shift in cfg['calibration']['shift_frames']:
            rr=[r for r in calibration if r['erased']==erased and r['shift_frames']==shift]
            metrics['calibration'].append(dict(erased=erased,shift_frames=shift,
                **{k:float(np.mean([r[k] for r in rr])) for k in ['error_after5_sec','mean_error_during_sec','mean_entropy_during','mean_true_mass_during']}))
    metrics['prefix_check']='all paired paths identical before intervention; zero-shift Bayes matches original'
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    for t in metrics['primary_tests']:log(json.dumps(t))
    log('complete');lf.close()


if __name__=='__main__':main()
