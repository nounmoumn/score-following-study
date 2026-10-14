"""Frozen development scenarios, not certified uncertainty or corrected labels."""
import csv,json,sys,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from score_following.interval_gt import sustained_bounds
BASE=ROOT/'results/p3_onset_development';OUT=BASE/'gt_dependency'
def read(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    marks=read(OUT/'landmarks.csv');frames=read(OUT/'frame_dependencies.csv')
    online=read(BASE/'online_precursor/landmarks.csv')
    lookup={(r['case_id'],r['time_sec']):r for r in marks}
    by=defaultdict(list)
    for r in frames:by[(r['case_id'],r['landmark_sec'])].append(r)
    scoped={}
    for field in ('evidence_any','evidence_two','odds_any','odds_two'):
        # For minimum FPR: all flagged alerts become events; all flagged nonalerts become nonevents.
        fixed_fp=fixed_tn=fixed_tp=fixed_fn=uncertain_alert=uncertain_noalert=0
        for r in online:
            flagged=int(lookup[(r['case_id'],r['time_sec'])]['onset_unsupported_endpoint_frames'])>0
            alert=r[field]=='True';event=r['sustained']=='True'
            if flagged:
                uncertain_alert+=alert;uncertain_noalert+=not alert
            elif not event:
                fixed_fp+=alert;fixed_tn+=not alert
            else:
                fixed_tp+=alert;fixed_fn+=not alert
        den=fixed_fp+fixed_tn+uncertain_noalert
        ranges={k:[] for k in ('fpr','sensitivity','ppv')}
        for alert_events in range(uncertain_alert+1):
            for nonalert_events in range(uncertain_noalert+1):
                tp=fixed_tp+alert_events;fp=fixed_fp+uncertain_alert-alert_events
                fn=fixed_fn+nonalert_events;tn=fixed_tn+uncertain_noalert-nonalert_events
                for metric,num,d in [('fpr',fp,fp+tn),('sensitivity',tp,tp+fn),('ppv',tp,tp+fp)]:
                    if d:ranges[metric].append(num/d)
        scoped[field]=dict(fixed_false_positives=fixed_fp,fixed_true_negatives=fixed_tn,
            flagged_alerts=uncertain_alert,flagged_nonalerts=uncertain_noalert,
            optimistic_fpr=fixed_fp/den if den else None,
            marginal_ranges={k:[min(v),max(v)] if v else None for k,v in ranges.items()})
    results=[];details=[]
    for eps in (0.,.05,.1,.25,.5):
        counts=Counter()
        for r in marks:
            ff=sorted(by[(r['case_id'],r['time_sec'])],key=lambda f:int(f['frame']))
            err=np.array([float(f['original_error_score_sec']) for f in ff]);lo=np.maximum(err-eps,0);hi=err+eps
            # Dependency range starts at start-21; original healthy history begins at offset1.
            healthy=np.all(hi[1:21]<=.5);unhealthy=np.any(lo[1:21]>.5)
            exposure_clean=np.all(hi[21:41]<=.5);exposure_bad=np.any(lo[21:41]>.5)
            short=sustained_bounds(lo,hi,21,41)
            # Original post-exposure search j=start+20 through start+65 inclusive.
            post=sustained_bounds(lo,hi,41,87)
            counts['healthy_definite' if healthy else 'unhealthy_definite' if unhealthy else 'health_ambiguous']+=1
            if healthy:
                counts['online_event_definite']+=short['definite_event'];counts['online_event_possible']+=short['possible_event']
            if healthy and exposure_clean:
                counts['current_riskset_definite']+=1
                counts['current_event_definite']+=post['definite_event'];counts['current_event_possible']+=post['possible_event']
            if eps==0:
                status='pre_unhealthy' if unhealthy else 'exposure_crossing' if exposure_bad else 'new_failure' if post['definite_event'] else 'no_failure'
                assert status==r['status'],(r['case_id'],r['time_sec'],status,r['status'])
                if healthy:assert short['definite_event']==(r['online_original_event']=='True')
            details.append(dict(case_id=r['case_id'],time_sec=r['time_sec'],epsilon_score_sec=eps,
                definite_healthy=bool(healthy),definite_unhealthy=bool(unhealthy),definite_exposure_clean=bool(exposure_clean),
                online_definite=short['definite_event'],online_possible=short['possible_event'],
                current_definite=post['definite_event'],current_possible=post['possible_event']))
        results.append(dict(epsilon_score_sec=eps,counts=dict(counts)))
    assert results[0]['counts']['healthy_definite']==396
    assert results[0]['counts']['online_event_definite']==7
    assert results[0]['counts']['current_riskset_definite']==371
    assert results[0]['counts']['current_event_definite']==22
    output=dict(scope='Development only. Fixed alerts/risk set for scoped FPR; uniform epsilon is hypothetical, not a certified bound.',
        scoped_onset_unsupported_label_scenario=scoped,uniform_position_scenarios=results,zero_epsilon_equivalence='480 statuses, 396 short labels, 371 post-exposure labels passed')
    output['provenance']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'P3_GT_SENSITIVITY_PROTOCOL.md',OUT/'landmarks.csv',OUT/'frame_dependencies.csv',ROOT/'src/score_following/interval_gt.py')}
    (OUT/'sensitivity.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    with (OUT/'sensitivity_landmarks.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
