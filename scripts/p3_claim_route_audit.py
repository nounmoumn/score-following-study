"""Claim audit of all previously reported event windows; no new predictor or fit."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/p3_onset_development';SOURCE=ROOT/'results/p3_temporal_20260918_000058'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    post=read(BASE/'new_failures.csv');short=[r for r in read(BASE/'short_signal/landmarks.csv') if r['sustained']=='True']
    old={(r['case_id'],float(r['time_sec'])):r for r in read(BASE/'failure_route_audit.csv')}
    evidence={(r['case_id'],int(r['frame'])):r for r in read(BASE/'current_position/frames.csv')}
    rows=[];traces=[];hashes={}
    for kind,events in [('post_exposure',post),('short',short)]:
        for r in events:
            c=r['case_id'];j=int(r['failure_frame'] if kind=='post_exposure' else r['onset_frame'])
            mark=int(round(float(r['time_sec'])*10));start=mark+20 if kind=='post_exposure' else mark
            f=SOURCE/'cases'/(c+'.npz');hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
            with np.load(f) as z:path=z['intact_bayes'];truth=z['truth_frames'];hv=float(z['hop_score_sec'])
            steps=np.diff(path[start-1:j+5]);nonlocal_mask=(steps<0)|(steps>3)
            before=nonlocal_mask[:j-start+1]
            error=abs(path-truth)*hv
            assert np.all(error[j:j+5]>.5)
            if kind=='post_exposure':assert bool(nonlocal_mask.any())==(old[(c,float(r['time_sec']))]['nonlocal_MAP_switch']=='True')
            # Fixed 2 s pre-onset inventory; do not infer that missing cached B means evidence absent.
            cached=[(i,evidence[(c,i)]) for i in range(j-20,j) if (c,i) in evidence]
            signed_before=(path[start-1]-truth[start-1])*hv;signed_onset=(path[j]-truth[j])*hv
            row=dict(kind=kind,case_id=c,composition=r['composition'],landmark_sec=float(r['time_sec']),
                onset_frame=j,onset_sec=j*.1,confirmation_available_sec=(j+5)*.1,
                nonlocal_through_confirmation=bool(nonlocal_mask.any()),nonlocal_through_onset=bool(before.any()),
                first_nonlocal_frame=None if not nonlocal_mask.any() else start+int(np.flatnonzero(nonlocal_mask)[0]),
                signed_error_before=float(signed_before),signed_error_at_onset=float(signed_onset),
                tracker_progress=float((path[j]-path[start-1])*hv),score_progress=float((truth[j]-truth[start-1])*hv),
                pre_onset_B_cached_frames=len(cached),pre_onset_B_required_frames=20,
                cached_B_positive_frames=sum(float(x['B'])>0 for _,x in cached),
                interpretation='local-compatible candidate, not certified drift' if not nonlocal_mask.any() else 'candidate switch present, not proof of no drift')
            assert abs((signed_onset-signed_before)-(row['tracker_progress']-row['score_progress']))<1e-8
            rows.append(row)
            for i in range(j-20,j+5):
                traces.append(dict(kind=kind,case_id=c,landmark_sec=float(r['time_sec']),frame=i,
                    available_sec=(i+1)*.1,tracker_score_sec=float(path[i]*hv),annotated_score_sec=float(truth[i]*hv),
                    absolute_error_score_sec=float(error[i]),cached_B=evidence.get((c,i),{}).get('B','')))
    out=BASE/'claim_route';out.mkdir(exist_ok=True)
    for name,data in [('events',rows),('traces',traces)]:
        with (out/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    result={}
    for kind in ('post_exposure','short'):
        rr=[r for r in rows if r['kind']==kind]
        result[kind]=dict(windows=len(rr),nonlocal_through_onset=sum(r['nonlocal_through_onset'] for r in rr),
            nonlocal_through_confirmation=sum(r['nonlocal_through_confirmation'] for r in rr),
            complete_pre_onset_B_windows=sum(r['pre_onset_B_cached_frames']==20 for r in rr),
            local_compatible=[r for r in rr if not r['nonlocal_through_confirmation']])
    result['unique_onsets_union']=len({(r['case_id'],r['onset_frame']) for r in rows})
    result['caveat']='Retrospective diagnostic only. Selection by old outcomes, no independent test. Local MAP steps do not certify latent path continuity. Missing B is unknown.'
    for f in [Path(__file__),BASE/'new_failures.csv',BASE/'short_signal/landmarks.csv',BASE/'current_position/frames.csv',BASE/'failure_route_audit.csv']:
        hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    result['provenance']=hashes
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='provenance'},indent=2))
if __name__=='__main__':main()
