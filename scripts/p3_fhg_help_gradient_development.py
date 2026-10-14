#!/usr/bin/env python3
"""Multiplicity-controlled development test for abrupt observation-help loss."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fp_forward_projection as fp
import p3_fci_tight_envelope_diagnostic as tight
import p3_k_correction_deficit as stats
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

OUT=ROOT/'results/p3_fhg_help_gradient_development';RADIUS=.5211881394162992;WINDOW_SEC=12.;HORIZON_SEC=.5;CAP=5.;STEP_SEC=.5;LAGS=(.5,1.,2.,4.);SEED=20260921;PERMUTATIONS=2000
def median_state(p):return int(np.searchsorted(np.cumsum(p),.5,side='left'))
def auc(rows):return stats.rank_auc_safe([r['deficit'] for r in rows if r['event']],[r['deficit'] for r in rows if not r['event']])

def main():
 from p3_kraisler_signal_run import load_case
 from p3_batik_external_run import load_gt
 m=json.loads((ROOT/'results/p3_kraisler_signal/manifest.json').read_text());cases=[r for r in m['rows'] if r['eligible']]
 cfg=yaml.safe_load((ROOT/'configs/p3_temporal.yaml').read_text());hop=float(cfg['hop_sec']);weights=tuple(cfg['bayes']['transition']);future=int(round(HORIZON_SEC/hop));window=int(round(WINDOW_SEC/hop));raw=[]
 for n,case in enumerate(cases,1):
  u,v,sh,duration=load_case(case,hop);beats,pt=load_gt(ROOT/case['match']);truth=np.interp(np.arange(len(u))*hop,pt,beats*.5)/sh;path,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);error=np.abs(path-truth)*sh
  for t in np.arange(max(WINDOW_SEC,15.),duration-HORIZON_SEC-.1+1e-9,STEP_SEC):
   end=int(round(t/hop));ep=end+future
   if ep>=len(trace) or not healthy_before(error,end-1,int(round(2/hop)),.5):continue
   p=trace[end-1].copy()
   for _ in range(future+1):p=dc.apply_prediction(p,weights)
   center=median_state(p);z=fp.projected_action(trace,weights,end-window,end,future+1);low,_=tight.help_envelope(z,center,RADIUS,sh,CAP);onset=sustained_start(error,end,ep)
   raw.append({'track':case['track'],'time_sec':round(float(t),6),'low_help':low,'event':bool(onset is not None),'onset_sec':None if onset is None else float(onset*hop)})
  print(f'[{n}/{len(cases)}] {case["track"]}',flush=True)
 by_track={track:{r['time_sec']:r for r in raw if r['track']==track} for track in sorted({r['track'] for r in raw})};candidates={}
 for lag in LAGS:
  rows=[]
  for track,mapping in by_track.items():
   for t,r in mapping.items():
    old=mapping.get(round(t-lag,6))
    if old is not None:rows.append({'track':track,'time_sec':t,'event':r['event'],'onset_sec':r['onset_sec'],'deficit':float(old['low_help']-r['low_help'])})
  candidates[lag]=rows
 metrics=[]
 for lag,rows in candidates.items():
  value=auc(rows);track_auc={track:auc([r for r in rows if r['track']==track]) for track in sorted({r['track'] for r in rows})};valid=[v for v in track_auc.values() if np.isfinite(v)]
  metrics.append({'lag_sec':lag,'cases':len(rows),'positive_cases':sum(r['event'] for r in rows),'auc':value,'event_track_aucs':track_auc,'median_event_track_auc':float(np.median(valid)) if valid else None,'event_track_auc_gt_half_share':float(np.mean(np.asarray(valid)>.5)) if valid else None})
 selected=sorted(metrics,key=lambda x:(-x['auc'],x['lag_sec']))[0];rows=candidates[selected['lag_sec']];rng=np.random.default_rng(SEED);ci=stats.cluster_bootstrap_auc(rows,rng)
 null=[]
 for _ in range(PERMUTATIONS):
  draw=[]
  for lag in LAGS:
   shifted=[]
   for track in sorted({r['track'] for r in candidates[lag]}):
    group=[dict(r) for r in candidates[lag] if r['track']==track];values=np.asarray([r['deficit'] for r in group]);values=np.roll(values,int(rng.integers(1,len(values)))) if len(values)>1 else values
    for r,value in zip(group,values):r['deficit']=float(value)
    shifted.extend(group)
   draw.append(auc(shifted))
  null.append(np.nanmax(draw))
 null=np.asarray(null);max_p=float((1+np.sum(null>=selected['auc']))/(1+len(null)));onsets={(r['track'],r['onset_sec']) for r in rows if r['event']};event_tracks={r['track'] for r in rows if r['event']}
 gates={'coverage':len(rows)>=100 and len(onsets)>=5 and len(event_tracks)>=5,'auc_ge_0_70':selected['auc']>=.70,'bootstrap_lower_gt_0_55':ci[0]>.55,'max_stat_p_le_0_05':max_p<=.05,'median_event_track_auc_ge_0_60':selected['median_event_track_auc'] is not None and selected['median_event_track_auc']>=.60,'event_track_direction_share_ge_0_70':selected['event_track_auc_gt_half_share'] is not None and selected['event_track_auc_gt_half_share']>=.70}
 result={'hypothesis':'abrupt drop in certified near-future observation help precedes natural drift','candidates':metrics,'selected_lag_sec':selected['lag_sec'],'counts':{'cases':len(rows),'unique_onsets':len(onsets),'event_tracks':len(event_tracks)},'selected_metrics':{'auc':selected['auc'],'track_bootstrap_auc_95':ci,'max_stat_circular_shift_p':max_p,'median_event_track_auc':selected['median_event_track_auc'],'event_track_auc_gt_half_share':selected['event_track_auc_gt_half_share']},'gates':{k:bool(v) for k,v in gates.items()},'development_pass':bool(all(gates.values())),'fresh_temporal_evaluation_authorized':bool(all(gates.values())),'original_natural_p3_confirmed':False}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
