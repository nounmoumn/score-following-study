#!/usr/bin/env python3
"""KRAISLER development test of a past-only relative help-drop signal."""
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

OUT=ROOT/'results/p3_fhd_help_drop_development';RADIUS=.5211881394162992;WINDOW_SEC=12.;HORIZON_SEC=.5;BASELINE_SEC=20.;CAP=5.;STEP_SEC=.5;SEED=20260921

def median_state(p):return int(np.searchsorted(np.cumsum(p),.5,side='left'))

def main():
 from p3_kraisler_signal_run import load_case
 from p3_batik_external_run import load_gt
 m=json.loads((ROOT/'results/p3_kraisler_signal/manifest.json').read_text());cases=[r for r in m['rows'] if r['eligible']]
 cfg=yaml.safe_load((ROOT/'configs/p3_temporal.yaml').read_text());hop=float(cfg['hop_sec']);weights=tuple(cfg['bayes']['transition']);future=int(round(HORIZON_SEC/hop));window=int(round(WINDOW_SEC/hop));baseline_frames=int(round(BASELINE_SEC/STEP_SEC));rows=[]
 for n,case in enumerate(cases,1):
  u,v,sh,duration=load_case(case,hop);beats,pt=load_gt(ROOT/case['match']);truth=np.interp(np.arange(len(u))*hop,pt,beats*.5)/sh;path,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);error=np.abs(path-truth)*sh;raw=[]
  for time_sec in np.arange(max(WINDOW_SEC,15.),duration-HORIZON_SEC-.1+1e-9,STEP_SEC):
   end=int(round(time_sec/hop));ep=end+future
   if ep>=len(trace) or not healthy_before(error,end-1,int(round(2/hop)),.5):continue
   p=trace[end-1].copy()
   for _ in range(future+1):p=dc.apply_prediction(p,weights)
   center=median_state(p);z=fp.projected_action(trace,weights,end-window,end,future+1);low,high=tight.help_envelope(z,center,RADIUS,sh,CAP);onset=sustained_start(error,end,ep)
   raw.append({'track':case['track'],'time_sec':float(time_sec),'low_help':low,'certified':bool(low>0 or high<0),'event':bool(onset is not None),'onset_sec':None if onset is None else float(onset*hop)})
  for i,r in enumerate(raw):
   previous=[x['low_help'] for x in raw[max(0,i-baseline_frames):i] if 0<r['time_sec']-x['time_sec']<=BASELINE_SEC]
   if len(previous)<20:continue
   rows.append(r|{'deficit_rank':float(np.mean(np.asarray(previous)>r['low_help'])),'baseline_count':len(previous)})
  print(f'[{n}/{len(cases)}] {case["track"]}',flush=True)
 event=[r['deficit_rank'] for r in rows if r['event']];none=[r['deficit_rank'] for r in rows if not r['event']];auc=stats.rank_auc_safe(event,none);rng=np.random.default_rng(SEED);stat_rows=[r|{'deficit':r['deficit_rank']} for r in rows];ci=stats.cluster_bootstrap_auc(stat_rows,rng);null=np.asarray([stats.shifted_auc(stat_rows,rng) for _ in range(stats.PERMUTATIONS)]);null=null[np.isfinite(null)];p=float((1+np.sum(null>=auc))/(1+len(null)));onsets={(r['track'],r['onset_sec']) for r in rows if r['event']};tracks={r['track'] for r in rows if r['event']}
 gates={'coverage':len(rows)>=100 and len(onsets)>=5 and len(tracks)>=5,'event_median_gt_nonevent':np.median(event)>np.median(none) if event else False,'auc_ge_0_70':auc>=.70,'bootstrap_lower_gt_0_55':ci[0]>.55,'circular_shift_p_le_0_05':p<=.05}
 result={'hypothesis':'within-performance online help drop precedes 0.5-second natural drift','counts':{'tracks':len({r['track'] for r in rows}),'cases':len(rows),'positive_cases':len(event),'unique_onsets':len(onsets),'event_tracks':len(tracks)},'metrics':{'event_median':float(np.median(event)) if event else None,'nonevent_median':float(np.median(none)),'rank_auc':auc,'track_bootstrap_auc_95':ci,'circular_shift_p':p,'certificate_share':float(np.mean([r['certified'] for r in rows]))},'gates':{k:bool(v) for k,v in gates.items()},'development_pass':bool(all(gates.values())),'fresh_temporal_evaluation_authorized':bool(all(gates.values())),'original_natural_p3_confirmed':False}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
