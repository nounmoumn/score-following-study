#!/usr/bin/env python3
"""One-shot natural-timing test of the frozen correction-burst alternative."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,yaml
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fp_forward_projection as fp
import p3_fci_tight_envelope_diagnostic as tight
import p3_fhl_batik_transfer as cohort
import p3_fhlr_batik_repair_transfer as repaired
import p3_k_correction_deficit as stats
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

OUT=ROOT/'results/p3_fhb_correction_burst_transfer';RADIUS=.5211881394162992;WINDOW_SEC=12.;HORIZON_SEC=.5;LAG_SEC=.5;CAP=5.;STEP_SEC=.5;SEED=20260921;PERMUTATIONS=2000
def median_state(p):return int(np.searchsorted(np.cumsum(p),.5,side='left'))
def fast_auc(event,none):
 event=np.asarray(event,dtype=float);none=np.asarray(none,dtype=float)
 if not len(event) or not len(none):return float('nan')
 values=np.r_[event,none];ranks=rankdata(values,method='average')
 return float((ranks[:len(event)].sum()-len(event)*(len(event)+1)/2)/(len(event)*len(none)))
def fast_cluster_bootstrap(rows,rng,draws=20000):
 tracks=sorted({r['track'] for r in rows});groups={t:[r for r in rows if r['track']==t] for t in tracks};n=len(tracks);pos=np.asarray([sum(r['event'] for r in groups[t]) for t in tracks],dtype=float);neg=np.asarray([sum(not r['event'] for r in groups[t]) for t in tracks],dtype=float);pair=np.zeros((n,n))
 for i,a in enumerate(tracks):
  ev=[r['deficit'] for r in groups[a] if r['event']]
  for j,b in enumerate(tracks):
   no=[r['deficit'] for r in groups[b] if not r['event']]
   pair[i,j]=fast_auc(ev,no)*len(ev)*len(no) if ev and no else 0.
 out=[]
 for _ in range(draws):
  c=np.bincount(rng.integers(0,n,n),minlength=n).astype(float);den=(c@pos)*(c@neg)
  if den:out.append(float(c@pair@c/den))
 return [float(np.percentile(out,2.5)),float(np.percentile(out,97.5))]
def fast_shift_auc(rows,rng):
 shifted=[]
 for track in sorted({r['track'] for r in rows}):
  group=[r for r in rows if r['track']==track];values=np.asarray([r['deficit'] for r in group]);values=np.roll(values,int(rng.integers(1,len(values)))) if len(values)>1 else values
  shifted.extend((float(v),r['event']) for v,r in zip(values,group))
 return fast_auc([v for v,e in shifted if e],[v for v,e in shifted if not e])

def main():
 manifest=json.loads(cohort.MANIFEST.read_text());cases=[r for r in manifest['rows'] if r.get('exclusion_reason') in ('development family','ASAP overlap')]
 if len(cases)!=9:raise SystemExit('frozen nine-movement cohort changed')
 cfg=yaml.safe_load((ROOT/'configs/p3_temporal.yaml').read_text());hop=float(cfg['hop_sec']);weights=tuple(cfg['bayes']['transition']);future=int(round(HORIZON_SEC/hop));window=int(round(WINDOW_SEC/hop));rows=[]
 for n,case in enumerate(cases,1):
  u,v,sh,truth,duration=repaired.repaired_load(case,hop);path,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);error=np.abs(path-truth)*sh;raw={}
  for t in np.arange(max(WINDOW_SEC,15.),duration-HORIZON_SEC-.1+1e-9,STEP_SEC):
   end=int(round(t/hop));ep=end+future
   if ep>=len(trace) or not healthy_before(error,end-1,int(round(2/hop)),.5):continue
   p=trace[end-1].copy()
   for _ in range(future+1):p=dc.apply_prediction(p,weights)
   center=median_state(p);z=fp.projected_action(trace,weights,end-window,end,future+1);low,_=tight.help_envelope(z,center,RADIUS,sh,CAP);onset=sustained_start(error,end,ep);raw[round(float(t),6)]={'low':low,'event':bool(onset is not None),'onset_sec':None if onset is None else float(onset*hop)}
  track=f"{case['sonata']}-{case['movement']}"
  for t,r in raw.items():
   old=raw.get(round(t-LAG_SEC,6))
   if old is not None:rows.append({'track':track,'time_sec':t,'deficit':float(r['low']-old['low']),'event':r['event'],'onset_sec':r['onset_sec']})
  print(f'[{n}/9] {track}',flush=True)
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'temporal_rows.json').write_text(json.dumps(rows),encoding='utf8')
 event=[r['deficit'] for r in rows if r['event']];none=[r['deficit'] for r in rows if not r['event']];auc=fast_auc(event,none);rng=np.random.default_rng(SEED);ci=fast_cluster_bootstrap(rows,rng);null=np.asarray([fast_shift_auc(rows,rng) for _ in range(PERMUTATIONS)]);null=null[np.isfinite(null)];p=float((1+np.sum(null>=auc))/(1+len(null)));onsets={(r['track'],r['onset_sec']) for r in rows if r['event']};event_tracks={r['track'] for r in rows if r['event']};tracks={r['track'] for r in rows}
 gates={'nine_movements':len(tracks)==9,'cases_ge_100':len(rows)>=100,'events_ge_5_in_3_movements':len(onsets)>=5 and len(event_tracks)>=3,'event_median_gt_nonevent':bool(event and np.median(event)>np.median(none)),'auc_ge_0_70':auc>=.70,'bootstrap_lower_gt_0_55':ci[0]>.55,'circular_shift_p_le_0_05':p<=.05}
 result={'hypothesis':'P3-FHB abrupt correction-help burst precedes 0.5-second natural drift','counts':{'movements':len(tracks),'cases':len(rows),'positive_cases':len(event),'unique_onsets':len(onsets),'event_movements':len(event_tracks)},'metrics':{'event_median':float(np.median(event)) if event else None,'nonevent_median':float(np.median(none)),'rank_auc':auc,'movement_bootstrap_auc_95':ci,'circular_shift_p':p},'gates':{k:bool(v) for k,v in gates.items()},'qualified':bool(all(gates.values())),'repaired_11_temporal_test_authorized':bool(all(gates.values())),'original_natural_p3_confirmed':False}
 (OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
