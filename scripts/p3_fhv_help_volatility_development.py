#!/usr/bin/env python3
"""Cross-corpus development of direction-free online help volatility."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fp_forward_projection as fp
import p3_fci_tight_envelope_diagnostic as tight
import p3_fhb_correction_burst_transfer as fast
from p3_short_signal import sustained_start
from score_following.temporal import healthy_before

OUT=ROOT/'results/p3_fhv_help_volatility_development';RADIUS=.5211881394162992;WINDOW_SEC=12.;HORIZON_SEC=.5;LAG_SEC=.5;CAP=5.;STEP_SEC=.5;SEED=20260921;PERMUTATIONS=2000
def median_state(p):return int(np.searchsorted(np.cumsum(p),.5,side='left'))

def kraisler_rows():
 from p3_kraisler_signal_run import load_case
 from p3_batik_external_run import load_gt
 m=json.loads((ROOT/'results/p3_kraisler_signal/manifest.json').read_text());cases=[r for r in m['rows'] if r['eligible']];cfg=yaml.safe_load((ROOT/'configs/p3_temporal.yaml').read_text());hop=float(cfg['hop_sec']);weights=tuple(cfg['bayes']['transition']);future=int(round(HORIZON_SEC/hop));window=int(round(WINDOW_SEC/hop));rows=[]
 for n,case in enumerate(cases,1):
  u,v,sh,duration=load_case(case,hop);beats,pt=load_gt(ROOT/case['match']);truth=np.interp(np.arange(len(u))*hop,pt,beats*.5)/sh;path,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);error=np.abs(path-truth)*sh;raw={}
  for t in np.arange(max(WINDOW_SEC,15.),duration-HORIZON_SEC-.1+1e-9,STEP_SEC):
   end=int(round(t/hop));ep=end+future
   if ep>=len(trace) or not healthy_before(error,end-1,int(round(2/hop)),.5):continue
   p=trace[end-1].copy()
   for _ in range(future+1):p=dc.apply_prediction(p,weights)
   center=median_state(p);z=fp.projected_action(trace,weights,end-window,end,future+1);low,_=tight.help_envelope(z,center,RADIUS,sh,CAP);onset=sustained_start(error,end,ep);raw[round(float(t),6)]={'low':low,'event':bool(onset is not None),'onset_sec':None if onset is None else float(onset*hop)}
  for t,r in raw.items():
   old=raw.get(round(t-LAG_SEC,6))
   if old is not None:rows.append({'track':case['track'],'time_sec':t,'deficit':abs(float(r['low']-old['low'])),'event':r['event'],'onset_sec':r['onset_sec']})
  print(f'[KRAISLER {n}/{len(cases)}] {case["track"]}',flush=True)
 return rows

def evaluate(name,rows,seed):
 event=[r['deficit'] for r in rows if r['event']];none=[r['deficit'] for r in rows if not r['event']];auc=fast.fast_auc(event,none);rng=np.random.default_rng(seed);ci=fast.fast_cluster_bootstrap(rows,rng);null=np.asarray([fast.fast_shift_auc(rows,rng) for _ in range(PERMUTATIONS)]);p=float((1+np.sum(null>=auc))/(1+len(null)));onsets={(r['track'],r['onset_sec']) for r in rows if r['event']};units={r['track'] for r in rows if r['event']}
 gates={'coverage':len(rows)>=100 and len(onsets)>=5 and len(units)>=3,'event_median_gt_nonevent':bool(event and np.median(event)>np.median(none)),'auc_ge_0_65':auc>=.65,'bootstrap_lower_gt_0_55':ci[0]>.55,'circular_shift_p_le_0_05':p<=.05}
 return {'corpus':name,'counts':{'cases':len(rows),'positive_cases':len(event),'unique_onsets':len(onsets),'event_units':len(units)},'metrics':{'event_median':float(np.median(event)) if event else None,'nonevent_median':float(np.median(none)),'auc':auc,'cluster_bootstrap_auc_95':ci,'circular_shift_p':p},'gates':{k:bool(v) for k,v in gates.items()},'passed':bool(all(gates.values()))}

def main():
 k=kraisler_rows();b=json.loads((ROOT/'results/p3_fhb_correction_burst_transfer/temporal_rows.json').read_text());b=[r|{'deficit':abs(float(r['deficit']))} for r in b];OUT.mkdir(parents=True,exist_ok=True);(OUT/'development_rows.json').write_text(json.dumps({'KRAISLER':k,'Batik':b}),encoding='utf8');corpora=[evaluate('KRAISLER',k,SEED),evaluate('Batik',b,SEED+1)];passed=all(x['passed'] for x in corpora);result={'hypothesis':'direction-free abrupt observation-help volatility precedes natural drift','corpora':corpora,'development_pass':passed,'repaired_11_temporal_test_authorized':passed,'original_natural_p3_confirmed':False};(OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
