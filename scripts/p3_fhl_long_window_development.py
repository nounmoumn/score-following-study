#!/usr/bin/env python3
"""Post-P3-FH long-window development, still without natural-drift outcomes."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fp_forward_projection as fp
import p3_fci_tight_envelope_diagnostic as tight
import p3_vienna_forecast_validation as vienna
from score_following.temporal import healthy_before

MANIFEST=ROOT/'results/p3_vienna_forecast_preflight/manifest.json';OUT=ROOT/'results/p3_fhl_long_window_development'
WINDOWS=(6.,8.,12.);HORIZONS=(.5,1.);CAP=5.
def higher(x,q):return float(np.percentile(np.asarray(x,dtype=float),q,method='higher'))
def median_state(p):return int(np.searchsorted(np.cumsum(p),.5,side='left'))

def main():
 m=json.loads(MANIFEST.read_text());cfg=yaml.safe_load((ROOT/'configs/p3_temporal.yaml').read_text());hop=float(cfg['hop_sec']);weights=tuple(cfg['bayes']['transition']);cache={};recs=[]
 for n,case in enumerate(m['rows'],1):
  u,v,sh,truth,available=vienna.load_case(case,hop,cache);path,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);err=np.abs(path-truth)*sh
  for hs in HORIZONS:
   f=int(round(hs/hop))
   for t in np.arange(15.,available-hs-.1+1e-9,5.):
    end=int(round(t/hop));ep=end+f
    if ep>=len(trace) or not healthy_before(err,end-1,int(round(2/hop)),.5):continue
    p=trace[end-1].copy()
    for _ in range(f+1):p=dc.apply_prediction(p,weights)
    center=median_state(p);base={'work':case['work'],'stem':case['stem'],'split':case['split'],'horizon_sec':hs,'score_hop':sh,'center':center,'truth':float(truth[ep]),'position_error':abs(center-truth[ep])*sh}
    for ws in WINDOWS:
     z=fp.projected_action(trace,weights,end-int(round(ws/hop)),end,f+1);recs.append(base|{'window_sec':ws,'z':z})
  print(f'[{n}/88] {case["stem"]}',flush=True)
 radii={}
 for hs in HORIZONS:
  cal=[r for r in recs if r['split']=='calibration' and r['horizon_sec']==hs and r['window_sec']==WINDOWS[0]];stems=sorted({r['stem'] for r in cal})
  radii[hs]=higher([higher([r['position_error'] for r in cal if r['stem']==s],95) for s in stems],95)
 cells=[]
 for hs in HORIZONS:
  for ws in WINDOWS:
   radius=radii[hs];rows=[r for r in recs if r['split']=='validation' and r['horizon_sec']==hs and r['window_sec']==ws];ev=[]
   for r in rows:
    covered=r['position_error']<=radius;lo,hi=tight.help_envelope(r['z'],r['center'],radius,r['score_hop'],CAP);cert=lo>0 or hi<0;sgn=1 if lo>0 else -1 if hi<0 else 0
    states=np.arange(len(r['z']));oracle=-float(np.minimum(np.abs(states-r['truth'])*r['score_hop'],CAP)@r['z']);ev.append((r['work'],covered,cert,sgn==np.sign(oracle)))
   works=sorted({x[0] for x in ev});cov={w:float(np.mean([x[1] for x in ev if x[0]==w])) for w in works};cert=[x for x in ev if x[2]];cc=[x for x in cert if x[1]];acc=float(np.mean([x[3] for x in cc])) if cc else None
   gates={'landmarks_ge_200':bool(len(ev)>=200),'work_equal_coverage_ge_0_90':bool(np.mean(list(cov.values()))>=.90),'each_work_coverage_ge_0_85':bool(min(cov.values())>=.85),'certificate_share_ge_0_20':bool(len(cert)/len(ev)>=.20),'covered_sign_accuracy_ge_0_95':bool(acc is not None and acc>=.95),'both_works_certified':bool(len({x[0] for x in cert})==2)}
   cells.append({'window_sec':ws,'horizon_sec':hs,'radius_sec':radius,'landmarks':len(ev),'work_coverage':cov,'certificate_share':len(cert)/len(ev),'certificates':len(cert),'covered_certificates':len(cc),'covered_sign_accuracy':acc,'gates':gates,'candidate':all(gates.values())})
 candidates=[c for c in cells if c['candidate']];selected=sorted(candidates,key=lambda c:(-c['certificate_share'],c['horizon_sec'],c['window_sec']))[0] if candidates else None
 result={'scope':'post-P3-FH opened Vienna development; no natural drift','radii_sec':radii,'cells':cells,'selected':selected,'external_position_sign_validation_authorized':bool(selected),'natural_p3_confirmed':False}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
