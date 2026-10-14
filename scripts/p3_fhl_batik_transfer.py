#!/usr/bin/env python3
"""One-shot Batik transfer of the frozen 12 s -> 0.5 s certificate."""
from __future__ import annotations
import hashlib,json,sys,warnings
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import p3_clock_repair_preflight as repair
import p3_d_dose_response as dose
import p3_dc_conditional_erasure as dc
import p3_fp_forward_projection as fp
import p3_fci_tight_envelope_diagnostic as tight
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.temporal import healthy_before

MANIFEST=ROOT/'results/p3_batik_external/manifest.json';OUT=ROOT/'results/p3_fhl_batik_transfer'
WINDOW_SEC=12.;HORIZON_SEC=.5;RADIUS=.5211881394162992;CAP=5.;FAMILIES=('K.279','K.331','K.332')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def median_state(p):return int(np.searchsorted(np.cumsum(p),.5,side='left'))

def load(case,hop):
 import partitura as pt
 perf=ROOT/case['performance'];score=ROOT/case['score'];match=ROOT/case['match']
 for p,k in ((perf,'performance'),(score,'score'),(match,'match')):
  if sha(p)!=case['sha256'][k]:raise AssertionError(f'hash changed: {p}')
 score_clock,perf_clock=repair.raw_clock(match)
 if not np.all(np.diff(perf_clock)>0):raise AssertionError(f'nonmonotone transfer clock: {match}')
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');a=pt.score.unfold_part_maximal(pt.load_musicxml(str(score))[0],ignore_leaps=False).note_array()
 score_notes=[(float(r['onset_beat'])*.5,float(r['onset_beat']+r['duration_beat'])*.5,int(r['pitch'])) for r in a]
 perf_notes=load_notes(perf);pd=max(x[1] for x in perf_notes);sd=max(x[1] for x in score_notes);sh=hop*sd/pd
 u=notes_to_chroma_stream(perf_notes,pd,hop);v=notes_to_chroma_stream(score_notes,sd,sh)
 truth=np.interp(np.arange(len(u))*hop,perf_clock,score_clock,left=score_clock[0],right=score_clock[-1])/sh
 return u,v,sh,truth,min(pd,float(perf_clock[-1]))

def main():
 m=json.loads(MANIFEST.read_text());cases=[r for r in m['rows'] if r.get('exclusion_reason') in ('development family','ASAP overlap')]
 if len(cases)!=9 or {r['sonata'] for r in cases}!=set(FAMILIES):raise SystemExit('frozen nine-movement cohort changed')
 cfg=yaml.safe_load((ROOT/'configs/p3_temporal.yaml').read_text());hop=float(cfg['hop_sec']);weights=tuple(cfg['bayes']['transition']);future=int(round(HORIZON_SEC/hop));window=int(round(WINDOW_SEC/hop));rows=[]
 for n,case in enumerate(cases,1):
  u,v,sh,truth,duration=load(case,hop);path,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);err=np.abs(path-truth)*sh
  for t in np.arange(15.,duration-HORIZON_SEC-.1+1e-9,5.):
   end=int(round(t/hop));ep=end+future
   if ep>=len(trace) or not healthy_before(err,end-1,int(round(2/hop)),.5):continue
   p=trace[end-1].copy()
   for _ in range(future+1):p=dc.apply_prediction(p,weights)
   center=median_state(p);covered=abs(center-truth[ep])*sh<=RADIUS;z=fp.projected_action(trace,weights,end-window,end,future+1);lo,hi=tight.help_envelope(z,center,RADIUS,sh,CAP);cert=lo>0 or hi<0;sign=1 if lo>0 else -1 if hi<0 else 0
   states=np.arange(len(z));oracle=-float(np.minimum(np.abs(states-truth[ep])*sh,CAP)@z)
   rows.append({'sonata':case['sonata'],'movement':case['movement'],'covered':covered,'certified':cert,'sign':sign,'sign_match':bool(sign==np.sign(oracle))})
  print(f'[{n}/9] {case["sonata"]}-{case["movement"]}',flush=True)
 cov={s:float(np.mean([r['covered'] for r in rows if r['sonata']==s])) for s in FAMILIES};cert=[r for r in rows if r['certified']];cc=[r for r in cert if r['covered']];tracks={(r['sonata'],r['movement']) for r in rows}
 gates={'nine_movements':len(tracks)==9,'landmarks_ge_100':len(rows)>=100,'overall_coverage_ge_0_90':float(np.mean([r['covered'] for r in rows]))>=.90,'each_sonata_coverage_ge_0_85':min(cov.values())>=.85,'certificate_share_ge_0_20':len(cert)/len(rows)>=.20,'covered_sign_accuracy_ge_0_95':float(np.mean([r['sign_match'] for r in cc]))>=.95,'all_sonatas_certified':{r['sonata'] for r in cert}==set(FAMILIES)}
 result={'scope':'frozen position/sign method transfer; natural drift unopened','counts':{'movements':len(tracks),'landmarks':len(rows),'certificates':len(cert),'covered_certificates':len(cc),'positive_certificates':sum(r['sign']>0 for r in cert),'negative_certificates':sum(r['sign']<0 for r in cert)},'position_coverage':float(np.mean([r['covered'] for r in rows])),'sonata_coverage':cov,'certificate_share':len(cert)/len(rows),'covered_sign_accuracy':float(np.mean([r['sign_match'] for r in cc])),'gates':{k:bool(v) for k,v in gates.items()},'qualified':bool(all(gates.values())),'natural_p3_confirmed':False}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
