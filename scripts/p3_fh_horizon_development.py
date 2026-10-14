#!/usr/bin/env python3
"""Outcome-free multiscale development of exact online help certificates."""
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

MANIFEST=ROOT/'results/p3_vienna_forecast_preflight/manifest.json'
OUT=ROOT/'results/p3_fh_horizon_development'
WINDOWS=(1.,2.,4.);HORIZONS=(1.,2.,4.,8.);LANDMARK_STEP=5.;CAP=5.

def higher(values,q):return float(np.percentile(np.asarray(values,dtype=float),q,method='higher'))
def median_state(p):return int(np.searchsorted(np.cumsum(p),.5,side='left'))

def main():
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    cfg=yaml.safe_load((ROOT/'configs/p3_temporal.yaml').read_text(encoding='utf8'))
    hop=float(cfg['hop_sec']);weights=tuple(float(x) for x in cfg['bayes']['transition'])
    cache={};records=[]
    for number,case in enumerate(manifest['rows'],1):
        u,v,score_hop,truth,available=vienna.load_case(case,hop,cache)
        path,trace=dose.bayes_trace(dose.cosine_cost(u,v),weights);error=np.abs(path-truth)*score_hop
        for horizon_sec in HORIZONS:
            future=int(round(horizon_sec/hop));
            for time_sec in np.arange(15.,available-horizon_sec-.1+1e-9,LANDMARK_STEP):
                end=int(round(time_sec/hop));endpoint=end+future
                if endpoint>=len(trace) or not healthy_before(error,end-1,int(round(2./hop)),.5):continue
                predicted=trace[end-1].copy()
                for _ in range(future+1):predicted=dc.apply_prediction(predicted,weights)
                center=median_state(predicted);truth_state=float(truth[endpoint])
                common={'work':case['work'],'stem':case['stem'],'split':case['split'],'horizon_sec':horizon_sec,
                        'score_hop':score_hop,'center':center,'truth':truth_state,
                        'position_error':abs(center-truth_state)*score_hop}
                for window_sec in WINDOWS:
                    window=int(round(window_sec/hop));start=end-window
                    if start<1:continue
                    z=fp.projected_action(trace,weights,start,end,future+1)
                    records.append(common|{'window_sec':window_sec,'z':z})
        print(f"[{number}/{len(manifest['rows'])}] {case['stem']}",flush=True)

    radii={}
    for horizon in HORIZONS:
        cal=[r for r in records if r['split']=='calibration' and r['horizon_sec']==horizon and r['window_sec']==WINDOWS[0]]
        perfs=sorted({r['stem'] for r in cal})
        radii[horizon]=higher([higher([r['position_error'] for r in cal if r['stem']==s],95) for s in perfs],95)

    cells=[]
    for horizon in HORIZONS:
      for window in WINDOWS:
        radius=radii[horizon];rows=[r for r in records if r['split']=='validation' and r['horizon_sec']==horizon and r['window_sec']==window]
        evaluated=[]
        for r in rows:
            covered=r['position_error']<=radius
            low,high=tight.help_envelope(r['z'],r['center'],radius,r['score_hop'],CAP)
            certified=low>0 or high<0;sign=1 if low>0 else -1 if high<0 else 0
            states=np.arange(len(r['z']),dtype=float)
            oracle=-float(np.minimum(np.abs(states-r['truth'])*r['score_hop'],CAP)@r['z'])
            evaluated.append((r['work'],covered,certified,bool(sign==np.sign(oracle))))
        works=sorted({x[0] for x in evaluated})
        coverage={w:float(np.mean([x[1] for x in evaluated if x[0]==w])) for w in works}
        cert=[x for x in evaluated if x[2]];covered_cert=[x for x in cert if x[1]]
        cert_works=sorted({x[0] for x in cert})
        acc=float(np.mean([x[3] for x in covered_cert])) if covered_cert else None
        gates={'landmarks_ge_200':len(evaluated)>=200,'work_equal_coverage_ge_0_90':float(np.mean(list(coverage.values())))>=.90,
               'each_work_coverage_ge_0_85':min(coverage.values())>=.85,'certificate_share_ge_0_20':len(cert)/len(evaluated)>=.20,
               'covered_sign_accuracy_ge_0_95':acc is not None and acc>=.95,'both_works_certified':len(cert_works)==len(works)==2}
        cells.append({'window_sec':window,'horizon_sec':horizon,'radius_sec':radius,'landmarks':len(evaluated),
                      'work_coverage':coverage,'work_equal_coverage':float(np.mean(list(coverage.values()))),
                      'certificate_share':len(cert)/len(evaluated),'certificates':len(cert),'covered_certificates':len(covered_cert),
                      'covered_sign_accuracy':acc,'certified_works':cert_works,'gates':gates,'candidate':bool(all(gates.values()))})
    candidates=[c for c in cells if c['candidate']]
    selected=sorted(candidates,key=lambda c:(-c['certificate_share'],c['horizon_sec'],c['window_sec']))[0] if candidates else None
    result={'scope':'opened Vienna development only; no natural-drift outcomes','grid':{'windows_sec':WINDOWS,'horizons_sec':HORIZONS},
            'radii_sec':radii,'cells':cells,'selected':selected,'fresh_transfer_authorized':bool(selected),'natural_p3_confirmed':False}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'radii_sec':radii,'cells':cells,'selected':selected},indent=2))

if __name__=='__main__':main()
