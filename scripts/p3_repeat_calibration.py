"""Frozen development calibration of existing alignment helper on two repeat variants."""
import hashlib
import json
from pathlib import Path
import numpy as np
from p3_smd_alignment_pilot import ROOT, align
from p3_smd_repeat_audit import load

CASES=(('17-1','KaszoS09M'),('17-1_no_repeat','Giesbrecht02'))
OUT=ROOT/'results/p3_external_metadata/beethoven_repeat_calibration.json'

def main():
    result={'cases':[], 'purpose':'Development beat calibration only, not P3 outcomes or independent validation.'}
    for variant,name in CASES:
        base=ROOT/f'asap_dataset/Beethoven/Piano_Sonatas/{variant}'
        files=[base/'midi_score.mid',base/f'{name}.mid',base/'midi_score_annotations.txt',base/f'{name}_annotations.txt']
        se,_,_=load(files[0],threshold=1e-4)
        pe,_,_=load(files[1])
        print(f'Aligning {variant}/{name}: {len(se)} x {len(pe)}',flush=True)
        cost,mapping,steps=align(se,pe)
        ts=np.loadtxt(files[2],usecols=0)
        tp=np.loadtxt(files[3],usecols=0)
        if len(ts)!=len(tp): raise ValueError('Annotation lengths disagree')
        times=np.array([pe[j][0] for j in sorted(mapping)])
        pos=np.array([se[mapping[j]][0] for j in sorted(mapping)])
        np.savez_compressed(OUT.parent/f'{name}_alignment_mapping.npz',
                            performance_onset_sec=times, mapped_score_onset_sec=pos,
                            score_onset_sec=np.array([x[0] for x in se]))
        valid=(tp>=times[0])&(tp<=times[-1])
        pred=np.interp(tp[valid],times,pos)
        residual=pred-ts[valid]
        case={'variant':variant,'name':name,'mean_path_cost':float(cost),'path_steps':steps,
              'evaluated_beats':int(valid.sum()),'outside_onset_support':int((~valid).sum()),
              'median_abs_score_sec':float(np.median(abs(residual))),
              'p95_abs_score_sec':float(np.percentile(abs(residual),95)),
              'max_abs_score_sec':float(max(abs(residual))),
              'errors_gt_0_1_score_sec':int(sum(abs(residual)>.1)),
              'errors_gt_0_5_score_sec':int(sum(abs(residual)>.5)),
              'beat_rows':[{'performance_sec':float(t),'true_score_sec':float(s),'predicted_score_sec':float(p),'error_score_sec':float(e)}
                           for t,s,p,e in zip(tp[valid],ts[valid],pred,residual)],
              'provenance':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
        result['cases'].append(case)
        print(json.dumps({k:v for k,v in case.items() if k not in ('beat_rows','provenance')}),flush=True)
        result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['completed_cases']=len(result['cases'])
        result['expected_cases']=len(CASES)
        OUT.write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__': main()
