"""One-to-one onset sequence alignment, development-only GT construction candidate."""
import json
import hashlib
from pathlib import Path
import numpy as np
from p3_smd_alignment_pilot import ROOT
from p3_smd_repeat_audit import load

def sequence_align(score,perf,gap=.6):
    n,m=len(score),len(perf)
    d=np.zeros((n+1,m+1));steps=np.zeros((n+1,m+1),dtype=np.uint8)
    d[:,0]=np.arange(n+1)*gap;d[0,:]=np.arange(m+1)*gap
    for i in range(1,n+1):
        a=score[i-1][1]
        for j in range(1,m+1):
            b=perf[j-1][1]
            dist=1-len(a&b)/len(a|b)
            options=(d[i-1,j-1]+dist,d[i-1,j]+gap,d[i,j-1]+gap)
            k=int(np.argmin(options));steps[i,j]=k;d[i,j]=options[k]
    pairs=[];deletions=[];insertions=[];i=n;j=m
    while i or j:
        k=int(steps[i,j]) if i and j else (1 if i else 2)
        if k==0: pairs.append((i-1,j-1));i-=1;j-=1
        elif k==1: deletions.append(i-1);i-=1
        else: insertions.append(j-1);j-=1
    pairs.reverse()
    assert all(a[0]<b[0] and a[1]<b[1] for a,b in zip(pairs,pairs[1:]))
    return pairs,insertions,deletions,float(d[n,m])

def controls():
    s=[(i,{p}) for i,p in enumerate((60,60,60,62,64))]
    expected=list(zip(range(5),range(5)))
    assert sequence_align(s,s)[0]==expected
    assert sequence_align(s,[(i*i+3,p) for i,p in s])[0]==expected
    p=s[:3]+[(2.5,{99})]+s[3:]
    pairs,ins,dele,_=sequence_align(s,p)
    assert ins==[3] and not dele and pairs==[(0,0),(1,1),(2,2),(3,4),(4,5)]
    pairs,ins,dele,_=sequence_align(s,s[:-1])
    assert not ins and dele==[4] and len(pairs)==4
    assert sequence_align([],s)[0]==[] and len(sequence_align([],s)[1])==5
    print('5 known-answer sequence controls passed',flush=True)

def stats(e):
    e=np.abs(e)
    return {'n':len(e),'median':float(np.median(e)),'p95':float(np.percentile(e,95)),
            'max':float(max(e)),'gt_0_1':int(sum(e>.1)),'gt_0_5':int(sum(e>.5))}

def main():
    controls()
    baseout=ROOT/'results/p3_external_metadata'
    baseline_path=baseout/'beethoven_repeat_calibration.json'
    baseline=json.loads(baseline_path.read_text())
    result={'status':'running','cases':[],'baseline_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol_sha256':hashlib.sha256((ROOT/'P3_ONSET_SEQUENCE_PROTOCOL.md').read_bytes()).hexdigest()}
    for case in baseline['cases']:
        folder=ROOT/f"asap_dataset/Beethoven/Piano_Sonatas/{case['variant']}"
        se,_,_=load(folder/'midi_score.mid',threshold=1e-4);pe,_,_=load(folder/f"{case['name']}.mid")
        print('Aligning '+case['name'],flush=True)
        pairs,ins,dele,cost=sequence_align(se,pe)
        times=np.array([pe[j][0] for i,j in pairs]);pos=np.array([se[i][0] for i,j in pairs])
        rows=[];excluded=[]
        for row in case['beat_rows']:
            t=row['performance_sec']
            if t<times[0] or t>times[-1]: excluded.append(row);continue
            pred=float(np.interp(t,times,pos))
            rows.append({**row,'sequence_prediction':pred,'sequence_error':pred-row['true_score_sec']})
        summary={'name':case['name'],'matched_onsets':len(pairs),'insertions':len(ins),'deletions':len(dele),
                 'total_alignment_cost':cost,'excluded_baseline_beats':excluded,
                 'baseline_common':stats([r['error_score_sec'] for r in rows]),
                 'sequence_common':stats([r['sequence_error'] for r in rows]),
                 'previous_large_errors':[r for r in rows if abs(r['error_score_sec'])>.5],
                 'new_large_errors':[r for r in rows if abs(r['error_score_sec'])<=.5 and abs(r['sequence_error'])>.5],
                 'beat_rows':rows,'input_provenance':case['provenance']}
        result['cases'].append(summary)
        np.savez_compressed(baseout/f"{case['name']}_sequence_mapping.npz",performance_onset_sec=times,mapped_score_onset_sec=pos)
        print(json.dumps({k:summary[k] for k in ('name','matched_onsets','insertions','deletions','baseline_common','sequence_common')}),flush=True)
        result['status']='complete' if len(result['cases'])==len(baseline['cases']) else 'running'
        (baseout/'onset_sequence_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
