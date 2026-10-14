"""Multi-axis frozen diagnostic on opened development, never independent proof."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from score_following.features import load_notes,notes_to_chroma_stream
from score_following.anchor_feasibility import smooth
from score_following.p3_extension import cosine_cost
from score_following.contrast_probe import piano_roll,normalize,window,weights,choose


def write(p,x):
    p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')


def main():
    out=ROOT/'results/ambiguity_contrast_v1'
    if out.exists(): raise SystemExit('Refusing overwrite')
    prior=ROOT/'results/ambiguity_rival_support'
    hashes=json.loads((prior/'freeze.json').read_text())
    files=[Path(__file__),ROOT/'AMBIGUITY_CONTRAST_V1_PROTOCOL.md',ROOT/'src/score_following/contrast_probe.py',ROOT/'tests/test_contrast_probe.py']+list(prior.glob('c[0-9][0-9][0-9].json'))
    for p in files: hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    out.mkdir(); write(out/'freeze.json',hashes); write(out/'state.json',{'status':'RUNNING'})
    with (ROOT/'output/score_priority_plan/cohort_manifest.csv').open(encoding='utf-8-sig') as f:
        cases=[r for r in csv.DictReader(f) if r['role']=='primary_opened']
    rng=np.random.default_rng(20260923)
    records=[]; cache={}
    variants={'primary':20,'short':16,'long':25,'drop10':20}
    with threadpool_limits(limits=2):
        for case in cases:
            rows=json.loads((prior/(case['case_id']+'.json')).read_text())
            if case['score'] not in cache:
                sn=load_notes(ROOT/'asap_dataset'/case['score']); duration=max(n[1] for n in sn)
                sr=piano_roll(sn,duration); sc=notes_to_chroma_stream(sn,duration,.1)
                cache[case['score']]=sr,sc
            sr,sc=cache[case['score']]
            pn=load_notes(ROOT/'asap_dataset'/case['performance'])
            frames=int(np.floor(min(90.,max(n[1] for n in pn))/.1))
            pr=piano_roll(pn,frames*.1)[:frames]; pc=notes_to_chroma_stream(pn,frames*.1,.1)[:frames]
            dropped=[n for n in pn if rng.random()>=.1]
            dr=piano_roll(dropped,frames*.1)[:frames]; dc=notes_to_chroma_stream(dropped,frames*.1,.1)[:frames]
            full=smooth(cosine_cost(normalize(pr),normalize(sr))).argmax(1)
            baseline=smooth(cosine_cost(pc,sc)).argmax(1)
            data=[]
            for r in rows:
                f=np.searchsorted((np.arange(frames)+1)*.1,r['time'],side='right')-1
                assert int(baseline[f])==r['map_state']
                candidates=list(dict.fromkeys([r['map_state']]+r['candidates']['score']+r['candidates']['posterior']))
                ts=np.stack([window(sr,j) for j in candidates]); tc=np.stack([window(sc,j) for j in candidates])
                w=weights(ts); sham=w[rng.permutation(128)]
                correct=lambda j: bool(abs((j+1)*.1-r['truth'])<=.5+1e-12)
                item={'time':r['time'],'truth':r['truth'],'base':r['correct'],'pitch_full':correct(int(full[f])),
                      'candidate_truth_present':any(correct(j) for j in candidates),'conditions':{},'predictions':{}}
                for variant,width in variants.items():
                    obs=window(dr if variant=='drop10' else pr,f,width)
                    chroma=window(dc if variant=='drop10' else pc,f,width)
                    choices={'chroma':choose(tc,chroma),'pitch':choose(ts,obs),
                             'contrast':choose(ts,obs,w),'sham':choose(ts,obs,sham)}
                    for method,index in choices.items():
                        key=variant+'_'+method
                        item['predictions'][key]=candidates[index]
                        item['conditions'][key]=correct(candidates[index])
                data.append(item)
            record={'case':case['case_id'],'work':case['work_family'],'rows':data}
            records.append(record);write(out/(case['case_id']+'.json'),record)
            print(case['case_id'],len(data),flush=True)
    worknames=sorted({r['work'] for r in records})
    keys=['base','pitch_full']+list(records[0]['rows'][0]['conditions'])
    get=lambda row,key: row[key] if key in ('base','pitch_full') else row['conditions'][key]
    grouped={k:np.array([np.mean([np.mean([get(row,k) for row in r['rows']]) for r in records if r['work']==w]) for w in worknames]) for k in keys}
    summary={k:float(v.mean()) for k,v in grouped.items()}
    samples=np.random.default_rng(20260923).integers(0,len(worknames),(10000,len(worknames)))
    comparisons={}
    pairs=[('primary_contrast',k) for k in ('primary_pitch','pitch_full','primary_sham','base')]+[(v+'_contrast',v+'_pitch') for v in ('short','long','drop10')]
    for a,b in pairs:
        d=grouped[a]-grouped[b]
        comparisons[a+' minus '+b]={'mean':float(d.mean()),'ci95':np.quantile(d[samples].mean(1),[.025,.975]).tolist(),
                                    'positive_works':int((d>0).sum()),'negative_works':int((d<0).sum())}
    allrows=[row for r in records for row in r['rows']]
    fixed=sum(not r['base'] and r['conditions']['primary_contrast'] for r in allrows)
    broken=sum(r['base'] and not r['conditions']['primary_contrast'] for r in allrows)
    eligible=[r for r in allrows if not r['base'] and r['candidate_truth_present']]
    conditional={k:sum(get(r,k) for r in eligible) for k in keys}
    gate=all(comparisons['primary_contrast minus '+k]['mean']>=.01 and comparisons['primary_contrast minus '+k]['ci95'][0]>0 for k in ('primary_pitch','pitch_full'))
    gate=gate and comparisons['primary_contrast minus primary_sham']['mean']>0 and broken/len(allrows)<=.01
    gate=gate and all(comparisons[v+'_contrast minus '+v+'_pitch']['mean']>=0 for v in ('short','long','drop10'))
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    result={'scope':'opened MIDI diagnostic, not independent or audio confirmation','cases':len(records),'works':len(worknames),
            'rows':len(allrows),'accuracy_equal_work':summary,'comparisons':comparisons,'contrast_fixed':fixed,'contrast_broken':broken,
            'oracle_eligible_wrong_rows':len(eligible),'correct_in_oracle_subset':conditional,'development_gate':bool(gate),'hash_drift':[]}
    write(out/'result.json',result);write(out/'state.json',{'status':'COMPLETE','gate':bool(gate)})
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
