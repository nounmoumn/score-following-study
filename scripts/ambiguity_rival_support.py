"""Opened-development candidate support diagnostic; no correction policy."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from score_following.anchor_feasibility import smooth
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import cosine_cost


def select(values, center):
    values = values.copy()
    values[max(0, center-20):center+21] = -np.inf
    chosen = []
    for _ in range(5):
        i = int(np.argmax(values))
        if not np.isfinite(values[i]):
            break
        chosen.append(i)
        values[max(0, i-20):i+21] = -np.inf
    return chosen


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def main():
    out = ROOT / 'results/ambiguity_rival_support'
    if out.exists():
        raise SystemExit('Refusing overwrite')
    prior = ROOT / 'results/ambiguity_anchor_feasibility_v1'
    hashes = json.loads((prior/'freeze.json').read_text())['files']
    for p in (Path(__file__), ROOT/'AMBIGUITY_RIVAL_SUPPORT_PROTOCOL.md'):
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    out.mkdir()
    write(out/'freeze.json', hashes)
    write(out/'state.json', {'status':'RUNNING'})
    ann = json.loads((ROOT/'asap_dataset/asap_annotations.json').read_text())
    with (ROOT/'output/score_priority_plan/cohort_manifest.csv').open(encoding='utf-8-sig') as f:
        cases = [r for r in csv.DictReader(f) if r['role']=='primary_opened']
    rows, cache = [], {}
    rng = np.random.default_rng(20260923)
    with threadpool_limits(limits=2):
        for case in cases:
            if case['score'] not in cache:
                sn = load_notes(ROOT/'asap_dataset'/case['score'])
                v = notes_to_chroma_stream(sn,max(n[1] for n in sn),.1)
                f = np.array([v[i:i+20].ravel() for i in range(len(v)-20)])
                f /= np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-300)
                cache[case['score']] = v,f
            v,f = cache[case['score']]
            pn = load_notes(ROOT/'asap_dataset'/case['performance'])
            frames = int(np.floor(min(90.,max(n[1] for n in pn))/.1))
            cost = cosine_cost(notes_to_chroma_stream(pn,frames*.1,.1)[:frames],v)
            posterior = smooth(cost)
            gt = ann[case['performance']]
            ts, ys = np.array(gt['performance_beats']),np.array(gt['midi_score_beats'])
            keep = (ts>=15)&(ts<=85)&(abs(ts[:,None]-np.arange(15,86)).min(1)>.11)
            local = []
            for t,y in zip(ts[keep],ys[keep]):
                frame = np.searchsorted((np.arange(frames)+1)*.1,t,side='right')-1
                p = posterior[frame]
                i = int(p.argmax())
                choices = {'score':[], 'posterior':[], 'shuffled_score':[]}
                if i < len(f):
                    sims = f @ f[i]
                    choices = {'score':select(sims,i), 'posterior':select(p[:len(f)],i),
                               'shuffled_score':select(rng.permutation(sims),i)}
                row = {'case':case['case_id'],'work':case['work_family'],
                       'time':float(t),'truth':float(y),'map_state':i,
                       'correct':bool(abs((i+1)*.1-y)<=.5+1e-12),
                       'confidence':float(p[max(0,i-5):i+6].sum()),
                       'window_available':i<len(f),'candidates':choices}
                for name, selected in choices.items():
                    for k in (1,5):
                        row[name+str(k)] = any(abs((j+1)*.1-y)<=.5+1e-12 for j in selected[:k])
                local.append(row)
            previous = json.loads((prior/(case['case_id']+'.json')).read_text())
            assert len(local)==previous['evaluation_rows']
            assert abs(np.mean([r['correct'] for r in local])-previous['baseline_accuracy'])<1e-12
            rows.extend(local)
            write(out/(case['case_id']+'.json'),local)
            print(case['case_id'], 'rows',len(local),flush=True)
    groups = {}
    for name, group in [('all',rows),('wrong',[r for r in rows if not r['correct']]),
                        ('confident_wrong',[r for r in rows if not r['correct'] and r['confidence']>=.95])]:
        groups[name] = {'rows':len(group),'works':len({r['work'] for r in group}),
                       'window_unavailable':sum(not r['window_available'] for r in group)}
        for method in ('score','posterior','shuffled_score'):
            for k in (1,5):
                groups[name][method+str(k)] = sum(r[method+str(k)] for r in group)
        groups[name]['score5_only'] = sum(r['score5'] and not r['posterior5'] for r in group)
        groups[name]['posterior5_only'] = sum(r['posterior5'] and not r['score5'] for r in group)
    by_work = {}
    for work in sorted({r['work'] for r in rows}):
        group = [r for r in rows if r['work']==work and not r['correct']]
        by_work[work] = {'wrong_rows':len(group), 'score5_only':sum(r['score5'] and not r['posterior5'] for r in group),
                        'posterior5_only':sum(r['posterior5'] and not r['score5'] for r in group)}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    result = {'scope':'opened-development oracle candidate coverage, not correction',
              'groups':groups,'by_work':by_work,'baseline_reproduction':'PASS','hash_drift':[]}
    write(out/'result.json',result)
    write(out/'state.json',{'status':'COMPLETE'})
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    main()
