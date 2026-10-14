"""Post-hoc saved-result audit; see DIAGNOSTIC_MAP_PILOT_PROTOCOL.md."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

root = Path(__file__).resolve().parents[1]
out = root / 'results/diagnostic_map_pilot'
out.mkdir(exist_ok=True)
target = out / 'result.json'
if target.exists():
    raise RuntimeError('Refuse overwrite')
paths = sorted((root / 'results/ambiguity_contrast_v1').glob('c[0-9][0-9][0-9].json'))
csvpath = root / 'results/p2_context_utility_feasibility/observations.csv'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {str(p.relative_to(root)): digest(p) for p in paths + [csvpath]}
lookup = {}
for r in csv.DictReader(csvpath.open(encoding='utf-8-sig')):
    key = (r['case_id'], round(float(r['time']), 6))
    assert key not in lookup, key
    lookup[key] = r
groups = {}
n = 0
for p in paths:
    data = json.loads(p.read_text(encoding='utf-8'))
    values = []
    seen = set()
    for r in data['rows']:
        key = (data['case'], round(r['time'], 6))
        assert key not in seen
        seen.add(key)
        source = lookup[key]
        assert source['work_family'] == data['work']
        assert abs(float(source['time']) - r['time']) < 1e-7
        values.append([float(source['A_true']), int(r['base']), int(r['pitch_full'])])
    a = np.asarray(values)
    assert np.isfinite(a).all()
    x = a[:, 0] - a[:, 0].mean()
    gain = a[:, 2] - a[:, 1]
    assert x @ x > 0
    summary = dict(case=data['case'], n=len(a), A=float(a[:,0].mean()),
                   base=float(a[:,1].mean()), pitch=float(a[:,2].mean()),
                   row_oracle=float(np.maximum(a[:,1], a[:,2]).mean()),
                   slope=float(x @ gain / (x @ x)))
    groups.setdefault(data['work'], []).append(summary)
    n += len(a)
assert len(paths) == 40 and len(groups) == 23 and n == 3146
works = []
for work, cases in sorted(groups.items()):
    r = {k: float(np.mean([c[k] for c in cases])) for k in ['A','base','pitch','row_oracle','slope']}
    r.update(work=work, cases=cases)
    r['gain'] = r['pitch'] - r['base']
    r['work_oracle'] = max(r['pitch'], r['base'])
    works.append(r)
rng = np.random.default_rng(20260923)
idx = rng.integers(0, len(works), (5000, len(works)))
def estimate(v):
    v = np.asarray(v)
    return dict(mean=float(v.mean()), descriptive_bootstrap95=np.quantile(v[idx].mean(axis=1), [.025,.975]).tolist())
rho = spearmanr([r['A'] for r in works], [r['gain'] for r in works]).statistic
result = dict(status='opened-data diagnostic, not independent confirmation',
              n_rows=n, n_cases=len(paths), n_works=len(works),
              accuracy={k: float(np.mean([r[k] for r in works])) for k in ['base','pitch','work_oracle','row_oracle']},
              oracle_work_headroom_vs_pitch=estimate([r['work_oracle']-r['pitch'] for r in works]),
              A_within_case_gain_slope=estimate([r['slope'] for r in works]),
              A_work_gain_spearman=float(rho),
              pitch_better=sum(r['gain']>0 for r in works),
              chroma_better=sum(r['gain']<0 for r in works),
              ties=sum(r['gain']==0 for r in works), works=works, input_hashes=hashes)
assert all(digest(root / p)==h for p,h in hashes.items())
result['audit'] = 'PASS: complete unique time/work join; finite A; 40/23/3146; input hash drift 0'
target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['works','input_hashes']}, indent=2))
