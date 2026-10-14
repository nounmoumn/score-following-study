"""GT-free candidate evidence at pre-landmark frames; development-only audit."""
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from p3_temporal import load_case, read_csv, write_csv
from score_following.p3_extension import cosine_cost
from score_following.observation_evidence import predict_prior, regional_evidence

SOURCE = ROOT / 'results/p3_temporal_20260918_000058'
LANDMARKS = ROOT / 'results/p3_onset_development/short_signal/landmarks.csv'
OUT = ROOT / 'results/p3_onset_development/online_precursor'

def matrix(rows, field):
    tp = sum(r[field] and r['sustained'] for r in rows)
    fp = sum(r[field] and not r['sustained'] for r in rows)
    fn = sum(not r[field] and r['sustained'] for r in rows)
    tn = sum(not r[field] and not r['sustained'] for r in rows)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                sensitivity=tp/(tp+fn) if tp+fn else None,
                false_alarm_rate=fp/(fp+tn) if fp+tn else None,
                positive_predictive_value=tp/(tp+fp) if tp+fp else None)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(exist_ok=True)
    (OUT / 'source_snapshot').mkdir(exist_ok=True)
    cfg = yaml.safe_load((SOURCE / 'config.yaml').read_text(encoding='utf-8'))
    hop = float(cfg['hop_sec'])
    if hop != .1:
        raise AssertionError('protocol clock changed')
    b = cfg['bayes']
    cases = read_csv(SOURCE / 'manifest.csv')
    references = read_csv(LANDMARKS)
    if len(references) != 396 or sum(r['sustained'] == 'True' for r in references) != 7:
        raise AssertionError('risk set changed')
    by_case = {c['case_id']: [r for r in references if r['case_id'] == c['case_id']] for c in cases}
    annot = json.loads((ROOT / cfg['asap_root'] / 'asap_annotations.json').read_text(encoding='utf-8'))
    rows = []
    max_residual = 0.
    with threadpool_limits(limits=1):
        for case in cases:
            queries = by_case[case['case_id']]
            if not queries:
                continue
            u, v, hv, _, _ = load_case(case, cfg, annot)
            cost = cosine_cost(u, v)
            with np.load(SOURCE / 'cases' / (case['case_id'] + '.npz')) as z:
                original = z['intact_bayes']
            needed = {j for r in queries for j in range(int(round(float(r['time_sec']) / hop)) - 20, int(round(float(r['time_sec']) / hop)))}
            posterior = np.zeros(len(v)); posterior[0] = 1
            signals = {}
            radius = max(1, int(round(.5 / hv)))
            exclude = max(1, int(round(2. / hv)))
            for i, cost_row in enumerate(cost):
                if i:
                    posterior = predict_prior(posterior, b['transition'], b['restart_probability'])
                if i in needed:
                    center = int(np.argmax(posterior))
                    posterior, m = regional_evidence(posterior, cost_row, center, radius, exclude, b['emission_beta'])
                    signals[i] = (m['observation_log_bayes_factor'] < 0, m['posterior_log_odds'] < 0,
                                  m['prior_log_odds'], m['observation_log_bayes_factor'], m['posterior_log_odds'])
                    max_residual = max(max_residual, abs(m['identity_residual']))
                else:
                    posterior *= np.exp(-b['emission_beta'] * (cost_row - cost_row.min()))
                    posterior /= posterior.sum()
                if int(np.argmax(posterior)) != int(original[i]):
                    raise AssertionError(f'path mismatch {case["case_id"]} frame {i}')
            for r in queries:
                start = int(round(float(r['time_sec']) / hop))
                evidence = np.array([signals[j][0] for j in range(start - 20, start)], dtype=bool)
                odds = np.array([signals[j][1] for j in range(start - 20, start)], dtype=bool)
                prior_margin = np.array([signals[j][2] for j in range(start - 20, start)])
                observation_margin = np.array([signals[j][3] for j in range(start - 20, start)])
                posterior_margin = np.array([signals[j][4] for j in range(start - 20, start)])
                rows.append(dict(case_id=case['case_id'], work_family=r['work_family'],
                                 time_sec=float(r['time_sec']), sustained=r['sustained'] == 'True',
                                 evidence_any=bool(evidence.any()),
                                 evidence_two=bool(np.any(evidence[:-1] & evidence[1:])),
                                 odds_any=bool(odds.any()),
                                 odds_two=bool(np.any(odds[:-1] & odds[1:])),
                                 evidence_positive_frames=int(evidence.sum()),
                                 odds_positive_frames=int(odds.sum()),
                                 prior_margin_median=float(np.median(prior_margin)),
                                 observation_margin_median=float(np.median(observation_margin)),
                                 posterior_margin_median=float(np.median(posterior_margin))))
            print(case['case_id'], flush=True)
    if len(rows) != 396 or sum(r['sustained'] for r in rows) != 7:
        raise AssertionError('outcome/risk-set mismatch')
    write_csv(OUT / 'landmarks.csv', rows)
    fields = ('evidence_any', 'evidence_two', 'odds_any', 'odds_two')
    by_family = {}
    for fam in sorted({r['work_family'] for r in rows}):
        rr = [r for r in rows if r['work_family'] == fam]
        by_family[fam] = dict(event=any(r['sustained'] for r in rr), **{f: any(r[f] for r in rr) for f in fields})
    summary = dict(scope='GT-free signals, 40 existing development scores; no independent validation',
                   landmarks=len(rows), event_landmarks=sum(r['sustained'] for r in rows),
                   work_families=len(by_family), event_families=sum(f['event'] for f in by_family.values()),
                   max_bayes_identity_residual=max_residual,
                   rules={f: matrix(rows, f) for f in fields},
                   family_alerts={f: dict(event=sum(v['event'] and v[f] for v in by_family.values()),
                                          no_event=sum(not v['event'] and v[f] for v in by_family.values())) for f in fields},
                   family_counts=dict(Counter('event' if v['event'] else 'no_event' for v in by_family.values())),
                   margin_medians={group: {key: float(np.median([r[key] for r in rows if r['sustained'] == is_event]))
                                           for key in ('prior_margin_median', 'observation_margin_median', 'posterior_margin_median')}
                                   for group, is_event in (('event', True), ('no_event', False))},
                   caveat='Posterior/likelihood regions are prior-defined and GT-free; risk-set health/outcome labels use GT. All four rules exploratory on repeatedly inspected development data.')
    (OUT / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    files = [Path(__file__), ROOT / 'P3_ONLINE_PRECURSOR_PROTOCOL.md', SOURCE / 'config.yaml', SOURCE / 'manifest.csv', LANDMARKS,
             ROOT / 'src/score_following/observation_evidence.py']
    for file in files:
        shutil.copy2(file, OUT / 'source_snapshot' / file.name)
    (OUT / 'provenance.json').write_text(json.dumps({str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
