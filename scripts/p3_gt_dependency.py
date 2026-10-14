"""Read-only development audit: cached paths, GT dependencies, label-only scenarios."""
import csv
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from score_following.onset import onset_outcome
from score_following.ground_truth import piecewise_linear_gt

BASE = ROOT / 'results/p3_onset_development'
SOURCE = ROOT / 'results/p3_temporal_20260918_000058'
OUT = BASE / 'gt_dependency'

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def onset(error, start, stop):
    return next((i for i in range(start, stop) if len(error[i:i+5]) == 5 and np.all(error[i:i+5] > .5)), None)

def main():
    OUT.mkdir(exist_ok=True)
    cases = read(SOURCE / 'manifest.csv')
    marks = read(BASE / 'landmarks.csv')
    online = {(r['case_id'], float(r['time_sec'])): r for r in read(BASE / 'online_precursor/landmarks.csv')}
    short = {(r['case_id'], float(r['time_sec'])): r for r in read(BASE / 'short_signal/landmarks.csv')}
    annotations_path = ROOT / 'asap_dataset/asap_annotations.json'
    annotations = json.loads(annotations_path.read_text(encoding='utf-8'))
    rows, frames, provenance = [], [], {}
    for c in cases:
        score = ROOT / 'asap_dataset' / c['score']
        perf = ROOT / 'asap_dataset' / c['performance']
        sn = [n for inst in pretty_midi.PrettyMIDI(str(score)).instruments for n in inst.notes]
        pn = [n for inst in pretty_midi.PrettyMIDI(str(perf)).instruments for n in inst.notes]
        hv = .1 * max(n.end for n in sn) / max(n.end for n in pn)
        a = annotations[c['performance']]
        pb, sb = np.array(a['performance_beats']), np.array(a['midi_score_beats'])
        assert len(pb) == len(sb) and len(pb) > 1
        order = np.argsort(pb); pb, sb = pb[order], sb[order]
        assert np.all(np.diff(pb) > 0)
        supported = np.array([any(abs(n.start-s) <= .020+1e-9 for n in sn) for s in sb])
        pathfile = SOURCE / 'cases' / (c['case_id'] + '.npz')
        with np.load(pathfile) as z:
            path = z['intact_bayes']
        time = np.arange(len(path)) * .1
        gt, _, _ = piecewise_linear_gt(pb, sb)
        truth = gt(time)
        error = abs(path - truth/hv)*hv
        provenance[c['case_id']] = {str(p.relative_to(ROOT)): digest(p) for p in (score, perf, pathfile)}
        for r in (r for r in marks if r['case_id'] == c['case_id']):
            t = float(r['time_sec']); start = int(round(t*10))
            outcome = onset_outcome(error, start)
            assert outcome['status'] == r['status']
            assert str(outcome['eligible']) == r['eligible']
            assert ('' if outcome['failure_frame'] is None else str(outcome['failure_frame'])) == r['failure_frame']
            key = (c['case_id'], t)
            on = onset(error, start, start+20) if outcome['eligible'] else None
            if key in short:
                assert (on is not None) == (short[key]['sustained'] == 'True')
                assert ('' if on is None else str(on)) == short[key]['onset_frame']
            # Union of history, exposure, both follow-ups; no new risk-set selection.
            indices = range(start-21, start+70)
            unsupported, outside, nearest = 0, 0, []
            for i in indices:
                j = int(np.searchsorted(pb, time[i], side='left'))
                exact = j < len(pb) and abs(pb[j]-time[i]) <= 1e-9
                left = j if exact else max(0, j-1)
                right = min(j, len(pb)-1)
                out = bool(time[i] < pb[0] or time[i] > pb[-1])
                no_support = bool(not supported[left] or not supported[right])
                unsupported += no_support; outside += out
                nearest.append(abs(error[i]-.5))
                frames.append(dict(case_id=c['case_id'], landmark_sec=t, frame=i,
                    performance_sec=float(time[i]), available_sec=float(time[i]+.1),
                    original_score_sec=float(truth[i]), original_error_score_sec=float(error[i]),
                    left_sorted_beat=left, right_sorted_beat=right,
                    left_perf_sec=float(pb[left]), right_perf_sec=float(pb[right]),
                    left_score_sec=float(sb[left]), right_score_sec=float(sb[right]),
                    left_onset_supported=bool(supported[left]), right_onset_supported=bool(supported[right]),
                    exact_beat=exact, outside_annotation_support=out))
            rows.append(dict(**r, score_hop_sec=hv,
                online_included=key in online, online_original_event=on is not None,
                online_onset_sec=None if on is None else on*.1,
                online_confirmation_available_sec=None if on is None else (on+5)*.1,
                current_position_included=r['status'] in ('new_failure','no_failure'),
                current_feature_available_sec=t+2,
                current_confirmation_available_sec=None if outcome['failure_frame'] is None else (outcome['failure_frame']+5)*.1,
                dependency_frames=len(indices), onset_unsupported_endpoint_frames=unsupported,
                outside_annotation_support_frames=outside,
                minimum_distance_to_error_threshold_score_sec=float(min(nearest))))
    assert len(rows) == 480 and len(online) == 396
    scenarios = {}
    # Fixed risk set and fixed alerts only; optimistic algebra, not GT corrections or a pass gate.
    for field in ('evidence_any','evidence_two','odds_any','odds_two'):
        rr = list(online.values())
        tp = sum(r[field]=='True' and r['sustained']=='True' for r in rr)
        fp = sum(r[field]=='True' and r['sustained']!='True' for r in rr)
        tn = sum(r[field]!='True' and r['sustained']!='True' for r in rr)
        need = {}
        for target in (.5,.2,.1,.05):
            need[str(target)] = next((k for k in range(fp+1) if fp+tn-k > 0 and (fp-k)/(fp+tn-k) <= target), None)
        scenarios[field] = dict(tp=tp, fp=fp, tn=tn,
            minimum_alerted_negative_to_positive_flips_for_descriptive_fpr=need,
            note='Keep all other labels and alerts fixed. Targets are descriptive scenarios, not new acceptance thresholds. Risk-set changes and oracle feature refits are not covered.')
    summary = dict(status='complete', landmarks=len(rows), dependency_rows=len(frames),
        original_status_counts=dict(Counter(r['status'] for r in rows)),
        online_events=sum(r['online_original_event'] for r in rows if r['online_included']),
        online_landmarks_with_unsupported_endpoints=sum(r['onset_unsupported_endpoint_frames']>0 for r in rows if r['online_included']),
        all_landmarks_with_outside_annotation_support=sum(r['outside_annotation_support_frames']>0 for r in rows),
        label_only_scenarios=scenarios,
        limitations=['Onset support is not annotation certification or a valid GT error bound.',
                    'Frame union is for dependencies, not a new outcome horizon.',
                    'Current-position features and inclusion depend on GT; no label-only refit claim.',
                    'Score hop uses full performance duration, so GT-free alerts are not an end-to-end online pipeline.'])
    for name, data in [('landmarks', rows), ('frame_dependencies', frames)]:
        with (OUT/(name+'.csv')).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    provenance['inputs']={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),annotations_path,BASE/'landmarks.csv',BASE/'online_precursor/landmarks.csv',BASE/'short_signal/landmarks.csv',SOURCE/'config.yaml',SOURCE/'manifest.csv')}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
