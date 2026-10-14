"""One predesignated Batik development work; frozen ASAP warning rules, no holdout claim."""
import hashlib
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import partitura as pt
import yaml
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from score_following.features import load_notes, notes_to_chroma_stream
from score_following.p3_extension import cosine_cost, bayes_from_cost
from score_following.temporal import healthy_before
from score_following.current_evidence import current_evidence
from score_following.observation_evidence import predict_prior
from p3_batik_adapter_audit import FILES, MATCHED
from p3_short_signal import sustained_start

OUT = ROOT / 'results/p3_external_metadata/batik_kv279_1_development'
CONFIG = ROOT / 'configs/p3_temporal.yaml'

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    hop = float(cfg['hop_sec'])
    if hop != .1 or cfg['duration_sec'] != 90:
        raise AssertionError('pilot protocol requires frozen ASAP frame clock and 90s excerpt')
    audit = json.loads((ROOT / 'results/p3_external_metadata/batik_kv279_1_audit.json').read_text(encoding='utf-8'))
    if audit['match_ids_missing_from_unfolded_xml'] or audit['match_xml_onset_max_abs_beat'] or audit['match_xml_offset_max_abs_beat']:
        raise AssertionError('XML/match clock preflight failed')
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        score = pt.load_musicxml(str(FILES['score']))
        score_notes = pt.score.unfold_part_maximal(score[0], ignore_leaps=False).note_array()
    score_end_beats = float((score_notes['onset_beat'] + score_notes['duration_beat']).max())
    score_duration = score_end_beats * .5
    performance_notes = load_notes(FILES['performance'])
    performance_duration = max(note[1] for note in performance_notes)
    score_hop = hop * score_duration / performance_duration
    score_stream_notes = [(float(r['onset_beat']) * .5, float(r['onset_beat'] + r['duration_beat']) * .5, int(r['pitch'])) for r in score_notes]
    with threadpool_limits(limits=1):
        u = notes_to_chroma_stream(performance_notes, 90, hop)
        v = notes_to_chroma_stream(score_stream_notes, score_duration, score_hop)
        by_beat = defaultdict(list)
        for line in FILES['match'].read_text(encoding='utf-8').splitlines():
            m = MATCHED.match(line)
            if m:
                score_id, start, _, _, tick = m.groups()
                by_beat[float(start)].append(int(tick) * audit['seconds_per_tick'])
        beats = np.array(sorted(by_beat), dtype=float)
        perf_times = np.array([np.median(by_beat[b]) for b in beats])
        if np.any(np.diff(perf_times) <= 0):
            raise AssertionError('GT anchor times not strictly increasing')
        time = np.arange(len(u)) * hop
        truth = np.interp(time, perf_times, beats * .5) / score_hop
        cost = cosine_cost(u, v)
        b = cfg['bayes']
        path = bayes_from_cost(cost, b['transition'], b['emission_beta'], b['restart_probability'])
        error = np.abs(path - truth) * score_hop
        landmarks = []
        for t in np.arange(20, 75.001, 5):
            i = int(round(t / hop))
            if healthy_before(error, i - 1, 20, .5):
                landmarks.append(i)
        needed = {j for i in landmarks for j in range(i - 20, i)}
        state_index = np.arange(len(v))
        posterior = np.zeros(len(v)); posterior[0] = 1
        signals = {}
        max_identity_residual = 0.
        for i, row in enumerate(cost):
            if i:
                posterior = predict_prior(posterior, b['transition'], b['restart_probability'])
            if i in needed:
                posterior, m = current_evidence(posterior, row, np.abs(state_index - truth[i]) * score_hop <= .5, b['emission_beta'])
                signals[i] = m['B']
                max_identity_residual = max(max_identity_residual, abs(m['identity_residual']))
            else:
                posterior *= np.exp(-b['emission_beta'] * (row - row.min()))
                posterior /= posterior.sum()
            if int(posterior.argmax()) != int(path[i]):
                raise AssertionError(f'Bayes path mismatch at {i}')
        rows = []
        for i in landmarks:
            bs = np.array([signals[j] for j in range(i - 20, i)])
            onset = sustained_start(error, i, i + 20)
            rows.append({'time_sec': i * hop, 'sustained': onset is not None,
                         'onset_sec': onset * hop if onset is not None else None,
                         'any_positive': bool(np.any(bs > 0)),
                         'two_consecutive': bool(np.any((bs[:-1] > 0) & (bs[1:] > 0)))})
    def confusion(field):
        tp = sum(r[field] and r['sustained'] for r in rows)
        fp = sum(r[field] and not r['sustained'] for r in rows)
        fn = sum(not r[field] and r['sustained'] for r in rows)
        tn = sum(not r[field] and not r['sustained'] for r in rows)
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'false_alarm_rate': fp / (fp + tn) if fp + tn else None,
                'sensitivity': tp / (tp + fn) if tp + fn else None}
    summary = {'scope': 'One selected Batik development movement, no independent P3 confirmation',
               'source_score_notes': len(score_notes), 'score_hop_sec': score_hop,
               'gt_anchor_count': len(beats), 'landmarks_eligible': len(rows),
               'sustained_event_landmarks': sum(r['sustained'] for r in rows),
               'unique_event_onsets': len({r['onset_sec'] for r in rows if r['sustained']}),
               'max_bayes_identity_residual': max_identity_residual,
               'rules': {'any_positive': confusion('any_positive'), 'two_consecutive': confusion('two_consecutive')},
               'rows': rows,
               'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [CONFIG, Path(__file__), ROOT / 'P3_BATIK_ADAPTER_PROTOCOL.md'] + list(FILES.values())},
               'caveat': 'Oracle B uses GT neighborhood. One work cannot establish event prevalence or generalization; no parameter tuning from this result.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez_compressed(OUT / 'trace.npz', time_sec=time, truth_frame=truth, path_frame=path, error_score_sec=error)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'sha256')}, indent=2))

if __name__ == '__main__':
    main()
