"""Development-only clock and alignment audit for Batik kv279_1; no tracker run."""
import hashlib
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import partitura as pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from score_following.features import load_notes

BASE = ROOT / 'data/batik_plays_mozart'
FILES = {
    'match': BASE / 'match/kv279_1.match',
    'performance': BASE / 'midi/kv279_1.mid',
    'score': BASE / 'scores_edited/kv279_1.musicxml',
}
OUT = ROOT / 'results/p3_external_metadata/batik_kv279_1_audit.json'
MATCHED = re.compile(r'^snote\(([^,]+),.*?,(-?\d+\.\d+),(-?\d+\.\d+),\[[^]]*\]\)-note\([^,]+,(\d+),(\d+),')

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for name, path in FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f'{name}: {path}')
    if len({path.resolve() for path in FILES.values()}) != len(FILES):
        raise AssertionError('score, performance, match must be different files')
    sha = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in FILES.items()}
    if len(set(sha.values())) != len(sha):
        raise AssertionError('input files have identical content')
    match_text = FILES['match'].read_text(encoding='utf-8')
    units = int(re.search(r'info\(midiClockUnits,(\d+)\)', match_text).group(1))
    rate = int(re.search(r'info\(midiClockRate,(\d+)\)', match_text).group(1))
    seconds_per_tick = rate / (units * 1_000_000)
    paired = []
    prefix_counts = defaultdict(int)
    for line in match_text.splitlines():
        prefix_counts[line.split('(', 1)[0].split('-', 1)[0]] += 1
        m = MATCHED.match(line)
        if m:
            score_id, score_onset, score_offset, pitch, perf_tick = m.groups()
            paired.append((score_id, float(score_onset), float(score_offset), int(pitch), int(perf_tick) * seconds_per_tick))
    if len(paired) < 100:
        raise AssertionError('too few matched notes parsed')
    by_beat = defaultdict(list)
    for _, score_on, _, _, perf_on in paired:
        by_beat[score_on].append(perf_on)
    beat = np.array(sorted(by_beat), dtype=float)
    perf = np.array([np.median(by_beat[b]) for b in beat], dtype=float)
    backward = np.diff(perf) < -0.05
    # Compare converted onset ticks to nearest pitch-matched note in original MIDI.
    midi_by_pitch = defaultdict(list)
    for onset, _, pitch in load_notes(FILES['performance']):
        midi_by_pitch[pitch].append(onset)
    residual = []
    for _, _, _, pitch, on in paired:
        arr = np.asarray(midi_by_pitch[pitch])
        if len(arr):
            pos = np.searchsorted(arr, on)
            residual.append(min(abs(arr[k] - on) for k in (pos - 1, pos) if 0 <= k < len(arr)))
    with warnings.catch_warnings(record=True) as xml_warnings:
        warnings.simplefilter('always')
        xml_score = pt.load_musicxml(str(FILES['score']))
        score_array = xml_score.note_array()
        unfolded_array = pt.score.unfold_part_maximal(xml_score[0], ignore_leaps=False).note_array()
    xml_by_id = {str(row['id']): (float(row['onset_beat']), float(row['onset_beat'] + row['duration_beat'])) for row in unfolded_array}
    match_ids = [row[0] for row in paired]
    missing_xml_ids = sorted(set(match_ids) - set(xml_by_id))
    matched_onset_delta = [abs(row[1] - xml_by_id[row[0]][0]) for row in paired if row[0] in xml_by_id]
    matched_offset_delta = [abs(row[2] - xml_by_id[row[0]][1]) for row in paired if row[0] in xml_by_id]
    if not {'onset_beat', 'duration_beat', 'pitch'}.issubset(score_array.dtype.names):
        raise AssertionError('XML score note array lacks timing or pitch')
    result = {
        'scope': 'Batik kv279_1 development-only annotation clock audit; no tracker, B signal, P3 event or held-out outcome',
        'sha256': sha,
        'midi_clock_units': units,
        'midi_clock_rate': rate,
        'seconds_per_tick': seconds_per_tick,
        'matched_note_pairs_parsed': len(paired),
        'distinct_score_onsets': len(beat),
        'performance_onset_range_sec': [float(perf.min()), float(perf.max())],
        'score_onset_range_quarters': [float(beat.min()), float(beat.max())],
        'median_by_score_onset_backward_steps_over_50ms': int(backward.sum()),
        'median_by_score_onset_backward_fraction': float(backward.mean()),
        'nearest_same_pitch_midi_residual_median_sec': float(np.median(residual)),
        'nearest_same_pitch_midi_residual_p95_sec': float(np.percentile(residual, 95)),
        'xml_note_count': len(score_array),
        'xml_score_onset_range_beats': [float(score_array['onset_beat'].min()), float(score_array['onset_beat'].max())],
        'xml_score_end_beats': float((score_array['onset_beat'] + score_array['duration_beat']).max()),
        'unfolded_xml_note_count': len(unfolded_array),
        'unfolded_xml_score_end_beats': float((unfolded_array['onset_beat'] + unfolded_array['duration_beat']).max()),
        'match_ids_missing_from_unfolded_xml': len(missing_xml_ids),
        'match_id_duplicates': len(match_ids) - len(set(match_ids)),
        'xml_id_duplicates': len(unfolded_array) - len(xml_by_id),
        'match_xml_onset_max_abs_beat': max(matched_onset_delta) if matched_onset_delta else None,
        'match_xml_offset_max_abs_beat': max(matched_offset_delta) if matched_offset_delta else None,
        'xml_warning_count': len(xml_warnings),
        'xml_warning_examples': list(dict.fromkeys(str(w.message).splitlines()[0] for w in xml_warnings))[:5],
        'match_line_prefix_counts': dict(prefix_counts),
        'caveat': 'No interpolation GT accepted yet. Chord asynchrony, repeats, unusual tempi, parsing omissions and MIDI tempo changes need review.',
    }
    OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sha256', 'match_line_prefix_counts')}, indent=2))

if __name__ == '__main__':
    main()
