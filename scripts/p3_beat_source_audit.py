"""Score-side onset support audit; never certifies annotation provenance or GT."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/p3_external_metadata'
CASES = {'KaszoS09M': '17-1', 'Giesbrecht02': '17-1_no_repeat'}

def supported(onsets, beat):
    # ASAP paper section 3.1: notes within 20 ms of score annotation.
    return bool(np.any(np.abs(onsets - beat) <= .020 + 1e-9))

def main():
    assert supported(np.array([1.]), 1.019)
    assert not supported(np.array([1.]), 1.021)
    assert not supported(np.array([]), 1.)
    src = BASE / 'anchor_interval_audit.json'
    source = json.loads(src.read_text(encoding='utf-8'))
    out = {'purpose': 'Development source audit, not independent annotation or P3 validation',
           'rule': 'Score note onset within 20 ms, following ASAP paper section 3.1. Absence is interpolation-eligible; presence does not prove a correct performance correspondence. Manual corrections may differ.',
           'paper': 'https://apmcleod.github.io/pdf/ismir-asap.pdf',
           'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(), 'cases': []}
    for case in source['cases']:
        score = ROOT / 'asap_dataset/Beethoven/Piano_Sonatas' / CASES[case['name']] / 'midi_score.mid'
        midi = pretty_midi.PrettyMIDI(str(score))
        onsets = np.array([n.start for i in midi.instruments for n in i.notes])
        rows = [{**r, 'score_onset_supported': supported(onsets, r['true_score_sec'])}
                for r in case['all_rows']]
        groups = {}
        for flag, label in ((True, 'onset_supported_candidate'), (False, 'interpolation_eligible')):
            rr = [r for r in rows if r['score_onset_supported'] == flag]
            groups[label] = {'n': len(rr), 'outside_conditional_bounds': sum(not r['inside_numeric_tolerance'] for r in rr),
                             'difference_gt_0_5_score_sec': sum(abs(r['sequence_error']) > .5 for r in rr)}
        out['cases'].append({'name': case['name'], 'groups': groups, 'rows': rows,
                             'score_sha256': hashlib.sha256(score.read_bytes()).hexdigest()})
        print(case['name'], json.dumps(groups))
    out['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (BASE / 'beat_source_audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('3 onset-support controls passed; no holdout or tracking outcomes opened.')

if __name__ == '__main__':
    main()
