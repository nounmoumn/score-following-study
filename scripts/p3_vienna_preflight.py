"""Read-only Vienna 4x22 file and duration preflight; no alignment or tracker outcome."""
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from score_following.features import load_notes

BASE = ROOT / 'data/vienna4x22_rematched'
OUT = ROOT / 'results/p3_external_metadata/vienna_preflight.json'
ASAP_FAMILY = {
    'Chopin_op10_no3': 'Chopin/Etudes_op_10/3',
    'Chopin_op38': 'Chopin/Ballades/2',
    'Mozart_K331_1st-mov': 'Mozart/Piano_Sonatas/11',
    'Schubert_D783_no15': 'Schubert/D783/no15',
}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    match_files = sorted((BASE / 'match').glob('*.match'))
    midi_files = sorted((BASE / 'midi').glob('*.mid'))
    score_files = sorted((BASE / 'musicxml').glob('*.musicxml'))
    by_work = defaultdict(list)
    missing_midi = []
    for match in match_files:
        stem = match.stem
        work = re.sub(r'_p\d+$', '', stem)
        midi = BASE / 'midi' / (stem + '.mid')
        if not midi.is_file():
            missing_midi.append(stem)
            continue
        notes = load_notes(midi)
        duration = float(max((end for _, end, _ in notes), default=0.))
        by_work[work].append({'stem': stem, 'duration_sec': duration,
                              'midi_sha256': hashlib.sha256(midi.read_bytes()).hexdigest(),
                              'match_sha256': hashlib.sha256(match.read_bytes()).hexdigest()})
    xml_stems = {p.stem for p in score_files}
    used_scores = json.loads((ROOT / 'results/p3_inventory/used_scores.json').read_text(encoding='utf-8'))
    used_families = {str(Path(score).parent).replace('\\', '/') for score in used_scores}
    used_sonata_families = {'Mozart/Piano_Sonatas/' + score.split('/')[2].split('-')[0]
                            for score in used_scores if score.startswith('Mozart/Piano_Sonatas/')}
    def collision(work):
        family = ASAP_FAMILY[work]
        return family in used_families or family in used_sonata_families
    summary = {
        'repo': 'https://github.com/OFAI/vienna4x22_rematched',
        'commit': subprocess.check_output(['git', '-C', str(BASE), 'rev-parse', 'HEAD'], text=True).strip(),
        'match_files': len(match_files), 'midi_files': len(midi_files), 'xml_scores': len(score_files),
        'works': {work: {'performances': len(rows),
                         'duration_ge_90s': sum(r['duration_sec'] >= 90 for r in rows),
                         'duration_range_sec': [min(r['duration_sec'] for r in rows), max(r['duration_sec'] for r in rows)],
                         'score_xml_present': work in xml_stems,
                         'asap_family': ASAP_FAMILY[work],
                         'used_asap_work_or_sonata_collision': collision(work),
                         'eligible_90s_without_known_family_collision': sum(r['duration_sec'] >= 90 for r in rows) if not collision(work) else 0,
                         'rows': rows} for work, rows in by_work.items()},
        'unmatched_performance_midi': sorted({p.stem for p in midi_files} - {p.stem for p in match_files}),
        'match_without_midi': missing_midi,
        'caveat': 'File/duration audit only. Work overlap with existing ASAP and independent score/GT content must be checked before use. No tracker outcomes or GT paths computed.'
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v if k != 'works' else {name: {kk: vv for kk, vv in work.items() if kk != 'rows'} for name, work in v.items()}
                      for k, v in summary.items() if k != 'unmatched_performance_midi'}, indent=2))

if __name__ == '__main__':
    main()
