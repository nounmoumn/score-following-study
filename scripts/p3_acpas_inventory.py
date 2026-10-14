"""Metadata-only inventory for ACPAS's independent direct-MIDI real MAPS subset."""
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data/acpas_metadata'
SOURCE = BASE / 'metadata_R.csv'
OUT = ROOT / 'results/p3_external_metadata/acpas_inventory.json'

def truth(value):
    return str(value).strip().lower() in {'true', '1', 'yes'}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    sources = Counter(row.get('source', '') for row in rows)
    maps = [row for row in rows if row.get('source', '').upper() == 'MAPS']
    selected = [row for row in maps if truth(row.get('aligned')) and float(row.get('duration') or 0) >= 90]
    works = {(row.get('composer', '').lower(), row.get('title', '').lower()) for row in selected}
    files = Counter()
    score_identity = Counter()
    for row in selected:
        folder = BASE / row.get('folder', '')
        for field in ('performance_MIDI', 'MIDI_score', 'performance_annotation', 'score_annotation'):
            files[field + '_present' if (folder / row.get(field, '')).is_file() else field + '_missing'] += 1
        score_identity['same_path' if row.get('performance_MIDI') == row.get('MIDI_score') else 'different_path'] += 1
        score_identity['same_annotation_path' if row.get('performance_annotation') == row.get('score_annotation') else 'different_annotation_path'] += 1
    summary = {
        'repo': 'https://github.com/cheriell/ACPAS-dataset',
        'commit': subprocess.check_output(['git', '-C', str(BASE), 'rev-parse', 'HEAD'], text=True).strip(),
        'total_real_rows': len(rows),
        'source_counts': dict(sources),
        'maps_rows': len(maps),
        'maps_aligned_and_90s_rows': len(selected),
        'maps_aligned_90s_composer_title_pairs': len(works),
        'maps_aligned_90s_composers': dict(Counter(row.get('composer', '') for row in selected)),
        'selected_file_presence': dict(files),
        'selected_score_identity': dict(score_identity),
        'eligible_for_p3_with_independent_score': score_identity['different_path'],
        'example_selected': [{field: row.get(field, '') for field in ('composer', 'title', 'folder', 'performance_MIDI', 'MIDI_score', 'performance_annotation', 'score_annotation', 'performance_MIDI_external', 'MIDI_score_external')} for row in selected[:2]],
        'fields': list(rows[0]) if rows else [],
        'caveat': 'All selected MAPS rows point performance MIDI and score MIDI at the same file; they fail the independent-score preflight. No tracker outcomes run.',
    }
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
