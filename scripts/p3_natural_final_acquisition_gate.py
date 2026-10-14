#!/usr/bin/env python3
"""Frozen coverage gate for PianoVAM+eligible SMD+MAESTRO development data."""

import json
from pathlib import Path

try:
    from scripts import p3_natural_coverage_units as coverage
    from scripts import p3_natural_event_restart_candidates as event
except ImportError:
    import p3_natural_coverage_units as coverage
    import p3_natural_event_restart_candidates as event

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/p3_natural_final_acquisition_gate.json'

def main():
    smd=json.loads((ROOT/'results/p3_natural_smd_exact_supplement.json').read_text(encoding='utf8'))
    maestro=json.loads((ROOT/'results/p3_natural_maestro_supplement.json').read_text(encoding='utf8'))
    additions=[r for r in smd['rows']+maestro['rows'] if r['eligible']]
    records=event.records()+additions; rows=[]
    for record in records:
        candidate=event.process(record); unit=coverage.process(record,candidate)
        unit['unique_anchor_coverage']=candidate['unique_anchor_coverage']
        unit['content_mapping_weak']=candidate['content_mapping_weak']; rows.append(unit)
    ready=[r for r in rows if r['reviewable_units']]
    gates={'all_eligible_supplements_included':len(records)==25+len(additions),
           'recordings_with_reviewable_unit_ge_20':len(ready)>=20,
           'represented_composers_ge_8':len({r['composer'] for r in ready})>=8,
           'all_units_have_gate_provenance':all('gates' in u for r in rows for u in r['units'])}
    result={'scope':'Final outcome-blind natural-P3 acquisition gate; no tracker/P3 outcomes',
      'sources':{'base':25,'eligible_smd_exact':sum(r['eligible'] for r in smd['rows']),
                 'eligible_maestro':sum(r['eligible'] for r in maestro['rows'])},
      'counts':{'recordings':len(rows),'recordings_with_reviewable_unit':len(ready),
                'reviewable_units':sum(r['reviewable_units'] for r in rows),
                'represented_composers':len({r['composer'] for r in ready}),
                'content_mapping_weak':sum(r['content_mapping_weak'] for r in rows)},
      'gates':gates,'continue_to_boundary_review':all(gates.values()),'validation_eligible':False,'rows':rows}
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
