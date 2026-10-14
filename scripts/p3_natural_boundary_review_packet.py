#!/usr/bin/env python3
"""Freeze the tracker-blind boundary/unit review packet for natural P3."""

import csv,hashlib,json
from pathlib import Path

try:
 from scripts import p3_natural_coverage_units as coverage
 from scripts import p3_natural_event_restart_candidates as event
except ImportError:
 import p3_natural_coverage_units as coverage
 import p3_natural_event_restart_candidates as event

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/p3_natural_boundary_review_packet'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def all_records():
 smd=json.loads((ROOT/'results/p3_natural_smd_exact_supplement.json').read_text(encoding='utf8'))
 maestro=json.loads((ROOT/'results/p3_natural_maestro_supplement.json').read_text(encoding='utf8'))
 scriabin=json.loads((ROOT/'results/p3_natural_maestro_scriabin.json').read_text(encoding='utf8'))['record']
 return event.records()+[r for r in smd['rows']+maestro['rows'] if r['eligible']]+[scriabin]
def write_csv(path,fields,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(parents=True,exist_ok=True); candidates=[];units=[];files=[]; summaries=[]
 for record in all_records():
  c=event.process(record);u=coverage.process(record,c);summaries.append(u)
  files.append({'record_time':record['record_time'],'composer':record['score_composer'],'piece':record['piece'],
   'performance_path':record['performance_path'],'score_path':record['score_midi_path'],
   'performance_sha256':digest(ROOT/record['performance_path']),'score_sha256':digest(ROOT/record['score_midi_path']),
   'candidate_count':c['candidate_count'],'reviewable_units':u['reviewable_units'],
   'missed_boundary_check':'','reviewer':'','review_reason':''})
  for x in c['candidates']:
   candidates.append({'candidate_id':x['candidate_id'],'record_time':record['record_time'],'composer':record['score_composer'],
    'piece':record['piece'],'time_sec':x['time_sec'],'signals':'+'.join(x['signals']),
    'context_top_pitches':' '.join(str(v['top_pitch']) for v in x['context']),
    'decision':'','reviewer':'','reason':''})
  for x in u['units']:
   units.append({'unit_id':x['unit_id'],'record_time':record['record_time'],'composer':record['score_composer'],
    'piece':record['piece'],'start_sec':x['start_sec'],'end_sec':x['end_sec'],'anchor_count':x['anchor_count'],
    'score_span':x['score_span'],'monotone_share':x['monotone_share'],'algorithm_reviewable':x['reviewable'],
    'decision':'','reviewer':'','reason':''})
 fields_files=list(files[0]);fields_candidates=list(candidates[0]);fields_units=list(units[0])
 write_csv(OUT/'files.csv',fields_files,files);write_csv(OUT/'candidates.csv',fields_candidates,candidates);write_csv(OUT/'units.csv',fields_units,units)
 ready=[r for r in summaries if r['reviewable_units']]
 gates={'recordings_with_reviewable_unit_ge_20':len(ready)>=20,
  'represented_composers_ge_8':len({r['composer'] for r in ready})>=8,
  'all_files_hashed':all(r['performance_sha256'] and r['score_sha256'] for r in files),
  'review_fields_blank':all(not r['decision'] and not r['reviewer'] for r in candidates+units) and all(not r['missed_boundary_check'] for r in files)}
 manifest={'scope':'Tracker-blind natural-P3 boundary review packet; no outcomes or GT labels',
  'counts':{'recordings':len(files),'candidates':len(candidates),'provisional_units':len(units),
   'recordings_with_reviewable_unit':len(ready),'represented_composers':len({r['composer'] for r in ready})},
  'gates':gates,'ready_for_independent_boundary_review':all(gates.values()),'validation_eligible':False,
  'instructions':['Review candidates.csv without tracker output; decision=accept/reject/uncertain.',
   'For every file, complete files.csv missed_boundary_check=yes/no/uncertain; candidate review alone cannot certify a file.',
   'Review units.csv only after boundary decisions. Algorithm reviewable is a quality flag, not ground truth.',
   'Use independent reviewers; adjudicate disagreements without opening P3 outcomes.'],
  'artifacts':{name:digest(OUT/name) for name in ['files.csv','candidates.csv','units.csv']}}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
