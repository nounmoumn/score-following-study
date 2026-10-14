#!/usr/bin/env python3
"""Rebuild unit-review templates from agreed phase-1 boundary decisions."""

import csv,json
from pathlib import Path

try:
 from scripts import p3_natural_boundary_review_packet as packet
 from scripts import p3_natural_boundary_review_ingest as ingest
 from scripts import p3_natural_coverage_units as coverage
 from scripts import p3_natural_event_restart_candidates as event
except ImportError:
 import p3_natural_boundary_review_packet as packet
 import p3_natural_boundary_review_ingest as ingest
 import p3_natural_coverage_units as coverage
 import p3_natural_event_restart_candidates as event

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/p3_natural_boundary_review_packet'
STATUS=ROOT/'results/p3_natural_boundary_review_status.json';OUT=ROOT/'results/p3_natural_consensus_units.json'
CSV_OUT=BASE/'consensus_units.csv'

def build_units(record,accepted_ids):
 performance=event.load_notes(ROOT/record['performance_path']);score=event.load_notes(ROOT/record['score_midi_path'])
 pe=event.onset_events(performance,event.PERFORMANCE_ONSET_TOLERANCE);se=event.onset_events(score,event.SCORE_ONSET_TOLERANCE)
 anchors=event.event_anchors([x['token'] for x in pe],[x['token'] for x in se]);candidate=event.process(record)
 accepted=[c for c in candidate['candidates'] if c['candidate_id'] in accepted_ids]
 cuts=[(0,'file_start')]+[(int(c['performance_index']),c['candidate_id']) for c in accepted]+[(len(pe),'file_end')]
 cuts=sorted(set(cuts),key=lambda x:x[0]);rows=[]
 for number,((start,left),(end,right)) in enumerate(zip(cuts,cuts[1:]),1):
  if end<=start:continue
  unit=coverage.unit_metrics(anchors,pe,len(se),start,end)
  unit.update({'unit_id':f"{record['record_time']}-C{number:03d}",'record_time':record['record_time'],
   'composer':record['score_composer'],'piece':record['piece'],'left_boundary_id':left,'right_boundary_id':right})
  rows.append(unit)
 return rows

def write_csv(rows):
 fields=['unit_id','record_time','composer','piece','left_boundary_id','right_boundary_id','start_sec','end_sec',
  'anchor_count','score_span','monotone_share','reviewable','decision','reviewer','reason']
 with CSV_OUT.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({**{key:row.get(key,'') for key in fields},'decision':'','reviewer':'','reason':''})

def main():
 if not STATUS.exists():
  result={'materialized':False,'reason':'phase-1 ingest status missing','validation_eligible':False}
 elif not (status:=json.loads(STATUS.read_text(encoding='utf8'))).get('ready_for_unit_materialization'):
  result={'materialized':False,'reason':'phase-1 reviews incomplete or disagree','status':status,'validation_eligible':False}
 else:
  a_candidates=ingest.read(BASE/'reviewer_A/candidates.csv');a_files=ingest.read(BASE/'reviewer_A/files.csv')
  accepted={r['candidate_id'] for r in a_candidates if r['decision'].strip().lower()=='accept'}
  safe={r['record_time'] for r in a_files if r['missed_boundary_check'].strip().lower()=='no'}
  records={r['record_time']:r for r in packet.all_records()};rows=[]
  for record_id in sorted(safe):rows.extend(build_units(records[record_id],accepted))
  write_csv(rows);result={'materialized':True,'safe_recordings':len(safe),'excluded_recordings':len(records)-len(safe),
   'accepted_boundaries':len(accepted),'consensus_units':len(rows),'reviewable_units':sum(r['reviewable'] for r in rows),
   'unit_review_template':str(CSV_OUT.relative_to(ROOT)).replace('\\','/'),'ready_for_phase2_unit_review':bool(rows),
   'validation_eligible':False}
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
