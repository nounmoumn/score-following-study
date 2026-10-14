#!/usr/bin/env python3
"""Validate phase-2 unit reviews and emit only doubly accepted certified units."""

import csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];PACKET=ROOT/'results/p3_natural_boundary_review_packet'
CONSENSUS=PACKET/'consensus_units.csv';OUT=ROOT/'results/p3_natural_certified_units.json'
DECISIONS={'accept','reject','uncertain'};REVIEWERS=('reviewer_A','reviewer_B')

def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def validate(rows,source,reviewer):
 issues=[];expected=[r['unit_id'] for r in source];actual=[r.get('unit_id','') for r in rows]
 if actual!=expected:issues.append('unit IDs/order differ from consensus template')
 if len(actual)!=len(set(actual)):issues.append('duplicate unit IDs')
 for row in rows:
  uid=row.get('unit_id','')
  if row.get('decision','').strip().lower() not in DECISIONS:issues.append(f'{uid}: invalid decision')
  if row.get('reviewer','').strip()!=reviewer:issues.append(f'{uid}: reviewer must be {reviewer}')
  if not row.get('reason','').strip():issues.append(f'{uid}: missing reason')
 return issues
def compare(a,b):
 aa={r['unit_id']:r['decision'].strip().lower() for r in a};bb={r['unit_id']:r['decision'].strip().lower() for r in b}
 return [{'unit_id':key,'reviewer_A':aa[key],'reviewer_B':bb[key]} for key in aa if aa[key]!=bb[key]]
def main():
 if not CONSENSUS.exists():
  result={'finalized':False,'reason':'consensus unit template missing; phase 1 not complete','validation_eligible':False}
 else:
  source=read(CONSENSUS);reviews={};issues={}
  for reviewer in REVIEWERS:
   path=PACKET/f'{reviewer}_units.csv'
   if path.exists():reviews[reviewer]=read(path);issues[reviewer]=validate(reviews[reviewer],source,reviewer)
   else:reviews[reviewer]=[];issues[reviewer]=['review file missing']
  complete=all(not x for x in issues.values());conflicts=compare(reviews['reviewer_A'],reviews['reviewer_B']) if complete else []
  certified=[]
  if complete and not conflicts:
   decisions={r['unit_id']:r['decision'].strip().lower() for r in reviews['reviewer_A']}
   certified=[r for r in source if decisions[r['unit_id']]=='accept' and r['reviewable'].strip().lower()=='true']
  composers={r['composer'] for r in certified};recordings={r['record_time'] for r in certified}
  gates={'reviews_complete':complete,'no_review_conflicts':complete and not conflicts,
   'certified_recordings_ge_20':len(recordings)>=20,'certified_composers_ge_8':len(composers)>=8}
  result={'scope':'Phase-2 unit certification; no tracker/P3 outcomes','finalized':all(gates.values()),
   'issue_counts':{key:len(value) for key,value in issues.items()},'issue_examples':{key:value[:20] for key,value in issues.items()},
   'conflicts':conflicts,'counts':{'consensus_units':len(source),'certified_units':len(certified),
   'certified_recordings':len(recordings),'certified_composers':len(composers)},'gates':gates,
   'certified_units':certified if all(gates.values()) else [],'ready_for_alignment_construction':all(gates.values()),
   'validation_eligible':False,'provenance':{'consensus_units_sha256':sha(CONSENSUS),**{
    f'{reviewer}_sha256':sha(PACKET/f'{reviewer}_units.csv') for reviewer in REVIEWERS if (PACKET/f'{reviewer}_units.csv').exists()}}}
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='certified_units'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
