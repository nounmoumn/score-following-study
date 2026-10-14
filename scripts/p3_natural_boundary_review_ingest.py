#!/usr/bin/env python3
"""Phase-1 validation for two independent natural-P3 boundary reviews."""

import csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PACKET=ROOT/'results/p3_natural_boundary_review_packet'
OUT=ROOT/'results/p3_natural_boundary_review_status.json'
REVIEWERS=('reviewer_A','reviewer_B')
DECISIONS={'accept','reject','uncertain'}; MISSED={'yes','no','uncertain'}

def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def validate_submission(folder, expected_reviewer):
 issues=[]; payload={}
 for name,id_field,decision_field,allowed in [('candidates.csv','candidate_id','decision',DECISIONS),
   ('files.csv','record_time','missed_boundary_check',MISSED)]:
  source=read(PACKET/name); path=folder/name
  if not path.exists():issues.append(f'missing {name}');continue
  rows=read(path);payload[name]=rows
  source_ids=[r[id_field] for r in source];ids=[r[id_field] for r in rows]
  if ids!=source_ids:issues.append(f'{name}: IDs/order differ from frozen packet')
  if len(set(ids))!=len(ids):issues.append(f'{name}: duplicate IDs')
  for row in rows:
   if row.get(decision_field,'').strip().lower() not in allowed:issues.append(f"{name}:{row.get(id_field)} invalid {decision_field}")
   if row.get('reviewer','').strip()!=expected_reviewer:issues.append(f"{name}:{row.get(id_field)} reviewer must be {expected_reviewer}")
   if not row.get('reason','').strip() and not row.get('review_reason','').strip():issues.append(f"{name}:{row.get(id_field)} missing reason")
 return payload,issues

def compare_reviews(submissions):
 disagreements=[]
 for name,id_field,field in [('candidates.csv','candidate_id','decision'),('files.csv','record_time','missed_boundary_check')]:
  a={r[id_field]:r[field].strip().lower() for r in submissions['reviewer_A'][name]}
  b={r[id_field]:r[field].strip().lower() for r in submissions['reviewer_B'][name]}
  disagreements += [{'file':name,'id':key,'reviewer_A':a[key],'reviewer_B':b[key]} for key in a if a[key]!=b[key]]
 return disagreements
def main():
 submissions={};issues={}
 for reviewer in REVIEWERS:
  submissions[reviewer],issues[reviewer]=validate_submission(PACKET/reviewer,reviewer)
 complete=all(not value for value in issues.values())
 disagreements=compare_reviews(submissions) if complete else []
 agreed_files={}
 if complete and not disagreements:
  agreed_files={r['record_time']:r['missed_boundary_check'].strip().lower() for r in submissions['reviewer_A']['files.csv']}
 excluded=[key for key,value in agreed_files.items() if value!='no']
 result={'scope':'Phase-1 boundary-review completeness and disagreement audit; no tracker/P3 outcomes',
  'submission_complete':complete,'issue_counts':{key:len(value) for key,value in issues.items()},
  'issue_examples':{key:value[:20] for key,value in issues.items()},'disagreement_count':len(disagreements),
  'disagreements':disagreements,'ready_for_adjudication':complete and bool(disagreements),
  'ready_for_unit_materialization':complete and not disagreements,
  'files_excluded_for_missed_or_uncertain_boundary':excluded,
  'ready_for_certification':False,'validation_eligible':False}
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
