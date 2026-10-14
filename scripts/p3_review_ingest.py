"""Validate independent review files without manufacturing consensus."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/p3_onset_development/null_numeric/review_packet'
REQUIRED=('review_id','reviewer_id','score_position_before','score_position_after','judgment','evidence','reviewed_at','status')
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def validate(rows,expected_reviewer):
    errors=[]
    for i,r in enumerate(rows,2):
        for k in REQUIRED:
            if not r.get(k,'').strip():errors.append(f'row{i}: missing {k}')
        if r.get('reviewer_id')!=expected_reviewer:errors.append(f'row{i}: reviewer_id must be {expected_reviewer}')
        if r.get('status') not in ('complete','unresolved'):errors.append(f'row{i}: status must be complete or unresolved')
        try:
            a=float(r.get('score_position_before',''));b=float(r.get('score_position_after',''))
            if b<a:errors.append(f'row{i}: score position decreases')
        except ValueError:errors.append(f'row{i}: invalid score position')
    return errors
def main():
    requests=read(BASE/'requests.csv');ids={r['review_id'] for r in requests}
    reviews={x:read(BASE/f'reviewer_{x}.csv') for x in ('A','B')}
    errors={x:validate(rows,x) for x,rows in reviews.items()}
    for x,rows in reviews.items():
        seen=[r.get('review_id') for r in rows]
        if set(seen)!=ids:errors[x].append(f'review IDs must equal {sorted(ids)}; got {sorted(set(seen))}')
        if len(seen)!=len(set(seen)):errors[x].append('duplicate review_id')
    ready=all(not e for e in errors.values())
    conflicts=[]
    if ready:
        aa={r['review_id']:r for r in reviews['A']};bb={r['review_id']:r for r in reviews['B']}
        for rid in sorted(ids):
            a,b=aa[rid],bb[rid]
            fields=('score_position_before','score_position_after','repeat_occurrence','judgment','status')
            diff=[f for f in fields if a.get(f)!=b.get(f)]
            if diff:conflicts.append(dict(review_id=rid,differing_fields=diff))
    result=dict(ready_for_adjudication=ready,ready_for_gt=False,errors=errors,conflicts=conflicts,
        decision='pending independent reviews' if not ready else 'adjudication required' if conflicts else 'matching reviews still require provenance/adjudication approval',
        caveat='Matching reviewer text is not automatically certified GT. This tool validates completeness and exposes conflicts only.',
        provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),BASE/'requests.csv',BASE/'reviewer_A.csv',BASE/'reviewer_B.csv')})
    (BASE/'ingest_status.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
