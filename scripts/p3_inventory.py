"""Read metadata and prior result identifiers, never held-out tracker outcomes."""
import csv,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from score_following.onset import work_family
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
meta=read(ROOT/'asap_dataset/metadata.csv')
used=set();sources=[]
for p in (ROOT/'results').rglob('*.csv'):
    if 'p3_inventory' in str(p):continue
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames:continue
        keys=[k for k in ['score','midi_score','composition'] if k in reader.fieldnames]
        if not keys:continue
        sources.append(str(p.relative_to(ROOT)))
        for r in reader:
            for k in keys:
                if r.get(k):used.add(r[k])
labels={r['composer']+'/'+r['title']:r['midi_score'] for r in meta}
used_scores={labels.get(s,s) for s in used}
folders=sorted({r['folder'] for r in meta})
out=ROOT/'results/p3_inventory';out.mkdir(exist_ok=True)
(out/'folders.json').write_text(json.dumps(folders,indent=2),encoding='utf-8')
(out/'used_scores.json').write_text(json.dumps(sorted(used_scores),indent=2),encoding='utf-8')
(out/'sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
used_families={work_family(r['folder']) for r in meta if r['midi_score'] in used_scores}
annot=json.loads((ROOT/'asap_dataset/asap_annotations.json').read_text(encoding='utf-8'))
candidates=[]
for r in meta:
    family=work_family(r['folder']);a=annot.get(r['midi_performance'],{});beats=a.get('performance_beats',[])
    if family in used_families:continue
    eligible=bool(a.get('score_and_performance_aligned') and beats and min(beats)<=15 and max(beats)>=90)
    candidates.append(dict(work_family=family,score=r['midi_score'],performance=r['midi_performance'],annotation_eligible=eligible))
with (out/'candidate_inventory.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(candidates[0]));w.writeheader();w.writerows(candidates)
summary=dict(score_count=len(folders),used_metadata_scores=sum(s in used_scores for s in {r['midi_score'] for r in meta}),
    unused_families=sorted({r['work_family'] for r in candidates}),annotation_eligible_unused_families=sorted({r['work_family'] for r in candidates if r['annotation_eligible']}),
    caveat='Local result CSV identifiers only; absence is not proof no other historical analysis used the work. Candidate inventory does not run held-out outcomes.')
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
