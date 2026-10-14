"""Metadata-only selection of previously unreferenced ASAP work families."""
import csv,hashlib,json,re
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'asap_dataset';OUT=ROOT/'output/ambiguity_map_fresh_cohort'
SALT='ambiguity-map-fresh-v1-20260923';N=8
TEXT={'.md','.json','.csv','.yaml','.yml','.py','.txt','.toml'}
SKIP={'.git','.venv','asap_dataset','output/ambiguity_map_fresh_cohort','results/ambiguity_map_fresh_dev','results/ambiguity_map_fresh_lockbox'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def family(folder):
 parts=folder.replace('\\','/').split('/');parts[-1]=re.sub(r'_(?:no_.*repeat|no_trio|repeat)$','',parts[-1])
 if parts[0]=='Bach' and parts[1] in ['Fugue','Prelude']:return 'Bach/'+parts[2]
 if len(parts)>=3 and parts[1] in ['Piano_Sonatas','Keyboard_Sonatas','Sonatas']:return '/'.join(parts[:2]+[parts[2].split('-')[0]])
 if len(parts)>=3 and (parts[1].startswith('Sonata_') or parts[1] in ['Kreisleriana','Gaspard_de_la_Nuit','Miroirs','Images_Book_1','Pour_le_Piano']):return '/'.join(parts[:2])
 return '/'.join(parts)
def performer(path):
 stem=Path(path).stem;stem=re.sub(r'\d.*$','',stem);stem=re.sub(r'M$','',stem);return stem.lower()
def key(s):return hashlib.sha256((SALT+'|'+s).encode()).hexdigest()
def main():
 if OUT.exists():raise SystemExit('Refuse overwrite')
 refs=set();hashes=set();scanned=[];large_metadata_skipped=[]
 scan_files=list(ROOT.glob('*.md'))+[ROOT/'output/score_priority_plan/cohort_manifest.csv']
 for folder in ('results','configs'):
  base=ROOT/folder
  if base.exists():scan_files.extend(base.rglob('*'))
 for p in scan_files:
  if not p.is_file() or p.suffix.lower() not in TEXT:continue
  rel=p.relative_to(ROOT).as_posix()
  if any(rel==x or rel.startswith(x+'/') for x in SKIP):continue
  low=rel.lower()
  if rel!='output/score_priority_plan/cohort_manifest.csv' and any(x in low for x in ('metadata','inventory','preflight','manifest','freeze.json','audit_snapshot')):continue
  # Very large tables are numeric frame/external-corpus metadata; cohort paths and
  # hashes are frozen in their smaller manifests/configs. Record rather than load
  # hundreds of MB into one regex string.
  if p.stat().st_size>25*1024*1024:
   large_metadata_skipped.append(dict(path=rel,bytes=p.stat().st_size));continue
  try:
   with p.open(encoding='utf-8',errors='ignore') as f:
    for text in f:
     refs.update(x.replace('\\','/') for x in re.findall(r"(?:[A-Za-z0-9_.() -]+/)+[A-Za-z0-9_.() -]+\.mid",text))
     hashes.update(re.findall(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])',text.lower()))
  except OSError:continue
  scanned.append(rel)
 ann=json.loads((DATA/'asap_annotations.json').read_text());raw=[]
 for perf,a in ann.items():
  if not a.get('score_and_performance_aligned'):continue
  score=(Path(perf).parent/'midi_score.mid').as_posix();pp=DATA/perf;sp=DATA/score
  if not pp.exists() or not sp.exists():continue
  ph,sh=sha(pp),sha(sp);fam=family(str(Path(perf).parent));reason=[]
  if perf in refs or ph in hashes:reason.append('performance_previously_referenced')
  score_previously_profiled=bool(score in refs or sh in hashes)
  raw.append(dict(performance=perf,score=score,work_family=fam,composer=Path(perf).parts[0],performer=performer(perf),performance_sha256=ph,score_sha256=sh,score_previously_profiled=score_previously_profiled,eligible=not reason,reason=';'.join(reason)))
 exposed_family={r['work_family'] for r in raw if not r['eligible']}
 eligible=[r for r in raw if r['eligible'] and r['work_family'] not in exposed_family and len(ann[r['performance']]['performance_beats'])>=30]
 byfam=defaultdict(list)
 for r in eligible:byfam[r['work_family']].append(r)
 chosen=[]
 for fam in sorted(byfam,key=key):
  opts=sorted(byfam[fam],key=lambda r:key(r['performance']))
  pick=next((r for r in opts if r['performer']),None)
  if pick:chosen.append(pick)
 roles={'development':[],'lockbox':[]};composer_counts={k:defaultdict(int) for k in roles};performer_role={}
 for r in chosen:
  if all(len(v)>=N for v in roles.values()):break
  preferred='development' if int(key(r['work_family'])[:2],16)%2==0 else 'lockbox';other='lockbox' if preferred=='development' else 'development'
  assigned=performer_role.get(r['performer'])
  role=assigned or (preferred if len(roles[preferred])<N and composer_counts[preferred][r['composer']]<4 else other)
  if assigned is None and (len(roles[role])>=N or composer_counts[role][r['composer']]>=4):role=other
  if len(roles[role])>=N or composer_counts[role][r['composer']]>=4:continue
  roles[role].append({**r,'role':role});composer_counts[role][r['composer']]+=1;performer_role[r['performer']]=role
 if any(len(v)!=N for v in roles.values()):raise RuntimeError({k:len(v) for k,v in roles.items()})
 assert not ({r['work_family'] for r in roles['development']}&{r['work_family'] for r in roles['lockbox']})
 assert not ({r['performer'] for r in roles['development']}&{r['performer'] for r in roles['lockbox']})
 rows=[]
 for role in ('development','lockbox'):
  for i,r in enumerate(roles[role]):rows.append(dict(case_id=f"{'fd' if role=='development' else 'fl'}{i:03d}",**r))
 OUT.mkdir(parents=True)
 with (OUT/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 audit={'scope':'metadata-only; no tracker outcomes read','salt':SALT,'scanned_text_files':len(scanned),'large_metadata_skipped':large_metadata_skipped,'referenced_midi_paths':len(refs),'referenced_sha256_tokens':len(hashes),'aligned_performances':len(raw),'exposed_work_families':len(exposed_family),'eligible_unexposed_families':len(byfam),'selected':{k:len(v) for k,v in roles.items()},'unique_performers':len({r['performer'] for r in rows}),'work_overlap':0,'performer_overlap':0,'same_alignment_pipeline':'ASAP annotations for both splits','script_sha256':sha(Path(__file__))}
 (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
