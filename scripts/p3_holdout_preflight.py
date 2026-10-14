"""Eligibility/lineage audit only: never loads held-out features or tracker outcomes."""
import csv,json,sys,hashlib
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from score_following.features import load_notes
from score_following.onset import work_family
from p3_temporal import read_csv,write_csv

def main():
    sys.stdout.reconfigure(encoding='utf-8');root=ROOT/'asap_dataset';out=ROOT/'results/p3_onset_development/holdout_preflight';out.mkdir(exist_ok=True)
    meta=read_csv(root/'metadata.csv');annotations=json.loads((root/'asap_annotations.json').read_text(encoding='utf-8'))
    used=set(json.loads((ROOT/'results/p3_inventory/used_scores.json').read_text(encoding='utf-8')))
    used_families={work_family(r['folder']) for r in meta if r['midi_score'] in used}
    buckets=defaultdict(list)
    for r in meta:
        family=work_family(r['folder'])
        if family not in used_families:buckets[family].append(r)
    rows=[];selected=[];files=[]
    for family,group in sorted(buckets.items()):
        family_eligible=[]
        for r in group:
            a=annotations.get(r['midi_performance'],{});beats=a.get('performance_beats',[]);reason=None
            if not a.get('score_and_performance_aligned'):reason='not_aligned'
            elif not beats or min(beats)>15 or max(beats)<90:reason='annotation_coverage'
            elif not (root/r['midi_performance']).exists() or not (root/r['midi_score']).exists():reason='missing_midi'
            elif (root/r['midi_performance']).resolve()==(root/r['midi_score']).resolve():reason='performance_is_score'
            else:
                notes=load_notes(root/r['midi_performance'])
                if not notes or max(n[1] for n in notes)<90:reason='short_performance'
            item=dict(work_family=family,composer=r['composer'],folder=r['folder'],score=r['midi_score'],performance=r['midi_performance'],reason=reason or 'eligible')
            rows.append(item)
            if reason is None:family_eligible.append(item)
        if family_eligible:
            chosen=family_eligible[0];selected.append(chosen);files.extend([root/chosen['score'],root/chosen['performance']])
    write_csv(out/'all_candidates.csv',rows);write_csv(out/'eligible_first_per_family.csv',selected)
    # Cross-check against explicit old manifests rather than using outcome-dependent files.
    manifests=[ROOT/'results/p3_extension_20260917_234336/manifest.csv',ROOT/'results/p3_temporal_20260918_000058/manifest.csv',ROOT/'results/p3_state_conditioned_20260918_003032/manifest.csv']
    old_families={work_family(str(Path(r['score']).parent).replace('\\','/')) for f in manifests for r in read_csv(f)}
    collision=sorted({r['work_family'] for r in selected}&old_families)
    if collision:raise AssertionError(f'family leak {collision}')
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    identical=[r['work_family'] for r in selected if hashes[str((root/r['score']).relative_to(ROOT))]==hashes[str((root/r['performance']).relative_to(ROOT))]]
    if identical:raise AssertionError(f'performance content identical to score: {identical}')
    summary=dict(candidate_families=len(buckets),eligible_families=len(selected),eligible_per_composer=dict(Counter(r['composer'] for r in selected)),
        eligible_rows=sum(r['reason']=='eligible' for r in rows),reason_counts=dict(Counter(r['reason'] for r in rows)),
        explicit_manifest_family_collision=collision,
        sha256=hashes,
        scope='Metadata, annotation coverage, MIDI duration and hashes only; no audio/chroma/cost/path/outcome computed.')
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='sha256'},indent=2))

if __name__=='__main__':main()
