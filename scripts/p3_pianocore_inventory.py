"""Inspect official PianoCoRe metadata without loading performance bytes/outcomes."""
import csv,hashlib,json,sys,re
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/p3_external_metadata/PianoCoRe_metadata.csv'
OUT=SOURCE.parent

def boolean(v):return str(v).strip().lower() in ('true','1','yes')
def num(v):
    try:return float(v)
    except (ValueError,TypeError):return 0.
def key(row):
    # Conservative metadata-only grouping; source IDs reviewed separately.
    return (row.get('composer','').strip().lower(),row.get('composition','').strip().lower(),row.get('movement','').strip().lower())

def main():
    sys.stdout.reconfigure(encoding='utf-8');csv.field_size_limit(2**30)
    if SOURCE.stat().st_size!=205846408:raise ValueError('metadata download incomplete')
    md5=hashlib.md5(SOURCE.read_bytes()).hexdigest()
    if md5!='873f55178fc7343bcadcc1ef97305b93':raise ValueError('metadata checksum mismatch')
    with SOURCE.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;source_counts=Counter();score_sources=Counter();pieces=defaultdict(set)
        selected=set();score_ids=set();by_composer=Counter();durations=[];total=0;criteria=Counter();cross=Counter();selected_sources=Counter();midi_origins=Counter();astar_origins=Counter()
        for r in reader:
            total+=1;src=r.get('performance_dataset','');score_src=r.get('score_dataset','')
            source_counts[src]+=1;score_sources[score_src]+=1
            astar=boolean(r.get('tier_a_star'));not_asap=src.upper()!='ASAP' and score_src.upper()!='ASAP'
            long=num(r.get('performance_duration'))>=90
            raw_align=bool(r.get('raw_alignment_path'))
            no_duplicate=not boolean(r.get('is_duplicate'))
            if astar:
                criteria['a_star']+=1
                astar_origins[(src,'transcription' if boolean(r.get('is_transcription')) else 'direct_or_unspecified')]+=1
            if astar and not_asap:criteria['a_star_non_asap']+=1;pieces[src].add(key(r))
            if astar and not_asap and long:criteria['a_star_non_asap_90s']+=1
            if astar and not_asap and long and raw_align and no_duplicate:
                criteria['metadata_preflight_rows']+=1;selected.add(key(r));by_composer[r.get('composer','')]+=1;durations.append(num(r.get('performance_duration')))
                score_ids.add((score_src,r.get('score_id','')));cross[(src,score_src)]+=1;selected_sources[src]+=1
                midi_origins['transcription' if boolean(r.get('is_transcription')) else 'direct_or_unspecified']+=1
                if not boolean(r.get('is_transcription')):criteria['non_transcription_rows']+=1
    summary=dict(source_url='https://zenodo.org/records/19186016/files/metadata.csv?download=1',bytes=SOURCE.stat().st_size,md5=md5,
        fields=fields,total_rows=total,performance_sources=dict(source_counts),score_sources=dict(score_sources),criteria=dict(criteria),
        distinct_composer_composition_movement_pairs=len(selected),distinct_source_score_ids=len(score_ids),composers=len(by_composer),
        source_score_pairs={str(k):v for k,v in cross.items()},source_piece_counts={k:len(v) for k,v in pieces.items()},
        selected_performance_sources=dict(selected_sources),selected_midi_origins=dict(midi_origins),
        astar_performance_origin_counts={str(k):v for k,v in astar_origins.items()},
        duration_median_eligible=(sorted(durations)[len(durations)//2] if durations else None),
        caveat='Metadata candidate count only. Composer/title normalization, shared underlying recordings, alignments, license, selection bias and GT quality require review. No tracking outcomes run.')
    (OUT/'inventory.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('fields','performance_sources','score_sources')},indent=2))

if __name__=='__main__':main()
