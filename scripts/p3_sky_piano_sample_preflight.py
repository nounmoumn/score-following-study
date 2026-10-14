#!/usr/bin/env python3
"""Outcome-blind acquisition and quality audit of public SKY-Piano samples."""
from __future__ import annotations
import hashlib,json,urllib.request
from pathlib import Path
import numpy as np,pretty_midi
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/p3_sky_piano_sample_preflight';DATA=ROOT/'data/sky_piano_samples'
BASE='https://joonhyungbae.github.io/skypiano/explorer';MANIFEST_URL=BASE+'/Data/manifest.json';FILES=('midi.mid','score.mxl','alignment.json')
KNOWN_OVERLAP={'Schubert_Op90_No4':'ASAP work overlap','Chopin_Prelude_Op28_No4':'ASAP work overlap','Mozart_K545':'ASAP sonata-family overlap'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(url,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():urllib.request.urlretrieve(url,path)
 return path
def main():
 OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True);manifest=json.loads(urllib.request.urlopen(MANIFEST_URL).read());rows=[]
 for sample in manifest['samples']:
  tid=sample['trialId'];folder=DATA/tid;paths={name:fetch(f"{BASE}/samples/{tid}/{name}",folder/name) for name in FILES};a=json.loads(paths['alignment.json'].read_text());matches=a['matches'];by_score={}
  for x in matches:by_score.setdefault(float(x['score_onset_quarter']),[]).append(float(x['midi_time_sec']))
  q=np.asarray(sorted(by_score));t=np.asarray([np.median(by_score[x]) for x in q]);duration=float(pretty_midi.PrettyMIDI(str(paths['midi.mid'])).get_end_time());key=next((k for k in KNOWN_OVERLAP if k in tid),None);rows.append({'trial_id':tid,'piece_label':sample['pieceLabel'],'level':sample['level'],'duration_sec':duration,'alignment_stats':a['stats'],'matched_score_onsets':len(q),'clock_nonpositive_steps':int(np.sum(np.diff(t)<=0)),'clock_min_step_sec':float(np.min(np.diff(t))) if len(t)>1 else None,'history_exclusion':KNOWN_OVERLAP.get(key),'files':{n:str(p.relative_to(ROOT)).replace('\\','/') for n,p in paths.items()},'sha256':{n:sha(p) for n,p in paths.items()},'pilot_eligible':bool(duration>=90 and len(q)>=100 and np.all(np.diff(t)>0))})
 result={'scope':'SKY-Piano public samples; outcome-blind file/alignment audit only','source':MANIFEST_URL,'counts':{'samples':len(rows),'duration_ge_90':sum(r['duration_sec']>=90 for r in rows),'monotone':sum(r['clock_nonpositive_steps']==0 for r in rows),'pilot_eligible':sum(r['pilot_eligible'] for r in rows),'nonoverlap_pilot_eligible':sum(r['pilot_eligible'] and not r['history_exclusion'] for r in rows)},'rows':rows,'tracker_outcomes_opened':False}
 (OUT/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
