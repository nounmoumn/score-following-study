#!/usr/bin/env python3
"""Acquire and audit the frozen SMD Brahms supplement for natural-P3 development."""

import base64
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pretty_midi
import requests

ROOT=Path(__file__).resolve().parents[1]
PVM=ROOT/'results/p3_pianovam_file_preflight.json'; PC=ROOT/'results/p3_external_metadata/PianoCoRe_metadata.csv'
SMD=ROOT/'results/p3_external_metadata/smd_preflight.json'; OUT=ROOT/'results/p3_natural_multisource_preflight.json'
TARGET='Brahms_Op010No1_003_20090916-SMD.mid'; SCORE_ID='PDMX_6348501'

def sha(data): return hashlib.sha256(data).hexdigest()
def load(path):
 pm=pretty_midi.PrettyMIDI(str(path)); ns=sorted((n.start,n.end,n.pitch) for i in pm.instruments for n in i.notes); return ns,float(pm.get_end_time())
def cosine(a,b):
 x=np.bincount([n[2]%12 for n in a],minlength=12); y=np.bincount([n[2]%12 for n in b],minlength=12)
 return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)))

def main():
 smd=json.loads(SMD.read_text(encoding='utf-8')); source=next(r for r in smd['rows'] if r['filename']==TARGET)
 offsets={}; found=[]
 with PC.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   split=row['split']; offset=offsets.get(split,0); offsets[split]=offset+1
   if row['score_id']==SCORE_ID and row.get('score_midi_path'):
    found.append((min(float(row.get('raw_precision') or 0),float(row.get('raw_recall') or 0)),row,offset))
 if not found: raise RuntimeError('frozen Brahms score unavailable')
 _,meta,offset=max(found,key=lambda z:z[0])
 params={'dataset':'SyMuPe/PianoCoRe','config':'default','split':meta['split'],'offset':offset,'length':1}
 response=requests.get('https://datasets-server.huggingface.co/rows',params=params,timeout=90); response.raise_for_status()
 row=response.json()['rows'][0]['row']; assert row['score_id']==SCORE_ID
 smidi=base64.b64decode(row['score_midi_bytes']); sxml=base64.b64decode(row['score_xml_bytes'])
 folder=ROOT/'data/pianovam_p3_natural/supplement_smd_brahms'; folder.mkdir(parents=True,exist_ok=True)
 psource=ROOT/'data/smd_direct_midi'/TARGET; ppath=folder/'performance.mid'; ppath.write_bytes(psource.read_bytes())
 spath=folder/'score.mid'; xpath=folder/'score.mxl'; spath.write_bytes(smidi); xpath.write_bytes(sxml)
 pn,pd=load(ppath); sn,sd=load(spath); sim=cosine(pn,sn)
 gates={'duration_ge_90':pd>=90,'performance_notes_ge_200':len(pn)>=200,'score_notes_ge_100':len(sn)>=100,'pitch_class_cosine_ge_0_80':sim>=.80}
 pvm=json.loads(PVM.read_text(encoding='utf-8')); good=[r for r in pvm['rows'] if r['file_eligible']]
 combined={'recordings':len(good)+int(all(gates.values())),'work_families':len({r['rule_index'] for r in good})+int(all(gates.values())),
           'composers':len({r['score_composer'] for r in good}|({'Brahms,_Johannes'} if all(gates.values()) else set()))}
 cohort={'direct_midi_ge_20':combined['recordings']>=20,'work_families_ge_10':combined['work_families']>=10,'composers_ge_8':combined['composers']>=8}
 result={'scope':'Outcome-blind PianoVAM+SMD direct-MIDI natural-P3 development preflight','supplement':{'filename':TARGET,
  'performance_sha256':sha(ppath.read_bytes()),'score_sha256':sha(smidi),'score_xml_sha256':sha(sxml),
  'performance_duration_sec':pd,'score_duration_sec':sd,'performance_notes':len(pn),'score_notes':len(sn),
  'pitch_class_cosine':sim,'gates':gates,'eligible':all(gates.values())},'combined':combined,'cohort_gates':cohort,
  'continue_to_segmentation':all(cohort.values()),'validation_eligible':False}
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
