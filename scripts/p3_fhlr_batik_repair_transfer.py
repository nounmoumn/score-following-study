#!/usr/bin/env python3
"""Run the frozen Batik transfer after outcome-blind bounded clock repair."""
from __future__ import annotations
import sys,warnings
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import p3_fhl_batik_transfer as base
import p3_clock_repair_preflight as repair
from score_following.features import load_notes,notes_to_chroma_stream

OUT=ROOT/'results/p3_fhlr_batik_repair_transfer';MAX_CORRECTION=.15

def repaired_load(case,hop):
 import partitura as pt
 perf=ROOT/case['performance'];score=ROOT/case['score'];match=ROOT/case['match']
 for p,k in ((perf,'performance'),(score,'score'),(match,'match')):
  if base.sha(p)!=case['sha256'][k]:raise AssertionError(f'hash changed: {p}')
 score_clock,raw=repair.raw_clock(match);perf_clock=repair.pava(raw);correction=float(np.max(np.abs(perf_clock-raw)))
 if correction>MAX_CORRECTION or not np.all(np.diff(perf_clock)>0):raise AssertionError(f'clock repair failed: {match}')
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');a=pt.score.unfold_part_maximal(pt.load_musicxml(str(score))[0],ignore_leaps=False).note_array()
 score_notes=[(float(r['onset_beat'])*.5,float(r['onset_beat']+r['duration_beat'])*.5,int(r['pitch'])) for r in a]
 perf_notes=load_notes(perf);pd=max(x[1] for x in perf_notes);sd=max(x[1] for x in score_notes);sh=hop*sd/pd
 u=notes_to_chroma_stream(perf_notes,pd,hop);v=notes_to_chroma_stream(score_notes,sd,sh);truth=np.interp(np.arange(len(u))*hop,perf_clock,score_clock,left=score_clock[0],right=score_clock[-1])/sh
 return u,v,sh,truth,min(pd,float(perf_clock[-1]))

def main():
 base.load=repaired_load;base.OUT=OUT;base.main()
if __name__=='__main__':main()
