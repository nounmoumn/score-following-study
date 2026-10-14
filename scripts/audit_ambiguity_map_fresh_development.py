"""Saved-output audit for fresh ambiguity-map development gate."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ambiguity_map_fresh_development'
def main():
 result=json.loads((OUT/'result.json').read_text());cases=[json.loads(p.read_text()) for p in sorted(OUT.glob('fd[0-9][0-9][0-9].json'))];systems=('chroma','pitch','fused');works=sorted({c['work'] for c in cases})
 def wm(fn):return np.array([np.mean([np.mean([fn(r) for r in c['rows']]) for c in cases if c['work']==w]) for w in works])
 summary={s:{'accuracy':float(wm(lambda r,s=s:r['loss'][s]<=.5+1e-12).mean()),'capped_loss':float(wm(lambda r,s=s:r['loss'][s]).mean())} for s in systems}
 for s in systems:
  assert abs(summary[s]['accuracy']-result['systems'][s]['accuracy'])<1e-12;assert abs(summary[s]['capped_loss']-result['systems'][s]['capped_loss'])<1e-12
 assert result['selected_pair'] is None and result['lockbox_authorized'] is False
 freeze=json.loads((OUT/'freeze.json').read_text());drift=[p for p,h in freeze.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h];assert not drift
 manifest=(ROOT/'output/ambiguity_map_fresh_cohort/manifest.csv').read_text();assert not (ROOT/'results/ambiguity_map_fresh_lockbox').exists()
 audit={'status':'PASS','development_cases':len(cases),'development_works':len(works),'rows':sum(len(c['rows']) for c in cases),'selected_pair':None,'lockbox_authorized':False,'lockbox_output_absent':True,'max_summary_residual':0.0,'hash_drift':drift}
 (OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
