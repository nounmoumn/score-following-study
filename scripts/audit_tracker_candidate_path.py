import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 cdir=ROOT/'results/tracker_candidate_diagnosis';pdir=ROOT/'results/tracker_path_diagnosis'
 result={'status':'PASS','rows':0,'near_wrong_le_2_score_sec':0,'far_wrong_gt_2_score_sec':0,'near_fixed':0,'far_fixed':0,'path_endpoints_checked':0}
 for folder in [cdir,pdir]:
  h=json.loads((folder/'freeze.json').read_text())
  assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in h.items())
 for path in sorted(cdir.glob('c[0-9][0-9][0-9].json')):
  a=json.loads(path.read_text());b=json.loads((pdir/path.name).read_text())
  assert len(a['rows'])==len(b['rows'])
  for x,y in zip(a['rows'],b['rows']):
   for k in ['time','truth','frame','correct','errors','positions']:assert x[k]==y[k]
   for k in ['map','rival','base']:assert x['candidate'][k]==y['candidate'][k]
   for end,history in zip([x['candidate']['map'],x['candidate']['rival']],y['paths']):
    assert len(history)==20 and history[-1]==end and min(history)>=0
    result['path_endpoints_checked']+=1
   result['rows']+=1
   if not x['correct']['bayes_chroma']:
    close=x['errors']['bayes_chroma']<=2
    result['near_wrong_le_2_score_sec' if close else 'far_wrong_gt_2_score_sec']+=1
    if x['correct']['bayes_pitch']:result['near_fixed' if close else 'far_fixed']+=1
 assert result['rows']==4889
 cr=json.loads((cdir/'result.json').read_text());pr=json.loads((pdir/'result.json').read_text())
 for k in cr['comparisons']:assert cr['comparisons'][k]['mse']['base']==pr['comparisons'][k]['mse']['base']
 result['hash_drift']=0
 (pdir/'independent_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
