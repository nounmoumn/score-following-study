import numpy as np
from score_following.observation_evidence import predict_prior
from score_following.tracker_path_diagnosis import max_step,path_features

def test_max_step_is_exact_dense_transition_including_restart_and_clamping():
 rng=np.random.default_rng(290)
 for n in [5,13,50]:
  kernel=np.stack([predict_prior(row,(.2,.65,.1,.05),1e-6) for row in np.eye(n)])
  x=rng.normal(size=n)
  expected=x[:,None]+np.log(kernel)
  scores,parents=max_step(x)
  np.testing.assert_allclose(scores,expected.max(axis=0),atol=1e-12)
  np.testing.assert_array_equal(parents,expected.argmax(axis=0))

def test_paths_match_full_dense_recursion_and_are_causal():
 rng=np.random.default_rng(291);n=35;m=50
 cost=rng.random((m,n));c=rng.random((n,12));p=rng.random((n,128))
 candidates={25:dict(map=13,rival=34,base=[0]*4+[.2]),40:dict(map=21,rival=0,base=[0]*4+[.4])}
 full=path_features(cost,c,p,candidates)
 prefix=path_features(cost[:26],c,p,{25:candidates[25]})
 assert full[25]==prefix[25]
 kernel=np.stack([predict_prior(row,(.2,.65,.1,.05),1e-6) for row in np.eye(n)])
 state=np.full(n,-np.inf);state[0]=0;ptr=[]
 for t,row in enumerate(cost):
  if t:
   edges=state[:,None]+np.log(kernel);parent=edges.argmax(0);state=edges.max(0)
  else:parent=np.zeros(n,dtype=int)
  ptr.append(parent);state-=8*(row-row.min());state-=state.max()
  if t in candidates:
   for k,end in enumerate([candidates[t]['map'],candidates[t]['rival']]):
    q=[end]
    for s in range(t,t-19,-1):q.append(int(ptr[s][q[-1]]))
    assert q[::-1]==full[t]['paths'][k]
