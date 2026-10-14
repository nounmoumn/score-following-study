import numpy as np
from score_following.tracker_candidate_diagnosis import trace,pair_similarity
from score_following.p3_extension import bayes_from_cost

def test_features_use_no_future_observations_and_preserve_original_filter():
 rng=np.random.default_rng(987)
 cost=rng.random((75,90));c=rng.random((90,12));p=rng.random((90,128))
 maps,full=trace(cost,c,p,[20,40,60])
 small,limited=trace(cost[:41],c,p,[20,40])
 np.testing.assert_array_equal(maps,bayes_from_cost(cost))
 np.testing.assert_array_equal(maps[:41],small)
 for t in limited:
  assert full[t]['map']==limited[t]['map'] and full[t]['rival']==limited[t]['rival']
  np.testing.assert_array_equal(full[t]['base'],limited[t]['base'])
  np.testing.assert_array_equal(full[t]['score'],limited[t]['score'])
  assert abs(full[t]['map']-full[t]['rival'])>20

def test_same_chroma_can_hide_octave_distinction():
 c=np.zeros((70,12));p=np.zeros((70,128))
 c[:20,0]=1;c[40:60,0]=1
 p[:20,60]=1;p[40:60,72]=1
 np.testing.assert_allclose(pair_similarity(c,19,59),1.,atol=1e-14,rtol=0)
 assert pair_similarity(p,19,59)==0.
