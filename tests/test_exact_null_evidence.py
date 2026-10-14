import numpy as np
from score_following.current_evidence import current_evidence
from score_following.observation_evidence import regional_evidence

def test_exact_null_is_zero_across_uneven_priors():
    rng=np.random.default_rng(20260920)
    for n in (20,100,1000):
        for beta in (0.,8.):
            p=rng.lognormal(0,4,n);p/=p.sum();cost=np.full(n,.37);mask=np.arange(n)<n//3
            post,a=current_evidence(p,cost,mask,beta)
            post2,b=regional_evidence(p,cost,n//3,2,4,beta)
            assert a['B']==b['observation_log_bayes_factor']==0.
            np.testing.assert_array_equal(post,post2)
            np.testing.assert_allclose(post,p,atol=1e-15)

def test_nonflat_small_signal_is_not_thresholded():
    p=np.full(20,.05);mask=np.arange(20)<10;cost=np.ones(20);cost[mask]-=1e-10
    _,a=current_evidence(p,cost,mask)
    assert a['B']<0 and abs(a['B'])<1e-8
    likelihood=np.exp(-8*(cost-cost.min()))
    expected=np.log(p[~mask]@likelihood[~mask]/p[~mask].sum())-np.log(p[mask]@likelihood[mask]/p[mask].sum())
    assert a['B']==expected
