import itertools
import numpy as np
from score_following.path_evidence import path_evidence
from score_following.observation_evidence import predict_prior


def test_forward_matches_all_paths_with_restart_and_boundary():
    n=4;rng=np.random.default_rng(7);cost=rng.random((4,n));tr=(.2,.65,.1,.05);restart=.02;beta=2.
    regions=[(np.array([0,1]),np.log([.3,.7])),(np.array([2,3]),np.log([.6,.4]))]
    matrix=np.array([predict_prior(np.eye(n)[i],tr,restart) for i in range(n)])
    logs=[]
    for ix,lw in regions:
        initial=np.zeros(n);initial[ix]=np.exp(lw);lik=0.
        for path in itertools.product(range(n),repeat=4):
            prob=initial[path[0]]*np.exp(-beta*sum(cost[t,j] for t,j in enumerate(path)))
            for t in range(1,4):prob*=matrix[path[t-1],path[t]]
            lik+=prob
        logs.append(np.log(lik))
    np.testing.assert_allclose(path_evidence(cost,regions,tr,beta,restart),-(logs[0]-logs[1])/4,atol=1e-12)


def test_positive_negative_and_null_controls():
    regions=[(np.array([10]),np.array([0.])),(np.array([70]),np.array([0.]))]
    cost=np.ones((10,100));cost[:,10:40]=0
    assert path_evidence(cost,regions)<0
    assert path_evidence(1-cost,regions)>0
    assert abs(path_evidence(np.full_like(cost,.7),regions))<1e-12


def test_variable_speed_support_and_common_offset_invariance():
    regions=[(np.array([10]),np.array([0.])),(np.array([70]),np.array([0.]))]
    cost=np.ones((10,100));position=10
    for t in range(10):
        if t:position+=(0,3,1)[t%3]
        cost[t,position]=0
    w=path_evidence(cost,regions)
    assert w<0
    np.testing.assert_allclose(path_evidence(cost+np.arange(10)[:,None],regions),w,atol=1e-12)
