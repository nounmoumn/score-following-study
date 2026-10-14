import numpy as np
from score_following.multicandidate import select_regions,likelihood_trace,logistic_predict
from score_following.frozen_evidence import freeze_regions
from score_following.path_evidence import path_evidence

def test_disjoint_candidates_and_top1_agreement():
    rng=np.random.default_rng(3);p=rng.random(200);p/=p.sum()
    regions,mass,centers=select_regions(p,80,3,10)
    _,oldmass,oldcenter=freeze_regions(p,80,3,10)
    assert centers[1]==oldcenter
    np.testing.assert_allclose(mass[:2],oldmass)
    for i in range(6):
        for j in range(i):assert not set(regions[i][0])&set(regions[j][0])

def test_trace_matches_pair_evidence_and_null():
    p=np.ones(200)/200;regions,_,_=select_regions(p,80,3,10);rng=np.random.default_rng(4);cost=rng.random((20,200))
    ll=likelihood_trace(cost,regions,(.2,.65,.1,.05),8.,1e-6)
    for k in range(1,6):np.testing.assert_allclose(-(ll[-1,0]-ll[-1,k])/20,path_evidence(cost,[regions[0],regions[k]]),atol=1e-12)
    np.testing.assert_allclose(likelihood_trace(np.ones_like(cost),regions,(.2,.65,.1,.05),8.,1e-6),0,atol=1e-12)

def test_logistic_constant_target_rate_and_direction():
    x=np.arange(20.)[:,None];y=(x[:,0]>9).astype(float)
    pred=logistic_predict(x,y,np.array([[0.],[19.]]),np.ones(20))
    assert pred[0]<.5<pred[1]
    pred=logistic_predict(np.zeros((20,1)),np.r_[np.ones(5),np.zeros(15)],np.zeros((2,1)),np.ones(20))
    np.testing.assert_allclose(pred,.25,atol=1e-6)
