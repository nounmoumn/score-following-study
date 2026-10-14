import numpy as np
from score_following.partition_lineage import partition_prior,advance
from score_following.multicandidate import select_regions,likelihood_trace
from score_following.observation_evidence import predict_prior


def test_both_partitions_reconstruct_filter_and_origin_odds():
    rng=np.random.default_rng(17);p=rng.random(100);p/=p.sum();regions,mass,_=select_regions(p,40,2,6)
    c=partition_prior(p,regions);o=c.copy();np.testing.assert_allclose(c.sum(axis=0),p)
    cost=rng.random((12,100));ll=likelihood_trace(cost,regions,(.2,.65,.1,.05),8.,.01)
    for t,row in enumerate(cost):
        c=advance(c,row,(.2,.65,.1,.05),8.,.01,first=t==0)
        o=advance(o,row,(.2,.65,.1,.05),8.,.01,first=t==0,origin=True)
        if t:p=predict_prior(p,(.2,.65,.1,.05),.01)
        p*=np.exp(-8*(row-row.min()));p/=p.sum()
        np.testing.assert_allclose(c.sum(axis=0),p,atol=1e-13)
        np.testing.assert_allclose(o.sum(axis=0),p,atol=1e-13)
    np.testing.assert_allclose(np.log(o[1:6].sum(axis=1)/o[0].sum()),np.log(mass[1:]/mass[0])+ll[-1,1:]-ll[-1,0],atol=1e-12)
    assert c[-1].sum()>0 and o[-1].sum()==0
