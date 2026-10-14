import numpy as np
from score_following.lineage import lineage_step,crossing_type
from score_following.observation_evidence import predict_prior


def test_lineage_reconstructs_original_filter_and_mass():
    rng=np.random.default_rng(2);p=rng.random(20);p/=p.sum();old=p.copy();new=np.zeros(20)
    for row in rng.random((30,20)):
        old,new,prior=lineage_step(old,new,row,(.2,.65,.1,.05),8.,.02)
        expected=predict_prior(p,(.2,.65,.1,.05),.02)
        np.testing.assert_allclose(prior,expected,atol=1e-14)
        p=expected*np.exp(-8*(row-row.min()));p/=p.sum()
        np.testing.assert_allclose(old+new,p,atol=1e-14)
    assert new.sum()>0


def test_no_restart_has_no_new_lineage():
    p=np.ones(10)/10
    old,new,_=lineage_step(p,np.zeros(10),np.arange(10)/10,(.2,.65,.1,.05),8.,0.)
    assert new.sum()==0
    np.testing.assert_allclose(old.sum(),1.)


def test_crossing_types():
    assert crossing_type(100,110,.1)=='large_switch'
    assert crossing_type(100,104,.1)=='small_switch'
    assert crossing_type(100,99,.1)=='small_switch'
    assert crossing_type(100,103,.1)=='local_crossing'
