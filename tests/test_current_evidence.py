import numpy as np
from score_following.current_evidence import current_evidence


def test_null_and_correct_wrong_direction():
    p=np.array([.1,.7,.2]);mask=np.array([False,True,False])
    post,m=current_evidence(p,np.ones(3),mask)
    np.testing.assert_allclose(post,p);assert abs(m['B'])<1e-12 and abs(m['D'])<1e-12
    _,m=current_evidence(p,np.array([1.,0.,1.]),mask);assert m['B']<0 and m['D']<0
    _,m=current_evidence(p,np.array([0.,1.,0.]),mask);assert m['B']>0 and m['D']>0
    assert abs(m['identity_residual'])<1e-12


def test_identity_with_low_prior_mass():
    p=np.array([1e-9,1-1e-9]);_,m=current_evidence(p,np.array([0.,1.]),np.array([True,False]))
    assert abs(m['identity_residual'])<1e-12
