import numpy as np
from score_following.observation_evidence import regional_evidence,predict_prior
from score_following.p3_extension import bayes_from_cost


def test_wrong_candidate_is_fixed_before_observation():
    p=np.full(100,.001);p[20]=.5;p[70]=.4;p/=p.sum()
    _,a=regional_evidence(p,np.zeros(100),20,2,10)
    cost=np.ones(100);cost[90]=0
    _,b=regional_evidence(p,cost,20,2,10)
    assert a['wrong_center']==b['wrong_center']
    assert abs(a['observation_log_bayes_factor'])<1e-12


def test_evidence_sign_and_exact_odds_identity():
    p=np.ones(100)/100
    cost=np.ones(100);cost[18:23]=0
    _,r=regional_evidence(p,cost,20,2,10)
    assert r['observation_log_bayes_factor']>0
    assert abs(r['identity_residual'])<1e-12
    cost=1-cost
    _,r=regional_evidence(p,cost,20,2,10)
    assert r['observation_log_bayes_factor']<0


def test_instrumentation_preserves_filter():
    rng=np.random.default_rng(1);cost=rng.random((40,100));p=np.zeros(100);p[0]=1
    got=[]
    for i,row in enumerate(cost):
        if i:p=predict_prior(p,[.2,.65,.1,.05],1e-6)
        if i==0:
            p*=np.exp(-8*(row-row.min()));p/=p.sum()
        else:p,_=regional_evidence(p,row,i,2,10)
        got.append(np.argmax(p))
    np.testing.assert_array_equal(got,bayes_from_cost(cost))
