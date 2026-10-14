import numpy as np
from score_following.temporal import (shift_mass,bayes_instrumented,local_weakness,
    healthy_before,confirmed_events,first_sustained_failure,match_events)
from score_following.p3_extension import bayes_from_cost,evidence_at


def test_mass_shift_conserves_edges_and_zero_shift():
    p=np.array([.1,.2,.3,.4])
    np.testing.assert_array_equal(shift_mass(p,0),p)
    for k in [-5,-1,1,5]:
        assert np.isclose(shift_mass(p,k).sum(),1)
        assert np.all(shift_mass(p,k)>=0)


def test_instrumentation_preserves_original_filter_and_prefix():
    rng=np.random.default_rng(8)
    cost=rng.random((50,80)); truth=np.arange(50)
    base=bayes_from_cost(cost)
    got,h,m=bayes_instrumented(cost,truth,3,onset=20,shift=0)
    np.testing.assert_array_equal(base,got)
    shifted,_,_=bayes_instrumented(cost,truth,3,onset=20,shift=5)
    np.testing.assert_array_equal(shifted[:20],base[:20])
    assert np.all((m>=0)&(m<=1)) and np.all(h>=0)


def test_local_measure_matches_existing_past_window():
    rng=np.random.default_rng(4); v=rng.random((150,12)); u=v[:80].copy();truth=np.arange(80)
    expected=evidence_at(u,v,truth,40,window=20,exclude=10,radius=30,frontier=40,band=30)['weakness_local']
    assert abs(local_weakness(u,v,truth,40,20,10,30)-expected)<1e-12
    u[41:]=999
    assert abs(local_weakness(u,v,truth,40,20,10,30)-expected)<1e-12


def test_confirmation_health_and_failure_include_nonfailures():
    w=np.array([-.2,-.01,0,-.2,0,0])
    assert confirmed_events(w,-.05,1)==[2,5]
    e=np.zeros(100);e[50:55]=1
    assert healthy_before(e,49,20,.5)
    assert not healthy_before(e,50,20,.5)
    assert first_sustained_failure(e,0,100,.5,5)==50
    assert first_sustained_failure(e,0,49,.5,5) is None


def test_matching_uses_only_exposure_and_past_state():
    times=np.arange(30.); w=np.full(30,-.2);w[15:17]=0
    cfg=dict(weak_threshold=-.05,strong_threshold=-.15,minimum_gap_sec=10,
             control_separation_sec=10,error_caliper_sec=.1,trend_caliper_sec=.1,tempo_caliper=.2)
    pairs,audit=match_events(times,w,np.zeros(30),np.zeros(30),np.ones(30),np.ones(30,bool),cfg)
    assert pairs==[(16,1)] and audit[0]['status']=='matched'
