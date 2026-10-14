import numpy as np
from score_following.frozen_evidence import freeze_regions,fixed_path_evidence


def test_fixed_candidates_and_uninformative_exposure():
    prior=np.full(100,.001);prior[20]=.5;prior[70]=.4;prior/=prior.sum()
    regions,masses,wrong=freeze_regions(prior,20,2,10)
    assert abs(wrong-70)<=2
    assert masses[0]>masses[1]
    assert abs(fixed_path_evidence(np.ones((20,100)),regions,8))<1e-12


def test_trajectory_evidence_direction_and_no_future_use():
    prior=np.ones(100)/100;regions,_,_=freeze_regions(prior,20,1,10)
    cost=np.ones((10,100))
    indices=regions[0][0]
    for t in range(10):cost[t,indices+t]=0
    assert fixed_path_evidence(cost,regions,8)<0
    assert fixed_path_evidence(1-cost,regions,8)>0
    # Function only receives the exposure; a caller's longer stream is never read.
    stream=np.vstack([cost,np.zeros((20,100))])
    value=fixed_path_evidence(stream[:10],regions,8)
    stream[10:]=999
    assert fixed_path_evidence(stream[:10],regions,8)==value
