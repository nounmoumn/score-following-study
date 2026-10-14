import itertools

import numpy as np

from score_following.anchor_feasibility import smooth, choose
from score_following.observation_evidence import predict_prior


def test_smoothing_matches_enumerated_joint_with_and_without_anchor():
    cost = np.random.default_rng(3).random((4, 5))
    transition, restart, beta = (.2,.65,.1,.05), .03, 2.
    matrix = np.stack([predict_prior(np.eye(5)[i],transition,restart) for i in range(5)])
    for anchors in ({}, {2:3}):
        reference = np.zeros_like(cost)
        for path in itertools.product(range(5), repeat=4):
            if path[0] != 0 or any(path[t] != j for t,j in anchors.items()):
                continue
            mass = np.exp(-beta*sum(cost[t,j] for t,j in enumerate(path)))
            mass *= np.prod([matrix[path[t-1],path[t]] for t in range(1,4)])
            for t,j in enumerate(path):
                reference[t,j] += mass
        reference /= reference.sum(1,keepdims=True)
        np.testing.assert_allclose(smooth(cost,anchors,transition,beta,restart),reference,atol=1e-12)


def test_selector_budget_spacing_and_sham_determinism():
    times = np.arange(15,86,dtype=float)
    a, u = np.linspace(0,1,len(times)), np.linspace(1,0,len(times))
    for policy in ("uniform","ambiguity","uncertainty","joint","sham_joint","random"):
        x = choose(times,a,u,4,policy,11)
        assert len(set(x)) == 4
        assert np.diff(np.sort(times[x])).min() >= 5
        np.testing.assert_array_equal(x,choose(times,a,u,4,policy,11))
