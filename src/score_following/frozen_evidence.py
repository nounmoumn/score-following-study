"""Marginal evidence for candidates and weights fixed before an exposure."""
import numpy as np
from scipy.special import logsumexp


def freeze_regions(prior,correct_center,radius,exclude):
    idx=np.arange(len(prior));prefix=np.concatenate(([0.],np.cumsum(prior)))
    masses=prefix[np.minimum(len(prior),idx+radius+1)]-prefix[np.maximum(0,idx-radius)]
    eligible=abs(idx-correct_center)>exclude+radius
    if not eligible.any():raise ValueError('no wrong region')
    wrong_center=int(np.flatnonzero(eligible)[np.argmax(masses[eligible])])
    regions=[];mass=[]
    for center in [correct_center,wrong_center]:
        indices=idx[abs(idx-center)<=radius]
        total=float(prior[indices].sum())
        if not len(indices) or total<=0:raise ValueError('invalid region mass')
        regions.append((indices,np.log(prior[indices]/total)))
        mass.append(total)
    return regions,mass,wrong_center


def fixed_path_evidence(cost,regions,beta):
    """Rows contain exposure only; no future GT or future observations accepted."""
    ll=[]
    for indices,log_weights in regions:
        j=np.minimum(indices[None,:]+np.arange(len(cost))[:,None],cost.shape[1]-1)
        trajectory_cost=cost[np.arange(len(cost))[:,None],j].sum(axis=0)
        ll.append(float(logsumexp(log_weights-beta*trajectory_cost)))
    return -(ll[0]-ll[1])/len(cost)
