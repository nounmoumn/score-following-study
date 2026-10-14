import numpy as np
from .observation_evidence import predict_prior


def partition_prior(prior,regions):
    components=np.zeros((len(regions)+2,len(prior)));covered=np.zeros(len(prior),bool)
    for k,(ix,_) in enumerate(regions):
        if covered[ix].any():raise ValueError('overlapping regions')
        components[k,ix]=prior[ix];covered[ix]=True
    components[-2,~covered]=prior[~covered]
    return components


def advance(components,row,transition,beta,restart,first=False,origin=False):
    if first:prior=components.copy()
    else:
        prior=np.array([(1-restart)*predict_prior(p,transition,0.) for p in components])
        if origin:prior+=restart*components.sum(axis=1)[:,None]/components.shape[1]
        else:prior[-1]+=restart/components.shape[1]
    prior*=np.exp(-beta*(row-row.min()))[None,:]
    return prior/prior.sum()
