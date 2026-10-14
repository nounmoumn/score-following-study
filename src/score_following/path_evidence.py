"""Initial-region conditional observation evidence over all filter paths."""
import numpy as np
from .observation_evidence import predict_prior


def path_evidence(cost,regions,transition=(.2,.65,.1,.05),beta=8.,restart=1e-6):
    logs=[]
    for indices,log_weights in regions:
        p=np.zeros(cost.shape[1]);p[indices]=np.exp(log_weights);total=0.
        for t,row in enumerate(cost):
            if t:p=predict_prior(p,transition,restart)
            p=p*np.exp(-beta*(row-row.min()));z=p.sum()
            if not z>0:raise ValueError('zero likelihood')
            total+=np.log(z);p/=z
        logs.append(total)
    return float(-(logs[0]-logs[1])/len(cost))
