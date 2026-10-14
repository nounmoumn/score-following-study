"""Exact decomposition by whether a restart happened after a chosen landmark."""
import numpy as np
from .observation_evidence import predict_prior


def lineage_step(existing,new,row,transition,beta,restart):
    old_prior=(1-restart)*predict_prior(existing,transition,0.)
    new_prior=(1-restart)*predict_prior(new,transition,0.)+restart/len(row)
    likelihood=np.exp(-beta*(row-row.min()))
    old_post=old_prior*likelihood;new_post=new_prior*likelihood
    total=(old_post+new_post).sum()
    return old_post/total,new_post/total,old_prior+new_prior


def crossing_type(previous,current,hop_score):
    delta=int(current)-int(previous)
    if abs(delta)*hop_score>.5:return 'large_switch'
    return 'local_crossing' if 0<=delta<=3 else 'small_switch'
