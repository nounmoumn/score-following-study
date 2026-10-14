"""Causal candidate-position features; truth and labels are never arguments."""
import numpy as np
from .observation_evidence import predict_prior

def pair_similarity(features,a,b,width=20):
    x=features[a-width+1:a+1].ravel();y=features[b-width+1:b+1].ravel()
    den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den>0 else 0.

def trace(cost,score_chroma,score_pitch,frames):
    targets=set(int(f) for f in frames);p=np.zeros(cost.shape[1]);p[0]=1.
    maps=[];rows={};states=np.arange(len(p))
    for t,v in enumerate(cost):
        prior=predict_prior(p,(.2,.65,.1,.05),1e-6) if t else p
        p=prior*np.exp(-8*(v-v.min()));p/=p.sum()
        a=int(p.argmax());maps.append(a)
        if t not in targets:continue
        eligible=abs(states-a)>20
        if not eligible.any():raise ValueError('No distant candidate')
        b=int(np.flatnonzero(eligible)[np.argmax(p[eligible])])
        conf=float(p[abs(states-a)<=5].sum());mass=float(p[abs(states-b)<=5].sum())
        valid=a>=19 and b>=19
        s=pair_similarity(score_chroma,a,b) if valid else 0.
        r=s-pair_similarity(score_pitch,a,b) if valid else 0.
        nonzero=p[p>0]
        base=[float(p[a]),conf,float(-(nonzero*np.log(nonzero)).sum()/np.log(len(p))),
              float(np.log(max(p[b],1e-300))-np.log(max(p[a],1e-300))),mass,
              abs(b-a)*.1,(a-maps[max(0,t-10)])*.1,(t+1)*.1,float(valid),float(v[a]),float(v[b])]
        score=[s,r,s*mass,r*mass]
        rows[t]=dict(map=a,rival=b,base=base,score=score,valid=bool(valid))
    return np.array(maps),rows
