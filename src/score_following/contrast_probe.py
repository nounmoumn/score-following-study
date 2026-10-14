"""Frozen candidate-contrast diagnostic, not a trained alignment system."""
import numpy as np


def piano_roll(notes, duration, hop=.1):
    roll = np.zeros((max(2, int(np.ceil(duration/hop))), 128))
    for start, end, pitch in notes:
        for f in range(max(0,int(start/hop)), min(len(roll)-1,int(np.ceil(end/hop)))+1):
            roll[f,pitch] += max(0., min(end,(f+1)*hop)-max(start,f*hop))
    return roll


def normalize(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-300)


def window(roll, start, width=20):
    raw = np.zeros((width,roll.shape[1]))
    available = max(0,min(width,len(roll)-start))
    raw[:available] = roll[start:start+available]
    raw = normalize(raw)
    if width != 20:
        points = np.linspace(0,width-1,20)
        raw = np.stack([np.interp(points,np.arange(width),raw[:,j]) for j in range(raw.shape[1])],axis=1)
    return raw / max(np.linalg.norm(raw),1e-300)


def weights(templates):
    w = templates.mean(axis=1).var(axis=0)
    return w/w.sum() if w.sum()>0 else np.ones(templates.shape[-1])/templates.shape[-1]


def choose(templates, observed, weight=None):
    errors = (templates-observed[None,:,:])**2
    if weight is not None:
        errors = errors * weight[None,None,:]
    return int(errors.sum(axis=(1,2)).argmin())
