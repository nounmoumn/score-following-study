"""Prior-fixed candidate evidence, distinct from posterior confidence.

GT marks the correct region for an oracle diagnostic. Wrong-region selection
uses only the predictive prior, never the new observation or future error.
"""
import numpy as np


def predict_prior(p, transition, restart):
    prior=np.zeros_like(p)
    for step,weight in enumerate(transition):
        if step==0:prior+=weight*p
        else:
            prior[step:]+=weight*p[:-step]
            prior[-1]+=weight*p[-step:].sum()
    return (1-restart)*prior+restart/len(prior)


def regional_evidence(prior,cost,true_center,radius,exclude,beta=8.):
    """Choose strongest prior-mass wrong region; then evaluate its likelihood.

    Returns log(L_correct/L_wrong), positive if observation favors truth.
    L_region = sum_j prior(j|region) exp(-beta*cost(j)).
    Log posterior odds must equal log prior odds plus log likelihood ratio.
    """
    n=len(prior);idx=np.arange(n)
    correct=abs(idx-true_center)<=radius
    prefix=np.concatenate(([0.],np.cumsum(prior)))
    lo=np.maximum(0,idx-radius);hi=np.minimum(n,idx+radius+1)
    masses=prefix[hi]-prefix[lo]
    # Every member of the wrong region lies outside the excluded truth zone.
    eligible=abs(idx-true_center)>exclude+radius
    if not eligible.any():raise ValueError('no disjoint wrong region')
    wrong_center=int(np.flatnonzero(eligible)[np.argmax(masses[eligible])])
    wrong=abs(idx-wrong_center)<=radius
    mc,mw=float(prior[correct].sum()),float(prior[wrong].sum())
    likelihood=np.exp(-beta*(cost-cost.min()))
    lc=float(prior[correct]@likelihood[correct]/mc)
    lw=float(prior[wrong]@likelihood[wrong]/mw)
    posterior=prior*likelihood;posterior/=posterior.sum()
    pc,pw=float(posterior[correct].sum()),float(posterior[wrong].sum())
    log_prior=float(np.log(mc)-np.log(mw))
    evidence=0.0 if np.all(likelihood==likelihood[0]) else float(np.log(lc)-np.log(lw))
    log_post=float(np.log(pc)-np.log(pw))
    return posterior,dict(correct_prior_mass=mc,wrong_prior_mass=mw,wrong_center=wrong_center,
        correct_likelihood=lc,wrong_likelihood=lw,observation_log_bayes_factor=evidence,
        prior_log_odds=log_prior,posterior_log_odds=log_post,
        identity_residual=log_post-log_prior-evidence)
