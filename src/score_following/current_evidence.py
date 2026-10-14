import numpy as np


def current_evidence(prior,cost,correct,beta=8.):
    likelihood=np.exp(-beta*(cost-cost.min()));posterior=prior*likelihood;posterior/=posterior.sum()
    pc=float(prior[correct].sum());pw=float(prior[~correct].sum())
    if min(pc,pw)<=0:raise ValueError('empty hypothesis mass')
    lc=float(prior[correct]@likelihood[correct]/pc);lw=float(prior[~correct]@likelihood[~correct]/pw)
    qc=float(posterior[correct].sum());qw=float(posterior[~correct].sum())
    # An exactly constant likelihood carries no regional evidence. Summing
    # conditional masses separately can otherwise create spurious signed epsilons.
    evidence=0.0 if np.all(likelihood==likelihood[0]) else float(np.log(lw)-np.log(lc))
    residual=float(np.log(qw)-np.log(qc)-np.log(pw)+np.log(pc)-evidence)
    return posterior,dict(B=evidence,D=qw-pw,prior_wrong=pw,posterior_wrong=qw,
        entropy=float(-np.sum(posterior[posterior>0]*np.log(posterior[posterior>0]))),identity_residual=residual)
