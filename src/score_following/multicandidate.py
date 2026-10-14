import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .observation_evidence import predict_prior


def select_regions(prior,center,radius,exclude,k=5):
    idx=np.arange(len(prior));prefix=np.r_[0.,np.cumsum(prior)]
    masses=prefix[np.minimum(len(prior),idx+radius+1)]-prefix[np.maximum(0,idx-radius)]
    eligible=abs(idx-center)>exclude+radius;centers=[center]
    for _ in range(k):
        options=np.flatnonzero(eligible)
        if not len(options):raise ValueError('insufficient disjoint candidate regions')
        chosen=int(options[np.argmax(masses[options])]);centers.append(chosen);eligible&=abs(idx-chosen)>2*radius
    regions=[];mass=[]
    for c in centers:
        ix=idx[abs(idx-c)<=radius];total=prior[ix].sum()
        if total<=0:raise ValueError('empty candidate')
        regions.append((ix,np.log(prior[ix]/total)));mass.append(total)
    return regions,np.array(mass),centers


def likelihood_trace(cost,regions,transition,beta,restart):
    result=np.zeros((len(cost),len(regions)))
    for k,(ix,lw) in enumerate(regions):
        p=np.zeros(cost.shape[1]);p[ix]=np.exp(lw);ll=0.
        for t,row in enumerate(cost):
            if t:p=predict_prior(p,transition,restart)
            p*=np.exp(-beta*(row-row.min()));z=p.sum();p/=z;ll+=np.log(z);result[t,k]=ll
    return result


def logistic_predict(x,y,test,weights,alpha=1.):
    weights=np.asarray(weights)*len(weights)/np.sum(weights)
    mean=np.average(x,axis=0,weights=weights);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weights));scale[scale<1e-10]=1.
    xx=np.column_stack([np.ones(len(x)),(x-mean)/scale]);tt=np.column_stack([np.ones(len(test)),(test-mean)/scale])
    def objective(coef):
        eta=xx@coef;loss=np.sum(weights*(np.logaddexp(0,eta)-y*eta))+.5*alpha*np.sum(coef[1:]**2)
        grad=xx.T@(weights*(expit(eta)-y));grad[1:]+=alpha*coef[1:]
        return loss,grad
    result=minimize(objective,np.zeros(xx.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
    if not result.success:raise RuntimeError(result.message)
    return expit(tt@result.x)
