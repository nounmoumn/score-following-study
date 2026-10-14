"""Exact max-product representative paths under the unchanged filter kernel."""
import numpy as np

def max_step(previous,transition=(.2,.65,.1,.05),restart=1e-6):
    n=len(previous);eps=restart/n
    best=int(np.argmax(previous))
    score=np.full(n,previous[best]+np.log(eps));parent=np.full(n,best,dtype=np.int32)
    for step,w in enumerate(transition):
        dest=np.arange(step,n);source=dest-step
        values=previous[source]+np.log(eps+(1-restart)*w)
        win=values>score[dest];score[dest[win]]=values[win];parent[dest[win]]=source[win]
    # At last state, every step that passes the boundary is clamped to n-1.
    for source in range(max(0,n-len(transition)),n):
        mass=sum(transition[n-1-source:])
        value=previous[source]+np.log(eps+(1-restart)*mass)
        if value>score[-1]:score[-1]=value;parent[-1]=source
    return score,parent

def backtrace(parents,t,end,width=20):
    path=[end]
    for frame in range(t,t-width+1,-1):path.append(int(parents[frame,path[-1]]))
    return np.array(path[::-1],dtype=int)

def path_features(cost,chroma,pitch,candidates):
    m,n=cost.shape;parents=np.zeros((m,n),dtype=np.int32)
    previous=np.full(n,-np.inf);previous[0]=0.;out={}
    for t,row in enumerate(cost):
        if t:previous,parents[t]=max_step(previous)
        previous=previous-8*(row-row.min());previous-=previous.max()
        if t not in candidates:continue
        if t<19:raise ValueError('Past path unavailable')
        f=candidates[t];a=backtrace(parents,t,f['map']);b=backtrace(parents,t,f['rival'])
        def sim(features):
            x=features[a].ravel();y=features[b].ravel();den=np.linalg.norm(x)*np.linalg.norm(y)
            return float(x@y/den) if den>0 else 0.
        s=sim(chroma);r=s-sim(pitch);mass=f['base'][4]
        out[t]=dict(score=[s,r,s*mass,r*mass],paths=[a.tolist(),b.tolist()])
    return out
