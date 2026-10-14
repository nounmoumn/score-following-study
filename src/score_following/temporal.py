"""Past-only P3 event rules and Bayesian state-calibration instrumentation."""
import numpy as np


def shift_mass(posterior, shift):
    out = np.zeros_like(posterior)
    destination = np.clip(np.arange(len(out))+shift, 0, len(out)-1)
    np.add.at(out, destination, posterior)
    return out


def bayes_instrumented(cost, truth, radius, *, onset=None, shift=0,
                       transition=(.2,.65,.1,.05), beta=8., restart=1e-6):
    m, n = cost.shape
    p = np.zeros(n)
    p[0] = 1
    estimates, entropy, mass = np.zeros(m, int), np.zeros(m), np.zeros(m)
    for i in range(m):
        if i == onset:
            p = shift_mass(p, shift)
        if i:
            prior = np.zeros(n)
            for step, weight in enumerate(transition):
                if step==0:
                    prior += weight*p
                else:
                    prior[step:] += weight*p[:-step]
                    prior[-1] += weight*p[-step:].sum()
            p = (1-restart)*prior+restart/n
        p *= np.exp(-beta*(cost[i]-cost[i].min()))
        p /= p.sum()
        estimates[i] = np.argmax(p)
        nonzero = p[p>0]
        entropy[i] = -np.sum(nonzero*np.log(nonzero))
        j = int(round(truth[i]))
        mass[i] = p[max(0,j-radius):min(n,j+radius+1)].sum()
    return estimates, entropy, mass


def local_weakness(u, v, truth, i, window, exclude, radius):
    indices = np.rint(truth[i-window+1:i+1]).astype(int)
    center = indices[-1]
    offsets = indices-center
    ends = np.arange(max(-offsets.min(), center-radius), min(len(v)-offsets.max(), center+radius+1))
    templates = v[ends[:,None]+offsets[None,:]].reshape(len(ends),-1)
    query = u[i-window+1:i+1].reshape(-1)
    denominator = np.linalg.norm(templates,axis=1)*np.linalg.norm(query)
    similarity = np.divide(templates@query,denominator,out=np.zeros(len(ends)),where=denominator>0)
    correct = similarity[ends==center][0]
    wrong = abs(ends-center)>exclude
    return float(similarity[wrong].max()-correct)


def healthy_before(error, i, history, threshold):
    # At query i, include the completed current frame and previous history.
    return i-history+1>=0 and bool(np.max(error[i-history+1:i+1])<=threshold)


def confirmed_events(weakness, threshold, minimum_gap_samples):
    events = []
    for i in range(2,len(weakness)):
        if weakness[i]>=threshold and weakness[i-1]>=threshold and weakness[i-2]<threshold:
            if not events or i-events[-1]>=minimum_gap_samples:
                events.append(i)
    return events


def match_events(times, weakness, current_error, trend, tempo, healthy, cfg):
    """No future outcome argument exists: selection only sees current/past state."""
    step = float(times[1]-times[0])
    events = confirmed_events(weakness,cfg['weak_threshold'],int(np.ceil(cfg['minimum_gap_sec']/step)))
    controls = [i for i in range(1,len(times)) if healthy[i] and
                max(weakness[i],weakness[i-1])<=cfg['strong_threshold']]
    selected, audit = [], []
    used = []
    for e in events:
        status = 'unhealthy'
        candidates = []
        if healthy[e]:
            status = 'unmatched'
            for c in controls:
                if abs(times[e]-times[c])<cfg['control_separation_sec']:
                    continue
                if any(abs(times[z]-times[u])<cfg['control_separation_sec'] for z in [e,c] for u in used):
                    continue
                differences = np.array([abs(current_error[e]-current_error[c])/cfg['error_caliper_sec'],
                    abs(trend[e]-trend[c])/cfg['trend_caliper_sec'],
                    abs(tempo[e]-tempo[c])/cfg['tempo_caliper']])
                if np.max(differences)<=1:
                    candidates.append((float(differences@differences),c))
            if candidates:
                _, c = min(candidates)
                selected.append((e,c))
                used.extend([e,c])
                status = 'matched'
        audit.append(dict(event_index=e,event_time_sec=float(times[e]),status=status))
    return selected, audit


def first_sustained_failure(error, start, end, threshold, duration):
    for i in range(start,end-duration+1):
        if np.all(error[i:i+duration]>threshold):
            return i
    return None
