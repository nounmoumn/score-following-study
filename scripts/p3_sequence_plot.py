"""Visual comparison of two development alignment maps; no GT modification."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from p3_smd_alignment_pilot import ROOT

def main():
    base=ROOT/'results/p3_external_metadata'
    d=json.loads((base/'onset_sequence_comparison.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,(name,center,title) in zip(axes,[('Giesbrecht02',115.27,'Repeated chord: improved'),('Giesbrecht02',122.67,'Long gap: unresolved')]):
        for suffix,label in [('alignment','Original DTW'),('sequence','One-to-one onset sequence')]:
            m=np.load(base/f'{name}_{suffix}_mapping.npz')
            t=m['performance_onset_sec'];s=m['mapped_score_onset_sec']
            lo=max(0,np.searchsorted(t,center-2)-1);hi=min(len(t),np.searchsorted(t,center+2)+1)
            ax.plot(t[lo:hi],s[lo:hi],'.-',label=label,alpha=.8)
        c=next(c for c in d['cases'] if c['name']==name)
        b=[r for r in c['beat_rows'] if abs(r['performance_sec']-center)<2]
        ax.scatter([r['performance_sec'] for r in b],[r['true_score_sec'] for r in b],marker='x',color='black',label='ASAP beat annotations',zorder=5)
        ax.set_xlim(center-2,center+2);ax.set_title(title);ax.set_xlabel('Performance seconds');ax.set_ylabel('Score seconds');ax.legend(fontsize=7)
    fig.suptitle('Development comparison: same performances, frozen parameters')
    fig.savefig(base/'onset_sequence_comparison.png',dpi=150)

if __name__=='__main__':main()
