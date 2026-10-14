"""Generate the compact LBD plot directly from the frozen experiment results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def main():
    midi = read('results/online_relocation_validation_lockbox/result.json')['work_equal']
    audio = read('results/online_relocation_audio_mean_evidence/result.json')
    rows = [
        ('MIDI: frozen delayed', midi['global_minus_delayed'], midi['global_minus_delayed_ci'], '#0072B2'),
        ('Audio: mean evidence', audio['work_equal_improvement'], audio['bootstrap95'], '#D55E00'),
    ]
    plt.rcParams.update({'font.size': 8, 'font.family':'DejaVu Sans', 'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(3.23, 1.42))
    fig.subplots_adjust(left=.10,right=.97,bottom=.29,top=.91)
    ax.axvline(0, color='#777777', lw=.7, linestyle='--')
    for y, (label, mean, interval, color) in zip([1,0],rows):
        ax.errorbar(mean,y,xerr=[[mean-interval[0]],[interval[1]-mean]], fmt='o',color=color,capsize=3,ms=4,lw=1.4)
        ax.text(-.50,y+.21,label,color=color,fontsize=7.5)
        ax.text(.83,y+.21,f'{mean:+.3f}',ha='right',fontsize=7.5,color=color)
    ax.set_xlim(-.53,.86)
    ax.set_ylim(-.37,1.56)
    ax.set_yticks([])
    ax.set_xticks([-.4,0,.4,.8])
    ax.tick_params(axis='x',labelsize=7,length=3)
    ax.set_xlabel('Accept regret - rule regret (score-seconds)',fontsize=7.3,labelpad=3)
    for s in ['left','right','top']: ax.spines[s].set_visible(False)
    out=ROOT/'paper/lbd2026'
    fig.savefig(out/'gain.pdf')
    fig.savefig(out/'gain.png',dpi=220)
    plt.close(fig)
    print('Generated paper/lbd2026/gain.pdf and gain.png')

if __name__ == '__main__':
    main()
