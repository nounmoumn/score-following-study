"""All-case observed change map, not a prospective failure forecast."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/online_ambiguity_diagnosis_v2'
cases=[json.loads(p.read_text()) for p in sorted(OUT.glob('c[0-9][0-9][0-9].json'))]
r=json.loads((OUT/'result.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(18,13),sharey=True,layout='constrained')
cmap=ListedColormap(['#c64949','#e9edf2','#198a70']);norm=BoundaryNorm([-1.5,-.5,.5,1.5],3)
for i,c in enumerate(cases):
 rows=c['rows'];t=[x['time'] for x in rows]
 axes[0].scatter(t,[i]*len(t),c=[np.nan if x['A'] is None else x['A'] for x in rows],cmap='viridis',vmin=0,vmax=1,s=17,marker='s')
 for ax,a,b in [(axes[1],'bayes_pitch','bayes_chroma'),(axes[2],'lag_chroma','bayes_chroma')]:
  ax.scatter(t,[i]*len(t),c=[x['correct'][a]-x['correct'][b] for x in rows],cmap=cmap,norm=norm,s=17,marker='s')
axes[0].set_yticks(range(len(cases)),[c['case']+' '+c['work'] for c in cases],fontsize=7)
axes[0].invert_yaxis()
for ax,title in zip(axes,['Score ambiguity at ground-truth location','Online Bayes: pitch minus chroma','Chroma: 2-second hindsight minus online']):
 ax.set_title(title,fontsize=11);ax.set_xlabel('Performance time (s)');ax.set_xlim(0,90);ax.grid(axis='x',alpha=.15)
fig.suptitle('Observed diagnosis on all 40 opened performances | green: fixed; red: broken; gray: unchanged\nLeft: yellow = greater ambiguity; right: delayed estimate of PAST position, not a current cursor',fontsize=13)
fig.savefig(OUT/'all_case_map.png',dpi=170);plt.close(fig)
labels=list(r['contrasts']);fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for i,k in enumerate(labels):
 for ax,field in [(axes[0],'within_case_A_slope_per_SD'),(axes[1],'LOOW_MSE_improvement_from_A')]:
  v=r['contrasts'][k][field];lo,hi=v['bootstrap95'];m=v['mean']
  ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#285a86',capsize=3)
for ax in axes:ax.axvline(0,color='gray',lw=1);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(len(labels)),labels,fontsize=8)
axes[1].set_yticks(range(len(labels)),['']*len(labels))
axes[0].set_title('Ambiguity / change relation (per within-case SD)')
axes[1].set_title('Added A: leave-work-out MSE reduction')
fig.suptitle('Development diagnostics; descriptive work-bootstrap 95% intervals')
fig.savefig(OUT/'diagnostic_summary.png',dpi=180);plt.close(fig)
print(OUT/'all_case_map.png');print(OUT/'diagnostic_summary.png')
