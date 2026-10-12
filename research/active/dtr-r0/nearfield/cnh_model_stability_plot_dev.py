"""Descriptive fixed-model curves and inherited operating points; no fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-model-stability-dev-20261007'


def curve(scores,event,control,deadline):
    positive=np.array([scores[i,2:deadline[i]+1].max() for i in np.flatnonzero(event)])
    negative=scores[control,2:12].ravel()
    values,inverse=np.unique(np.r_[positive,negative],return_inverse=True)
    pc=np.bincount(inverse[:len(positive)],minlength=len(values))[::-1]
    nc=np.bincount(inverse[len(positive):],minlength=len(values))[::-1]
    return np.r_[0,np.cumsum(nc)/len(negative)]*100,np.r_[0,np.cumsum(pc)/len(positive)]*100


def main(shared=False):
    out=ROOT/'artifacts.local/work/cnh-shared-calibration-dev-20261007' if shared else OUT
    policy='shared_calibrated' if shared else 'calibrated'
    result=json.loads((out/'result.json').read_text(encoding='utf8'))
    with np.load(out/'ledger.npz') as saved:z=dict(saved)
    colors=['#2676b8','#c76d27','#38856c']
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharex=True,sharey=True)
    for ax,tag in zip(axes,('original','HB')):
        mask=(z['tag']==tag)&z['evaluable'];event=mask&z['contact'];control=mask&z['control']
        ax.axvspan(0,5,color='#edf2f7',zorder=0)
        methods=[('original_center',0,'#202c39','M3 max')]+[('raw_hb_aug',f,colors[f],f'Raw + HB, model {f}') for f in range(3)]
        for method,fold,color,label in methods:
            score=z[f'score/fold{fold}/{method}']
            x,y=curve(score,event,control,z['deadline'])
            ax.step(x,y,where='post',color=color,lw=1.65,label=label,zorder=2)
        for fold in range(3):
            for method in ('original_center','raw_hb_aug'):
                if shared and method=='original_center' and fold>0:continue
                color='#202c39' if method=='original_center' else colors[fold]
                for cap,marker in (('0.025','o'),('0.050','D')):
                    m=result['metrics'][f'fold{fold}/{policy}/{cap}/{method}'][tag]['all']
                    x=100*m['false_alarm_intervals']/m['control_intervals']
                    y=100*m['timely_excluding_warmup']/m['events']
                    ax.plot(x,y,marker=marker,ms=6,color=color,mfc=color if cap=='0.025' else 'white',mew=1.3,zorder=4)
                    if method=='original_center' and cap=='0.025':
                        ax.annotate('shared cal' if shared else f'old cal {fold}',(x,y),xytext=(5,-14 if fold==0 else -12),textcoords='offset points',fontsize=8,color=color)
        ax.set(title=f'{tag}: {int(event.sum())} events / {int(control.sum())} controls',
               xlabel='Observed false alarm intervals (%)',xlim=(0,9 if shared else 8),ylim=(0,100))
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Timely events excluding warmup (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    prefix='Shared' if shared else 'Old'
    handles.extend([Line2D([],[],color='#555',marker='o',ls='',label=f'{prefix} calibration target 2.5%'),
                    Line2D([],[],color='#555',marker='D',mfc='white',ls='',label=f'{prefix} calibration target 5%')])
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,fontsize=9)
    title='Fixed evaluation scores: shared original-only calibration' if shared else 'Identical new observations: ranking versus inherited calibration'
    fig.suptitle(title,fontsize=14,y=.99)
    fig.text(.5,.018,'Whole-score ties; event deadline peaks vs. clear intervals. Curves are descriptive, not selected deployment thresholds.',ha='center',fontsize=8,color='#555')
    fig.tight_layout(rect=(0,.20,1,.94))
    for suffix in ('png','pdf'):
        stem='model_stability_shared_operating_points' if shared else 'model_stability_operating_points'
        path=out/f'{stem}.{suffix}'
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Saved operating-point figure; CPU plotting only')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shared-calibration',action='store_true')
    main(parser.parse_args().shared_calibration)
