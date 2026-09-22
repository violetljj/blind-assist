"""Descriptive plots from sealed predictions; no model or threshold selection."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from contact_boundary_data import critical_values, camera_boxes


def main():
    repo=Path(__file__).resolve().parents[4]
    root=repo/'artifacts.local/evidence/ba-contact-boundary-20260923'
    result=json.loads((root/'result.json').read_text())
    meta=json.loads((repo/'artifacts.local/evidence/ba-query-occupancy-20260922-prepared/observations/identities.json').read_text())
    geo=json.loads((repo/'artifacts.local/evidence/ba-query-occupancy-20260922-capture/evaluator/geometry.json').read_text())
    evaluation=[m for m in meta if m['split']=='evaluation']
    group=min(m['base_group_id'] for m in evaluation if m['type_id']=='head_hanging_plane')
    chosen=[(i,m) for i,m in enumerate(evaluation) if m['base_group_id']==group and m['frame_in_clip']==5]
    assert len(chosen)==3
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'#f7f8fb','axes.facecolor':'white'})
    fig,axs=plt.subplots(2,2,figsize=(12,8.1))
    colors={'direct':'#257a9e','geometry':'#be6841'}
    for j,mode in enumerate(('direct','geometry')):
        r=result['arms'][mode]
        values=[100*r['both']['recall'],100*r['both']['FPR']]
        bars=axs[0,0].bar(np.arange(2)+(j-.5)*.32,values,.3,label=mode.title(),color=colors[mode])
        axs[0,0].bar_label(bars,fmt='%.1f%%',padding=4,fontsize=9)
        bounds=[100*r['boundaries'][k]['joint_within_5cm'] for k in ('width','horizon')]
        bars=axs[0,1].bar(np.arange(2)+(j-.5)*.32,bounds,.3,label=mode.title(),color=colors[mode])
        axs[0,1].bar_label(bars,fmt='%.1f%%',padding=4,fontsize=9)
    axs[0,0].set(xticks=[0,1],xticklabels=['Recall','False-positive rate'],ylim=(0,104),ylabel='Queries (%)',
                 title='Unseen layouts + unseen width / horizon')
    axs[0,0].legend(frameon=False,loc='upper right')
    axs[0,1].set(xticks=[0,1],xticklabels=['Critical width','First-contact horizon'],ylim=(0,32),ylabel='All finite true boundaries (%)',
                 title='Boundary within 5 cm (missing predictions fail)')
    relation_colors={'INSIDE':'#299b7a','BOUNDARY':'#d79a32','OUTSIDE':'#7d6cb2'}
    width=np.linspace(.2,1.2,51)
    for ax,mode in zip(axs[1],('direct','geometry')):
        preds=np.load(root/(mode+'-evaluation-predictions.npz'))['width_curve'].reshape(-1,2,51)
        threshold=result['selection'][mode]['threshold']
        for i,m in chosen:
            color=relation_colors[m['layout_relation']]
            actual=critical_values(camera_boxes(geo[m['index']]),'width')[1]
            ax.plot(width,preds[i,1],color=color,lw=2,label=m['layout_relation'].title())
            if .2<=actual<=1.2:
                ax.axvline(actual,color=color,ls=':',lw=1.5)
        ax.axhline(threshold,color='#303846',ls='--',lw=1,label='Dev cutoff')
        ax.set(xlim=(.2,1.2),ylim=(-.02,1.02),xlabel='Virtual cross-section width (m)',ylabel='Predicted contact probability',
               title=mode.title()+': matched lateral interventions')
        ax.legend(frameon=False,fontsize=8,loc='lower right')
        ax.grid(alpha=.15)
    fig.suptitle('Counterfactual contact boundary: ranking works, metric boundaries drift',fontsize=15,fontweight='bold',y=.98)
    fig.text(.5,.025,'Consumed simulation Development: 16 evaluation layouts, 576 images. '
        'Dotted lines: exact contact widths. Bottom example selected by metadata, not outcomes.',ha='center',fontsize=9,color='#546173')
    fig.tight_layout(rect=[0,.06,1,.95])
    fig.savefig(root/'comparison.png',dpi=170)
    fig.savefig(root/'comparison.svg')
    (root/'plot-selection.json').write_text(json.dumps(dict(group=group,frame_in_clip=5,
        family='head_hanging_plane',layer='HEAD',horizon=3.,selected_by_metadata_only=True),indent=2)+'\n')


if __name__=='__main__':
    main()
