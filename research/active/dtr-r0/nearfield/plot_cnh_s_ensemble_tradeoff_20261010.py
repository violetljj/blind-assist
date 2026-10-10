"""Descriptive zoom and budget witnesses from sealed-point hold curve CSV."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cnh_counterfactual_common_dev as C

OUT=C.ROOT/'artifacts.local/work/cnh-s-ensemble-tradeoff-dev-20261010'
curve=pd.read_csv(OUT/'hold_tradeoff_curve.csv')
metrics=C.read(OUT/'metrics.json');seal=C.read(OUT/'sealed_calibration.json')
selected=metrics['reports']['hold/Sensemble/selected']
fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
panels=('all','dark_thin','sign_edge')
colors={'Sensemble':'#0072b2','E':'#009e73','both955':'#d55e00'}
for arm,data in curve.groupby('arm',sort=False):
    for q,height in enumerate(('HEAD','BODY')):
        for ax,group in zip(axes,panels):
            name=height if group=='all' else height+'_'+group
            column=height+'_net' if group=='all' else name+'_net'
            den=metrics['reports']['hold/fixed/old5']['contacts'][name]['denominator']
            points=np.unique(np.column_stack((data.far_clear_increment_pp,100*data[column]/den)),axis=0)
            ax.plot(points[:,0],points[:,1],color=colors[arm],linestyle='-' if q==0 else '--',label=arm+' '+height,linewidth=1.5)
            if arm!='both955':
                point=metrics['reports']['hold/'+arm+'/selected']
                ax.scatter(point['far_clear_increment_pp'],100*point['contacts'][name]['net']/den,s=28,color=colors[arm],marker='o' if q==0 else 's')
for ax,title in zip(axes,('All contact','Dark-thin contact','Sign-edge contact')):
    ax.scatter(0,0,color='black',label='old5',s=35,zorder=4)
    ax.axvline(3,color='grey',linestyle=':',alpha=.7)
    ax.set(xlim=(-.15,6),ylim=(-1,40),xlabel='Far+clear notified-clip increment (pp)',ylabel='Timely net gain (pp)',title=title)
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.savefig(OUT/'tradeoff_curves_zoom.png',dpi=200);fig.savefig(OUT/'tradeoff_curves_zoom.svg');plt.close(fig)
old=C.read(OUT.parent/'cnh-s-ensemble-confirm-dev-20261010/metrics.json')['reports']
prior={}
for split in ('cal','hold'):
    base=old[split+'/fixed/old5']['costs']['far_pass_plus_clear']
    cand=old[split+'/Sensemble/main']['costs']['far_pass_plus_clear']
    prior[split]=dict(old5_notified_clips=base['clips_with_any_notification'],candidate_notified_clips=cand['clips_with_any_notification'],
        clip_denominator=base['clip_denominator'],increment_pp=100*(cand['clip_notification_rate']-base['clip_notification_rate']))
old_drift=prior['hold']['increment_pp']-prior['cal']['increment_pp']
new_drift=selected['far_clear_increment_pp']-seal['calibrations']['Sensemble']['selected_increment_pp']
witnesses=[]
for arm,data in curve.groupby('arm'):
    for budget in (0,1,2,3,5):
        feasible=data[data.far_clear_increment_pp<=budget]
        best=feasible.loc[(feasible.HEAD_net+feasible.BODY_net).idxmax()]
        witnesses.append(dict(arm=arm,descriptive_budget_increment_pp=budget,attained_increment_pp=float(best.far_clear_increment_pp),
            HEAD_net=int(best.HEAD_net),BODY_net=int(best.BODY_net),HEAD_BODY_net=int(best.HEAD_net+best.BODY_net)))
summary=dict(prior_two_family_cal=prior,old_actual_clip_drift_pp=old_drift,
    pool_expected_increment_pp=seal['calibrations']['Sensemble']['selected_increment_pp'],hold_actual_increment_pp=selected['far_clear_increment_pp'],
    pooled_actual_clip_drift_pp=new_drift,absolute_drift_reduction_fraction=1-abs(new_drift)/abs(old_drift),
    causal_boundary='Single successor cohort; family count, pooled mixture and selection metric change together. Reduced drift is observed, not proof two-family cal caused prior failure.',
    descriptive_budget_witnesses=witnesses,hold_selection=False,interpretation='S stronger near1..5pp budgets here; E stronger at zero incremental far+clear notified clips. No universal dominance. Witnesses do not replace pooled selected threshold.')
with (OUT/'descriptive_summary.json').open('x',encoding='utf8') as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
pd.DataFrame(witnesses).to_csv(OUT/'descriptive_budget_witnesses.csv',index=False)
print(json.dumps(summary,ensure_ascii=False))
