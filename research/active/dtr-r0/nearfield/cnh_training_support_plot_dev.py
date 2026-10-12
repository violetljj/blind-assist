"""Plot frozen paired-score decision regions; no fitting or performance experiment."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name] = '4'

import hashlib
import json
from pathlib import Path
import time

# Select the same existing sklearn runtime before loading stored estimators.
import cnh_direction_information_dev as I
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = I.R.WORK/'cnh-training-support-dev-20261007'
ARMS = ('source_repeat','hb_aug')


def main():
    start = time.perf_counter()
    paths = [OUT/'result.json',OUT/'hb_ledger.npz',OUT/'augmentation_inputs.npz']
    result = json.loads(paths[0].read_text(encoding='utf8'))
    with np.load(paths[1],allow_pickle=False) as saved:
        data = {k:saved[k] for k in ('fold','contact','deadline','tag','anchor_id')}
        scores = {arm:saved['score/'+arm] for arm in ARMS}
    with np.load(paths[2],allow_pickle=False) as saved:
        pairs = saved['hb_aug']
        anchor_ids = saved['anchor_id']
    hb_rows = np.flatnonzero(data['tag']=='HB')
    np.testing.assert_array_equal(anchor_ids,data['anchor_id'][hb_rows])
    selected = hb_rows[data['contact'][hb_rows]]
    pair_index = {int(row):i for i,row in enumerate(hb_rows)}
    points = np.array([pairs[pair_index[int(row)],data['deadline'][row]] for row in selected])
    assert points.shape == (36,2) and np.isfinite(points).all()
    low,high = points.min(0),points.max(0)
    padding = .05*(high-low)
    assert (padding>0).all()
    limits = np.stack((low-padding,high+padding))
    x = np.linspace(limits[0,0],limits[1,0],200)
    y = np.linspace(limits[0,1],limits[1,1],200)
    xx,yy = np.meshgrid(x,y)
    grid = np.column_stack((xx.ravel(),yy.ravel()))
    records = {(r['fold'],r['key']):r for r in result['fold_records']}
    panel_records = []
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':11,'pdf.fonttype':42})
    fig,axes = plt.subplots(2,3,figsize=(12.8,8.3),sharex=True,sharey=True)
    with threadpool_limits(limits=4):
        for ri,arm in enumerate(ARMS):
            for fold in range(3):
                if time.perf_counter()-start > 120:
                    raise TimeoutError('Plot-only 120s budget reached')
                ax = axes[ri,fold]
                model_path = OUT/'models'/f'fold{fold}_{arm}.joblib'
                paths.append(model_path)
                model = joblib.load(model_path)
                theta = float(records[fold,f'calibrated/0.025/{arm}']['threshold'])
                prediction = model.predict_proba(grid)[:,1].reshape(xx.shape)
                alarm = prediction>=theta
                ax.contourf(xx,yy,alarm.astype(float),levels=[-.5,.5,1.5],
                            colors=['#f2f4f7','#b5d5ed'],antialiased=False)
                if alarm.any() and not alarm.all():
                    ax.contour(xx,yy,alarm.astype(float),levels=[.5],colors=['#2879b0'],linewidths=.9)
                mask = data['fold'][selected]==fold
                assert int(mask.sum())==12
                pts = points[mask]
                # Verify decision function against frozen per-event deadline score.
                at_deadline = model.predict_proba(pts)[:,1]
                frozen = np.array([scores[arm][row,data['deadline'][row]] for row in selected[mask]])
                np.testing.assert_allclose(at_deadline,frozen,rtol=0,atol=1e-14)
                ax.scatter(pts[:,0],pts[:,1],s=39,c='#172b3a',edgecolors='white',
                           linewidths=.8,zorder=4)
                arm_label = 'Source repeat' if arm=='source_repeat' else 'HB augmentation'
                ax.set_title(f'{arm_label} | Fold {fold}\n12 OOF HB events; threshold = {theta:.4f}',pad=9)
                ax.set_xlim(limits[:,0]);ax.set_ylim(limits[:,1])
                ax.grid(alpha=.16,linewidth=.6)
                ax.set_axisbelow(True)
                if ri==1:ax.set_xlabel('HEAD score (smoothed raw logit)')
                if fold==0:ax.set_ylabel('BODY score (smoothed raw logit)')
                panel_records.append(dict(arm=arm,fold=fold,threshold=theta,
                    oof_contact_events=int(mask.sum()),deadline_alarm_count=int((at_deadline>=theta).sum()),
                    frozen_deadline_score_check=True))
    fig.suptitle('Training support changes the paired-score decision regions',fontsize=16,y=.977)
    fig.text(.5,.929,'Original-calibration 2.5% thresholds; the same 36 held-out HB contacts in both rows',
             ha='center',fontsize=11,color='#40536a')
    handles=[Patch(facecolor='#b5d5ed',edgecolor='#2879b0',label='Alarm region'),
             Patch(facecolor='#f2f4f7',edgecolor='#c8d0d9',label='No-alarm region'),
             Line2D([0],[0],marker='o',linestyle='',color='#172b3a',markeredgecolor='white',
                    markersize=7,label='OOF HB contact at its deadline (position only)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.062),ncol=3,frameon=False)
    fig.text(.5,.033,'Point color does not encode timely detection. A deadline snapshot does not show the full event trajectory.',
             ha='center',fontsize=9,color='#4c5969')
    fig.text(.5,.013,'Consumed synthetic Development; HB structure is exposed during augmentation training on other units.',
             ha='center',fontsize=9,color='#4c5969')
    fig.subplots_adjust(left=.075,right=.983,bottom=.16,top=.86,hspace=.29,wspace=.12)
    for extension in ('png','pdf'):
        path = OUT/f'training_support_decision_regions.{extension}'
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=180,facecolor='white')
    plt.close(fig)
    receipt = dict(status='complete',purpose='Explanation of stored 2D decision regions; no fit or new performance experiment',
        grid_shape=[200,200],axis_limits=limits.tolist(),
        axis_rule='All 36 HB contact deadline scores, global min/max plus 5% range padding per axis',
        point_rule='Only OOF HB contact deadline scores; uniform markers, no timely encoding',
        thresholds='Original calibration 2.5% cap; inherited fold_records thresholds',
        panels=panel_records,threads=4,seconds=time.perf_counter()-start,
        source_sha256={p.relative_to(I.R.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs=['training_support_decision_regions.png','training_support_decision_regions.pdf'])
    with (OUT/'plot_receipt.json').open('x',encoding='utf8') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(status=receipt['status'],seconds=receipt['seconds'],panels=panel_records),indent=2))


if __name__=='__main__':
    main()
