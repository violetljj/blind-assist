"""Descriptive twin/real comparison, zero-input diagnostics and report tables."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import dorm_twin_discrimination as D
P=D.P; OUT=D.OUT

def compare():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ds=P.data(); scenes=D.read('scenes.json'); report={}
    bias=np.load(D.FROZEN_BIAS).reshape(64,16)
    gain=P.read('instrument_parameters.json')['amplitude']['gain_real_over_nominal']
    for sid in ('2','3','4','4b','5','6','7','8','9'):
        d=ds[sid]; condition='4' if sid in ('4b','9') else sid
        ranges=np.full(64,np.nan)
        for z in range(64):
            ok=P.good(d)[:,z]&(d['D'][:,z]>150)&d['stable']
            if ok.sum():ranges[z]=np.median(d['D'][ok,z])/1000
        candidates=[s for s in scenes if s['kind']=='condition' and s['condition']==condition]
        errors=[]
        for s in candidates:
            with np.load(OUT/(s['key']+'_input.npz')) as t:tr=t['geometric_range'].reshape(64)[::-1]
            valid=np.isfinite(tr)&np.isfinite(ranges)
            errors.append(dict(key=s['key'],pitch=s['pitch_deg'],zones=int(valid.sum()),median_error_m=float(np.median(abs(tr[valid]-ranges[valid]))) if valid.sum()>=4 else None))
        identified=[e for e in errors if e['median_error_m'] is not None]
        selected=min(identified,key=lambda e:e['median_error_m']) if identified else next(e for e in errors if e['pitch']==-10)
        with np.load(OUT/(selected['key']+'_input.npz')) as t:th=t['H'].reshape(-1,64,16).mean(0);ts=(th-bias)[::-1]
        h=d['H'][d['stable']].mean(0);rawsignal=(d['H']-D.bias3(d['A']))[d['stable']].mean(0)
        real=rawsignal/gain;diff=real-ts
        bins=np.abs(diff).sum(0);top=np.argsort(abs(diff).ravel())[::-1][:8]
        cells=[dict(zone=int(i//16),bin=int(i%16),difference=float(diff.ravel()[i]),real=float(real.ravel()[i]),twin=float(ts.ravel()[i])) for i in top]
        report[sid]=dict(pitch_status='RANGE_PROXY_SELECTED' if identified else 'NOT_IDENTIFIED',selected=selected,all_pitch_errors=errors,
                        valid_real_zones=int(np.isfinite(ranges).sum()),difference_by_bin=bins.tolist(),top_bins=np.argsort(bins)[::-1][:5].tolist(),top_cells=cells)
        np.savez_compressed(OUT/(sid+'_comparison.npz'),real_raw=h,twin_raw=th,real_signal=real,twin_signal_mapped=ts,twin_signal_raw=th-bias,difference=diff,firmware_range=ranges)
        fig,ax=plt.subplots(2,3,figsize=(13,7));v=max(np.percentile(abs(real),98),np.percentile(abs(ts),98),1)
        for a,x,title in zip(ax[0],[real,ts,diff],['Real (H-B3) / effective gain','Twin signal: fixed 180-degree map','Real - twin (weak units)']):
            im=a.imshow(x,aspect='auto',cmap='coolwarm',vmin=-v,vmax=v);a.set(title=title,xlabel='Aggregated bin',ylabel='Raw real zone');fig.colorbar(im,ax=a)
        ax[1,0].plot(h.mean(0),label='real raw');ax[1,0].plot(th.mean(0),label='twin raw');ax[1,0].legend();ax[1,0].set_title('Unmapped means, distinct instrument units')
        ax[1,1].bar(range(16),bins);ax[1,1].set_title('Sum of absolute signal difference')
        ax[1,2].imshow(ranges.reshape(8,8),vmin=0,vmax=4);ax[1,2].set_title('Firmware valid range (m)')
        fig.suptitle(f'Condition {sid}: pitch {selected["pitch"]} deg, {report[sid]["pitch_status"]}; geometry unmeasured')
        fig.tight_layout();fig.savefig(OUT/(sid+'_zone_comparison.png'),dpi=140);plt.close(fig)
    D.save('comparison_report.json',report)
    # Coverage/uncertainty is per cell, no spatial imputation.
    m=D.read('mosaic_bias_report.json');a=np.load(OUT/'mosaic_bias_arrays.npz')
    fig,axs=plt.subplots(1,3,figsize=(12,5))
    for ax,x,title in zip(axs,[a['coverage'][:,:4],a['bias'][:,:4],a['beta'][:,:4]],['Selected frames / cell','Intercept at local ambient reference','Ambient slope (scene-confounded)']):
        im=ax.imshow(x,aspect='auto');ax.set(title=title,xlabel='Near bin',ylabel='Zone');fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(OUT/'mosaic_coverage.png',dpi=140);plt.close(fig)
    print('COMPARED',[(s,r['selected']['pitch'],r['top_bins']) for s,r in report.items()],flush=True)

def counterfactual(cap):
    from cnh_sim2real_frozen import FrozenReplay,segments
    start=time.monotonic();engine=None;rows=[];ds=P.data();comparison=D.read('comparison_report.json')
    def check():
        if time.monotonic()-start>cap:raise TimeoutError('Contribution GPU command cap')
    try:
        engine=FrozenReplay(batch_size=32)
        for sid in ('4','4b','7','9'):
            d=ds[sid];groups,_=segments(d['seq'],d['ms']);ids=next(np.flatnonzero(groups==g)[:16] for g in np.unique(groups) if (groups==g).sum()>=16)
            for path in ('v2','v3'):
                z=D.real_input(d,path)[ids].reshape(16,8,8,16);base=engine.infer(z,d['seq'][ids],d['ms'][ids],check);ready=base['ready']
                targets=[dict(kind='bin',bin=b) for b in range(16)]+[dict(kind='cell',zone=c['zone'],bin=c['bin']) for c in comparison[sid]['top_cells']]
                for target in targets:
                    changed=z.copy()
                    if target['kind']=='bin':changed[...,target['bin']]=0
                    else:changed.reshape(16,64,16)[:,target['zone'],target['bin']]=0
                    cf=engine.infer(changed,d['seq'][ids],d['ms'][ids],check)
                    row=dict(condition=sid,path=path,first_frame=int(ids[0]),ready=int(ready.sum()),**target)
                    for q,name in enumerate(('HEAD','BODY')):
                        row[name]={}
                        for arm,field in [('m3','m3_score'),('local','local_score'),('S','s_score')]:
                            row[name][arm+'_score_delta']=float(np.mean(base[field][ready,q]-cf[field][ready,q]))
                        row[name]['old5_cancel']=int((base['old5'][ready,q]&~cf['old5'][ready,q]).sum())
                        row[name]['old5_add']=int((~base['old5'][ready,q]&cf['old5'][ready,q]).sum())
                        row[name]['baseline_old5']=int(base['old5'][ready,q].sum());row[name]['zero_old5']=int(cf['old5'][ready,q].sum())
                        row[name]['m3_cancel']=int((base['m3'][ready,q]&~cf['m3'][ready,q]).sum())
                        row[name]['S_light_base']=int((base['S_grade'][ready,q]==1).sum());row[name]['S_light_zero']=int((cf['S_grade'][ready,q]==1).sum())
                    rows.append(row)
                print('CONTRIBUTION',sid,path,'elapsed',round(time.monotonic()-start,2),flush=True)
        D.save('contribution_report.json',dict(status='COMPLETE',seconds=time.monotonic()-start,rows=rows,diagnostic_only=True,resources_released=True))
    finally:
        if engine:engine.close()

def tables():
    r=D.read('inference_report.json');scenes=D.read('scenes.json');rows=[]
    for s in scenes:
        for geometry,readout in r['twin'][s['key']].items():
            for arm,queries in readout['arms'].items():
                for q in queries:rows.append(dict(key=s['key'],kind=s['kind'],condition=s['condition'],pitch=s['pitch_deg'],gap_m=s['gap_m'],profile=s['profile'],geometry=geometry,arm=arm,ready=readout['ready'],**q))
    with (OUT/'twin_alarm_table.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rows=[]
    for sid,paths in r['real'].items():
        for path,readout in paths.items():
            for arm,qs in readout['arms'].items():
                for q in qs:rows.append(dict(condition=sid,path=path,arm=arm,ready=readout['ready'],**q))
    with (OUT/'real_alarm_table.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for q,ax in enumerate(axes):
        for gap in (.2,.25,.3,.4,.55):
            cells=[s for s in scenes if s['condition']=='4' and s['gap_m']==gap and s['kind'] in ('condition','gap')]
            ax.plot([s['pitch_deg'] for s in cells],[100*r['twin'][s['key']]['matched']['arms']['old5'][q]['strong_rate'] for s in cells],marker='o',label=f'{gap*100:g} cm')
        ax.set(title=('HEAD','BODY')[q],xlabel='Pitch (deg)',ylabel='Strong (%)',ylim=(-3,103));ax.legend(title='Gap from query edge')
    fig.tight_layout();fig.savefig(OUT/'gap_pitch_alarm.png',dpi=140);plt.close(fig)
    print('TABLES_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['compare','counterfactual','tables']);p.add_argument('--cap',type=float,default=120);a=p.parse_args()
    globals()[a.stage](a.cap) if a.stage=='counterfactual' else globals()[a.stage]()
