"""Generate descriptive pilot tables and scientific figures from retained replay."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import real_pilot_calibration as C

NAMES={'1':'Balcony','2':'Near wall','3':'Far wall','4':'Bright empty','4b':'Dark empty','5':'Near rod + cylinder','6':'Far rod + cylinder','7':'Outside rod','8':'Head bar','9':'Reboot empty'}


def make():
    report=C.read('replay_report.json');p=C.read('instrument_parameters.json');inventory=C.read('inventory.json');ds=C.data()
    figdir=C.OUT/'figures';figdir.mkdir(exist_ok=True)
    summary=[];ledger=[];examples=[]
    table=['|条件|满窗/总帧|v1 M3强 H/B|v2 M3强 H/B|v1 原5格/S强 H/B|v2 原5格/S强 H/B|S轻 H/B（v1；v2）|',
           '|---|---:|---:|---:|---:|---:|---:|']
    conform=['|条件|预期 H/B|v1 S符合/不符合/不可判（H；B）|v2 S符合/不符合/不可判（H；B）|', '|---|---|---:|---:|']
    stable=['|条件|宽松稳定帧/总帧|占比|距离+回波窗 / 仅回波窗|严格稳定帧|', '|---|---:|---:|---:|---:|']
    fixed=['|条件|zone34 bin1/2 中位|相对同bin跨zone中位的残差|超过3×空间MAD|', '|---|---:|---:|---|']
    strict=json.loads((C.OUT.parent/'inventory.json').read_text())
    strict_stable={r['id']:r['stable_frames'] for r in strict}
    def pct(v):return '/'.join(f"{x['strong_rate']*100:.1f}%" for x in v)
    def cnt(v):return '；'.join(f"{x['conform']}/{x['nonconform']}/{x['unknown']}" for x in v)
    for row in inventory:
        sid=row['id'];d=ds[sid];paths=report['segments'][sid];one,two=paths['v1'],paths['v2']
        stable.append(f"|{sid}|{row['stable_frames']}/{row['frames']}|{row['stable_fraction']*100:.1f}%|{row['stable_windows']-row['echo_only_windows']} / {row['echo_only_windows']}|{strict_stable[sid]}|")
        fp=p['fixed_pattern'][sid]['row4_col2_bins12']
        fixed.append(f"|{sid}|{fp['median'][0]:.3f} / {fp['median'][1]:.3f}|{fp['spatial_residual'][0]:.3f} / {fp['spatial_residual'][1]:.3f}|{fp['above_3_spatial_MAD']}|")
        table.append(f"|{sid}|{one['ready']}/{row['frames']}|{pct(one['arms']['m3'])}|{pct(two['arms']['m3'])}|{pct(one['arms']['old5'])}|{pct(two['arms']['old5'])}|"+
                     '/'.join(f"{v['light_rate']*100:.1f}%" for v in one['arms']['Sensemble'])+'；'+'/'.join(f"{v['light_rate']*100:.1f}%" for v in two['arms']['Sensemble'])+'|')
        conform.append(f"|{sid}|{' / '.join(v['expectation'] for v in one['arms']['Sensemble'])}|{cnt(one['arms']['Sensemble'])}|{cnt(two['arms']['Sensemble'])}|")
        scores={path:dict(np.load(C.OUT/f'{sid}_{path}_scores.npz')) for path in ('v1','v2')}
        for path,rs in paths.items():
            r=scores[path]
            for arm,queries in rs['arms'].items():
                for q,v in enumerate(queries):
                    summary.append(dict(condition=sid,path=path,model=arm,ready=rs['ready'],warmup=rs['warmup'],**v))
                    grade=r['S_grade'][:,q] if arm=='Sensemble' else np.where(r[arm][:,q],2,0)
                    for i in range(len(grade)):
                        exp=C.expected(sid,q);g=int(grade[i]);state='WARMUP' if not r['ready'][i] else 'UNKNOWN' if exp=='UNKNOWN' else 'CONFORM' if (g==0 if exp=='SILENT' else g<2 if exp=='NO_STRONG' else g>0) else 'NONCONFORM'
                        ledger.append(dict(condition=sid,path=path,model=arm,query=('HEAD','BODY')[q],frame=i,seq=int(d['seq'][i]),ready=bool(r['ready'][i]),grade=g,state=state))
        r=scores['v2'];bad=np.zeros(len(r['ready']),bool)
        for q in range(2):
            exp=C.expected(sid,q);g=r['S_grade'][:,q]
            if exp!='UNKNOWN':bad|=r['ready']&(~(g==0 if exp=='SILENT' else g<2 if exp=='NO_STRONG' else g>0))
        i=int(np.flatnonzero(bad)[0] if bad.any() else np.flatnonzero(r['ready'])[0])
        zs=C.inputs(d);valid=C.good(d)[i];maxzone=int(np.argmax(zs['v2'][i].max(-1)));maxbin=int(zs['v2'][i,maxzone].argmax())
        example=dict(condition=sid,frame=i,seq=int(d['seq'][i]),selection='first v2 S nonconform' if bad.any() else 'first ready',
                     max_z_zone=maxzone,max_z_bin=maxbin,firmware_distance_mm=d['D'][i].tolist(),firmware_valid=valid.tolist(),
                     v1_grade=scores['v1']['S_grade'][i].tolist(),v2_grade=r['S_grade'][i].tolist())
        examples.append(example)
        fig,ax=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        dist=np.where(valid,d['D'][i]/1000,np.nan).reshape(8,8)
        im=ax[0,0].imshow(dist,vmin=0,vmax=3,cmap='viridis');fig.colorbar(im,ax=ax[0,0],label='weak firmware range (m)')
        ax[0,0].set_title('Raw zone layout; white = invalid')
        for rr,cc in zip(*np.where(np.isfinite(dist))):ax[0,0].text(cc,rr,f'{dist[rr,cc]*1000:.0f}mm' if dist[rr,cc]<.15 else f'{dist[rr,cc]:.2f}',ha='center',va='center',fontsize=7,color='white')
        im=ax[0,1].imshow(d['H'][i],aspect='auto',cmap='magma');fig.colorbar(im,ax=ax[0,1],label='decoded H')
        ax[0,1].set(title='Raw CNH',xlabel='aggregate bin',ylabel='raw zone')
        im=ax[0,2].imshow(zs['v2'][i],aspect='auto',cmap='coolwarm');fig.colorbar(im,ax=ax[0,2],label='signed Z (no clipping)')
        ax[0,2].set(title='Adapter v2',xlabel='aggregate bin',ylabel='raw zone')
        bins=np.arange(16)*C.WIDTH
        for z,ls in [(maxzone,'-'),(34,'--')]:ax[1,0].plot(bins,d['H'][i,z],ls,label=f'zone {z}')
        ax[1,0].set(title='Measured profiles',xlabel='aggregate-bin lower edge (m)',ylabel='decoded H');ax[1,0].legend()
        for path in ('v1','v2'):ax[1,1].plot(bins,zs[path][i,maxzone],label=f'{path}, zone {maxzone}')
        ax[1,1].set(title=f'Input comparison: max v2 Z bin {maxbin}',xlabel='aggregate-bin lower edge (m)',ylabel='Z');ax[1,1].legend()
        for path,color in [('v1','C0'),('v2','C1')]:
            for q,ls in [(0,'-'),(1,'--')]:
                ax[1,2].plot(np.arange(len(r['ready'])),np.maximum(scores[path]['m3_score'][:,q]/.9404184587540165,scores[path]['local_score'][:,q]/4.625390338985158),color=color,ls=ls,label=f'{path} '+('HEAD','BODY')[q])
        ax[1,2].axhline(1,color='black',lw=1);ax[1,2].axvline(i,color='grey',lw=1)
        ax[1,2].set(title='Original5 strong ratio (max of 2 paths)',xlabel='frame',ylabel='score / frozen reference');ax[1,2].legend(fontsize=8)
        fig.suptitle(f"Condition {sid}: {NAMES[sid]} | frame {i}, seq {int(d['seq'][i])} | {example['selection']}\nS grade v1 {example['v1_grade']} / v2 {example['v2_grade']} | nominal -10deg, identity transport")
        fig.savefig(figdir/f'condition_{sid}.png',dpi=130);plt.close(fig)
    for name,rows in [('segment_alerts.csv',summary),('frame_alerts.csv',ledger)]:
        with (C.OUT/name).open('w',newline='',encoding='utf-8-sig') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    C.save('typical_frames.json',examples)
    npz=dict(np.load(C.OUT/'instrument_arrays.npz'));x=npz['noise_x'];y=npz['noise_y'];pred=x@np.array(p['noise']['adapter_coefficients_a_b_c'])
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for sid,color in [('4','C0'),('4b','C1')]:
        mask=npz['noise_sid']==sid;axes[0].scatter(x[mask,0],y[mask],s=2,alpha=.12,color=color,label=sid)
    axes[0].set(xscale='symlog',yscale='symlog',xlabel='positive decoded mean',ylabel='observed variance',title='Stable-window mean / variance');axes[0].legend()
    axes[1].scatter(pred,y,s=2,alpha=.10);lim=max(y.max(),pred.max());axes[1].plot([0,lim],[0,lim],'k--');axes[1].set(xscale='symlog',yscale='symlog',xlabel='predicted variance',ylabel='observed variance',title=f"NNLS R2={p['noise']['r_squared']:.3f}")
    for sid in ('2','3'):
        items=p['plane'][sid]['items'];profiles=np.array([v['profile'] for v in items]);axes[2].plot(range(-2,3),np.median(profiles,0),label=f'condition {sid} observed')
    axes[2].plot(range(-2,3),p['amplitude']['pulse_kernel'],'k--',label='fitted kernel (2)');axes[2].set(xlabel='offset from firmware bin',ylabel='positive local mass fraction',title='Near-fit / far-check pulse shape');axes[2].legend()
    fig.savefig(figdir/'instrument_fit.png',dpi=140);plt.close(fig)
    delta=[]
    for path in ('v1','v2'):
        count=0;den=0
        for sid in ('4','4b','9'):
            rs=report['segments'][sid][path];den+=rs['ready']*2;count+=sum(v['strong']+v['light'] for v in rs['arms']['Sensemble'])
        delta.append(dict(path=path,notifications=count,query_frames=den,rate=count/den))
    persistence={}
    for b in range(2):
        residuals=[p['fixed_pattern'][s]['row4_col2_bins12']['spatial_residual'][b] for s in ds]
        persistence[str(b+1)]=dict(positive_conditions=sum(v>0 for v in residuals),conditions=len(residuals),above_3mad_conditions=sum(p['fixed_pattern'][s]['row4_col2_bins12']['above_3_spatial_MAD'][b] for s in ds))
    totals={}
    for path in ('v1','v2'):
        totals[path]={}
        for arm in ('m3','old5','Sensemble'):
            rows=[r for r in summary if r['path']==path and r['model']==arm]
            totals[path][arm]={k:sum(r[k] for r in rows) for k in ('strong','light','conform','nonconform','unknown')}
    C.save('description_summary.json',dict(empty_query_frame_notifications=delta,fixed_zone34=persistence,model_totals=totals,
         verdict='仍不可迁移' if all(v['rate']==1 for v in delta) else 'See per-condition retention and prewritten descriptive rule',
         figure_selection=examples))
    pieces=dict(alert_table='\n'.join(table),conformity_table='\n'.join(conform),stability_table='\n'.join(stable),fixed_table='\n'.join(fixed))
    C.save('report_tables.json',pieces)
    print(json.dumps(dict(empty=delta,fixed=persistence,**pieces),ensure_ascii=False),flush=True)


if __name__=='__main__':
    import sys
    if '--fill-report' in sys.argv:
        dest=Path(__file__).resolve().parents[1]/'REAL_PILOT_CALIBRATION_REPLAY_DEV_20261011.md'
        text=dest.read_text(encoding='utf8');pieces=C.read('report_tables.json')
        for token,key in [('STABILITY','stability_table'),('FIXED','fixed_table'),('ALERT','alert_table'),('CONFORMITY','conformity_table')]:
            text=text.replace('@@'+token+'@@',pieces[key])
        text=text.replace('@@BUDGET@@','资源账见[budget receipt](../../../artifacts.local/work/real-pilot-calibration-replay-dev-20261011/budget_receipt.json)。累计CPU保守计入300/1500command-wall s，GPU保守计入65/300command-wall s（strict30、relaxed20、独立fresh15，含初始化）；实测脚本内部strict23.25s、relaxed11.95s，独立GPU命令10.004s。CPU计入全部读写/绘图/核验与CPU spot失败，并预留交付命令，未扩额。原始Python缺NumPy的启动失败、strict不可评、核验精度修正和CPU跨后端失败均保留。')
        dest.write_text(text,encoding='utf8')
    else:make()
