"""Compact reproducible tables/figure from the retained real replay artifacts."""
import json
from pathlib import Path
import numpy as np
import cnh_sim2real_input_contract as C

def build():
    r=C.read('replay_report.json');i=C.read('instrument_report.json');inventory=C.read('inventory.json')
    n=sum(s['paths']['old']['ready'] for s in r['segments'].values())
    summary=dict(frames=sum(s['frames'] for s in r['segments'].values()),paired_ready=n,warmup=sum(s['frames'] for s in r['segments'].values())-n,paths={})
    lines=['| 模型/输入 | HEAD触发 | BODY触发 |','|---|---:|---:|']
    decomposition=['| 模型/输入/查询 | 有表面支撑 | 无观测表面支撑 | 两格残差敏感 | 合计 |','|---|---:|---:|---:|---:|']
    for path in ('old','new'):
        summary['paths'][path]={}
        for arm in ('m3','old5','Sensemble'):
            counts=np.sum([s['paths'][path]['arms'][arm]['trigger_counts'] for s in r['segments'].values()],0)
            classes={k:np.sum([s['paths'][path]['classes'][arm][k] for s in r['segments'].values()],0).tolist() for k in ('surface_supported','no_observed_support','fixed_pattern_sensitive','fixed_sensitive_with_surface_support','counterfactual_added')}
            summary['paths'][path][arm]=dict(counts=counts.tolist(),rates=(counts/n).tolist(),classes=classes)
            lines.append(f'| {arm}/{path} | {counts[0]}/{n} ({counts[0]/n:.2%}) | {counts[1]}/{n} ({counts[1]/n:.2%}) |')
            for q,label in enumerate(('HEAD','BODY')):
                vals=[classes[k][q] for k in ('surface_supported','no_observed_support','fixed_pattern_sensitive')]
                decomposition.append(f'| {arm}/{path}/{label} | {vals[0]} | {vals[1]} | {vals[2]} | {int(counts[q])} |')
    C.save('summary.json',summary)
    table=['| 片段 | decoded/满8帧 | 旧M3 H/B | 旧5格 H/B | 旧S H/B | 新三模型 H/B |','|---|---:|---:|---:|---:|---:|']
    for sid,s in r['segments'].items():
        fields=['/'.join(map(str,s['paths']['old']['arms'][a]['trigger_counts'])) for a in ('m3','old5','Sensemble')]
        table.append(f'| {sid} | {s["frames"]}/{s["paths"]["old"]["ready"]} | {fields[0]} | {fields[1]} | {fields[2]} | {s["paths"]["new"]["ready"]}/{s["paths"]["new"]["ready"]} |')
    inventory_table=['| 片段 | 帧数 | 有效距离zone-frame/总数 | 场景与弱几何 |','|---|---:|---:|---|']
    for v in inventory:
        inventory_table.append(f'| {v["id"]} | {v["frames"]} | {str(v.get("distance_valid_zone_frames","NE"))+"/"+str(v.get("zone_frame_denominator","NE"))} | {v.get("scene",v.get("status"))}; {v.get("weak_geometry","")} |')
    mapping=['| 片段 | 弱平面轴向m | 有效gain (rho=.3) | 相对L2残差 | ambient/名义4 | sigma真实/名义(聚合bin) |','|---|---:|---:|---:|---:|---:|']
    for sid,m in i['mapping'].items():
        if 'effective_gain' not in m:
            mapping.append(f'| {sid} | NOT_EVALUABLE | — | — | — | — |');continue
        pulse=i['pulse'].get(sid,{})
        sig='/'.join(f'{pulse[k]["quantiles"]["50"]:.3f}' for k in ('sigma_real','sigma_nominal')) if pulse.get('zones') else 'NE'
        mapping.append(f'| {sid} | {m["axial_m_firmware_proxy"]:.3f} | {m["effective_gain"]:.3f} | {m["relative_l2_residual"]:.3f} | {m["ambient_to_nominal_ratio"]:.3f} | {sig} |')
    C.save('report_tables.json',dict(trigger=lines,classes=decomposition,segments=table,inventory=inventory_table,mapping=mapping))
    # Static scientific output; no rendering dependencies or active sessions.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(C.OUT/'instrument_arrays.npz') as a:
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        bins=np.arange(16)
        for sid in ('h3-01','h3-04'):
            axes[0,0].plot(bins,a[f'{sid}_median'][34],marker='o',label=sid)
        axes[0,0].plot(bins,a['bias'][34],label='far-selected bias')
        axes[0,0].set(xlabel='Aggregate bin (0.3002784 m)',ylabel='Decoded CNH units',title='row4 col2: surface and persistent shifts');axes[0,0].legend()
        cov=a['bias_available'];real=np.array([np.median(a['bias'][:,b][cov[:,b]]) if cov[:,b].any() else np.nan for b in bins])
        axes[0,1].semilogy(bins[:11],real[:11],marker='o',label='real covered-zone median')
        axes[0,1].semilogy(bins[:11],a['nominal_bias'][0,:11],label='frozen electronics')
        axes[0,1].set(xlabel='Aggregate bin',ylabel='Uncalibrated amplitude',title='Residual template: shape and scale');axes[0,1].legend()
        mv=a['mean_variance_pairs'];pos=(mv[:,0]>0)&(mv[:,1]>0)
        axes[1,0].loglog(mv[pos,0],mv[pos,1],'.',markersize=2,alpha=.5)
        axes[1,0].set(xlabel='Mean decoded output (positive subset)',ylabel='Adjacent diff variance / 2',title='Stable far/off-peak mean vs variance')
        sid='h3-04';gain=i['mapping'][sid]['effective_gain'];observed=a[f'{sid}_median'][34]-a['bias'][34]
        expected=a[f'{sid}_nominal_expectation'][34]-a['nominal_bias'][34]
        axes[1,1].plot(bins[:6],observed[:6],marker='o',label='real minus far-selected bias')
        axes[1,1].plot(bins[:6],gain*expected[:6],marker='o',label='gain-fitted nominal weak plane')
        axes[1,1].set(xlabel='Aggregate bin',ylabel='Decoded units',title='04 fitted weak plane vs measured waveform');axes[1,1].legend()
        fig.savefig(C.OUT/'instrument_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,ensure_ascii=True))

if __name__=='__main__':build()
