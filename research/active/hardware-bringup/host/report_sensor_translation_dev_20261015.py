"""Export the already-run pilot's translated data and a compact scientific figure.

No training. Illustration draw seed differs from the scored draws and is named.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sensor_translation_dev_20261015 as T


def main(out):
    r=T.read(out/'result.json')
    with np.load(out/'nuisance.npz',allow_pickle=False) as f:
        nuisance=dict(f)
    rows,sources=T.geometry()
    arrays={k:[] for k in (*T.ARMS,*(a+'_ambient' for a in T.ARMS),
            'modal_range_m','coverage','near_support','near_label_valid','weak_near')}
    metadata=[]; rng=np.random.default_rng(T.SEED+2)
    selected=None
    for row in rows:
        metadata.append({k:row[k] for k in ('source_id','family','role','scale')})
        rendered={}
        for arm in T.ARMS:
            h,a=T.render(row,arm,nuisance,rng,check_noise=row['role']=='check')
            arrays[arm].append(h);arrays[arm+'_ambient'].append(a);rendered[arm]=h
        arrays['modal_range_m'].append(row['y'])
        arrays['coverage'].append(row['coverage'])
        arrays['near_support'].append(row['near_target'])
        arrays['near_label_valid'].append(row['task_valid'])
        arrays['weak_near'].append(row['weak_near'])
        if selected is None and row['role']=='check' and row['weak_near'].any():
            selected=(row,int(np.flatnonzero(row['weak_near'])[0]),rendered)
    np.savez_compressed(out/'translated_public.npz',**{k:np.array(v) for k,v in arrays.items()})
    cameras={}
    for source in sources:
        with np.load(T.PUBLIC/(source['id']+'.npz'),allow_pickle=False) as f:
            cameras[source['id']]=dict(M_cam_from_uv=f['M_cam_from_uv'].tolist(),
                 projection='normalized UV to graphics-camera ray; not pixel K',
                 rgb_path=(T.DATA/source['rgb']).relative_to(T.ROOT).as_posix(),
                 radial_depth_path=(T.DATA/source['depth']).relative_to(T.ROOT).as_posix())
    T.save(out/'translated_manifest.json',dict(rows=metadata,sources=sources,cameras=cameras,seed=T.SEED+2,
        sensor=dict(rows=8,cols=8,bins=16,start_bin=0,sub_sample=8,
                    bin_edges_m=(np.arange(17)*T.WIDTH).tolist(),
                    zone_order='model: top to bottom, left to right',angular_grid='45 degree equal-angle proxy'),
        units=dict(geometry='relative diffuse inverse-square mass times 2000',
                   assumed='uncalibrated pseudo-count electronics',empirical='effective decoded-unit nuisance proxy'),
        actual_source_frames=8,geometry_instances=len(rows),
        illustration_not_scored_draws=True,
        layout='[48 geometry-scale instances,64 model-order zones,16 physical range bins]; source RGB paths remain in sources',
        labels='Modal inverse-square diffuse reference / angular near support, not human collision labels',
        unknown='near_label_valid=false must not become FREE; absent modal range is NaN',
        hypothesis='Empirical nuisance is a decoded-unit proxy, not calibrated ST photon simulation'))
    colors={'geometry':'#5b8db8','assumed':'#b68042','empirical':'#31815c','raw_peak':'#696969'}
    names={'geometry':'Ideal geometry','assumed':'Assumed electronics','empirical':'Empirical nuisance','raw_peak':'Raw peak'}
    fig,axs=plt.subplots(1,3,figsize=(17,5.2),layout='constrained')
    if selected:
        row,z,rendered=selected
        centers=(np.arange(16)+.5)*T.WIDTH
        for arm in T.ARMS:
            h=rendered[arm][z].copy();h[0]=np.nan
            norm=np.nanmax(abs(h))
            axs[0].plot(centers,h/max(norm,1e-8),marker='.',label=names[arm],color=colors[arm])
        axs[0].axvline(2,color='#333333',linestyle='--',linewidth=1)
        axs[0].set(title='One near/far mixture (geometry proxy)',xlabel='Nominal radial range (m)',
                   ylabel='Signed response / per-arm max abs (bin0 omitted)',ylim=(-1.1,1.1))
        axs[0].text(.03,.03,f"{row['family']}, scale {row['scale']}, zone {z}",transform=axs[0].transAxes,fontsize=8)
        axs[0].legend(fontsize=8)
    tasks=r['public_near_task']['empirical']; arms=(*T.ARMS,'raw_peak')
    x=np.arange(len(arms))
    recall=[100*tasks[a]['hits']/tasks[a]['positive'] for a in arms]
    false=[100*tasks[a]['false_alarms']/tasks[a]['negative'] for a in arms]
    axs[1].bar(x-.17,recall,.34,label='Near-support recall',color='#31815c')
    axs[1].bar(x+.17,false,.34,label='False near-support',color='#c27b62')
    axs[1].set_xticks(x,[names[a].replace(' ','\n') for a in arms],fontsize=8)
    axs[1].set(title='Public check, same empirical input domain',ylabel='Percent (%)',ylim=(0,108))
    axs[1].legend(fontsize=8)
    for i,a in enumerate(arms):
        axs[1].text(i-.17,recall[i]+1,f"{tasks[a]['hits']}/176",ha='center',fontsize=7)
        axs[1].text(i+.17,false[i]+1,f"{tasks[a]['false_alarms']}/1008",ha='center',fontsize=7)
    for j,(group,key) in enumerate((('Dorm, 6 segments','pilot_check_macro'),('Phone, 18 segments','phone_macro'))):
        for i,a in enumerate(arms):
            val=r[key][a]['mae_m']
            xp=j+(i-1.5)*.18
            axs[2].bar(xp,val,.17,label=names[a] if j==0 else None,color=colors[a])
            axs[2].text(xp,val+.025,f'{val:.2f}',ha='center',fontsize=8)
    axs[2].set_xticks([0,1],['Dorm: 6 segments','Phone: 18 segments'])
    axs[2].set(title='Real recordings: firmware agreement only',ylabel='Segment-macro absolute discrepancy (m)',ylim=(0,2.25))
    axs[2].legend(fontsize=8)
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Sensor translation pilot: useful synthetic bridge, real transfer still incomplete',fontsize=14)
    fig.supxlabel('Public check: 2 source families, 296 zone-scale cases x 4 nuisance draws; weak-near: 8 cases x 4, including range-boundary cases.\n'
                  'Real metrics condition on valid firmware returns; no independent distance or collision ground truth.',fontsize=9)
    fig.savefig(out/'overview.png',dpi=160)
    plt.close(fig)
    T.save(out/'export_receipt.json',dict(source_sha256=T.sha(__file__),
          source_result_sha256=T.sha(out/'result.json'),translated_sha256=T.sha(out/'translated_public.npz'),
          figure_sha256=T.sha(out/'overview.png'),instances=len(rows),source_frames=8))
    print(json.dumps(dict(instances=len(rows),figure=str(out/'overview.png'))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=T.OUT/'run-c')
    main(p.parse_args().output)
