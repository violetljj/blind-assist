"""Consumed real-data diagnosis with cached normalized input and per-frame scores.

prepare uses CPU only. replay is explicitly selected, keeps frozen model cuts,
and can be resumed by cell. All outputs are Development descriptions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import real_phone_imu_replay as R
import real_pilot_calibration as P
from real_contract_v5 import adapter_v5

DEFAULT = P.ROOT / 'artifacts.local/work/cnh-noise-ablation-dev-20261012/diagnosis'

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')

def prepare(out):
    (out / 'inputs').mkdir(parents=True, exist_ok=True)
    arrays = np.load(P.OUT / 'instrument_arrays.npz')
    bias = arrays['bias'].reshape(64, 16)
    noise = P.read('instrument_parameters.json')['noise']
    rotation = np.asarray(R.load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    cells, stats = [], {}
    def add_input(name, d):
        z = adapter_v5(d['H'], d['A'], bias, noise, None, True, False)
        dist = np.where(np.isin(d['S'], (5, 6, 9, 10)), d['D'], np.nan).reshape(-1,8,8)[:,::-1,::-1]
        np.savez_compressed(out / 'inputs' / (name + '.npz'), z=z, seq=d['seq'], ms=d['ms'], distance=dist)
        same = (np.diff(d['seq']) == 1) & (np.diff(d['ms']) > 0) & (np.diff(d['ms']) <= 500)
        sd = np.sqrt(np.mean(np.diff(z, axis=0)[same]**2, axis=0)/2)
        def finite(a):
            return np.where(np.isfinite(a), a, np.nan).tolist()
        stats[name] = dict(frames=len(z), contiguous_pairs=int(same.sum()),
                           median_z_by_row_bin=np.median(z, axis=(0,2)).tolist(),
                           mean_z_by_row_bin=z.mean(axis=(0,2)).tolist(),
                           median_difference_sd_by_bin=np.median(sd,axis=(0,1)).tolist(),
                           median_distance_by_zone=finite(np.nanmedian(dist,axis=0)))
        return name
    for directory in sorted(R.RAW.iterdir()):
        key = f'{int(directory.name[17:19])+8:02d}{directory.name[19:23]}'
        if key not in R.LABELS:
            continue
        d = R.load(directory, rotation)
        name = add_input('phone_' + key, d)
        p, r = float(np.nanmean(d['pitch'])), float(np.nanmean(d['roll']))
        fit = R.floor_fit(d)
        if fit and (abs(fit[1]-p)>15 or not .5<=fit[0]<=2 or abs(fit[2])>20):
            fit = None
        arms = {'imu': [p,r,0.]}
        if fit:
            arms.update(imu_h=[p,r,R.SIM_FLOOR_M-fit[0]], floor_h=[fit[1],fit[2],R.SIM_FLOOR_M-fit[0]])
        for arm, pose in arms.items():
            cells.append(dict(id=name+'_'+arm, input=name, pose=pose, scene=R.LABELS[key][0], expect=R.LABELS[key][1], session=directory.name))
        if key == '100754':
            cells.append(dict(cells[-1], id=name+'_floor_h_drop_bottom3', drop_rows=[5,6,7]))
    data = P.data()
    for sid in ('4','4b','5','6','7','9'):
        name = add_input('dorm_'+sid, data[sid])
        for p in (0.,5.):
            cell = dict(id=name+'_pitch'+str(int(p)), input=name, pose=[p,0.,0.], scene='dorm '+sid,
                        expect='quiet' if sid in ('4','4b','9') else 'near-pass' if sid=='7' else 'target-present; query location uncertain')
            cells.append(cell)
            if sid == '5' and p == 5:
                cells.append(dict(cell,id=name+'_pitch5_drop_bottom3',drop_rows=[5,6,7]))
    save(out/'cells.json',cells)
    # Missing firmware ranges remain null, never substituted as clear evidence.
    save(out/'input_statistics.json',json.loads(json.dumps(stats).replace('NaN','null')))
    print(json.dumps(dict(inputs=len(stats),cells=len(cells),output=str(out))),flush=True)

def replay(out, manifest, device, tag, only, drop_input_bins=()):
    import torch
    from local_signed_bias_adapter import ContractReplay
    torch.set_num_threads(4)
    dest=out/tag;dest.mkdir(parents=True,exist_ok=True)
    meta=dict(manifest=str(Path(manifest).resolve()), manifest_sha256=P.sha(manifest),device=device,
              development_only=True, cuts_frozen=True,drop_input_bins=list(drop_input_bins), note='Previously consumed data; input masks are explanatory interventions, not deployed filters.')
    old=dest/'receipt.json'
    if old.exists() and json.loads(old.read_text())['manifest_sha256'] != meta['manifest_sha256']:
        raise ValueError('Do not resume under changed manifest')
    if old.exists() and json.loads(old.read_text()).get('drop_input_bins',[]) != list(drop_input_bins):
        raise ValueError('Do not resume under changed intervention')
    save(old,meta)
    cells=json.loads((out/'cells.json').read_text())
    for cell in cells:
        if cell['input']=='dorm_7':cell['expect']='near-pass'
    save(out/'cells.json',cells)
    records=[]
    for cell in json.loads((out/'cells.json').read_text()):
        if cell.get('drop_rows') or cell.get('diagnostic'): continue
        if only and cell['id'] not in only: continue
        source,key=cell['input'].split('_',1)
        records.append(dict(id=cell['id'],source=source,key=key,arm=cell['id'][len(cell['input'])+1:],
                            expect=cell['expect'],pose=cell['pose'],npz=str(dest/(cell['id']+'.npz'))))
    save(dest/'index.json',dict(records=records,status='running',**meta))
    engine=ContractReplay(manifest,device=device)
    try:
        for cell in json.loads((out/'cells.json').read_text()):
            if drop_input_bins and (cell.get('drop_rows') or cell.get('diagnostic')):continue
            if only and cell['id'] not in only: continue
            target=dest/(cell['id']+'.npz')
            if target.exists(): continue
            d=dict(np.load(out/'inputs'/(cell['input']+'.npz')))
            if drop_input_bins:d['z'][...,list(drop_input_bins)]=0
            if cell.get('drop_rows'):d['z'][:,cell['drop_rows']]=0
            if cell.get('drop_bins'):d['z'][...,cell['drop_bins']]=0
            if cell.get('drop_support'):
                support=np.load(out/cell['drop_support'])['floor_support']
                d['z'][:,support]=0
            start=time.monotonic()
            engine.engine.configure_geometry(*cell['pose'])
            result=engine.infer(d['z'],d['seq'],d['ms'])
            np.savez_compressed(target,**result,seq=d['seq'],ms=d['ms'])
            summary=dict(cell=cell,seconds=time.monotonic()-start,**R.summarize(result,d['distance']))
            save(dest/(cell['id']+'.json'),summary)
            print(json.dumps(summary),flush=True)
    finally:engine.close()
    save(dest/'index.json',dict(records=records,status='complete',**meta))

def geometry(out):
    import torch
    from cnh_cvr_pilot import rotation
    from cnh_cvr_projection import Projector, query_masks, SUB, WIDTH, EDGE
    torch.set_num_threads(2)
    proj=Projector('cpu');pts=proj.points.float();vol=proj.volumes.float()
    mask=torch.tensor(query_masks().reshape(2,-1)).repeat_interleave(SUB**3,dim=1)
    report={}
    cells=json.loads((out/'cells.json').read_text())
    for cell in cells:
        if cell.get('drop_rows') or cell.get('diagnostic'):continue
        p,r,h=cell['pose'];c,s=np.cos(np.deg2rad(r)),np.sin(np.deg2rad(r))
        rot=rotation(p,'x')@np.array([[c,-s,0],[s,c,0],[0,0,1]])
        t=torch.tensor(rot,dtype=torch.float32)
        xyz=(pts-torch.tensor([0.,h,0.]))@t
        radius=torch.linalg.vector_norm(xyz,dim=1)
        ij=torch.floor((xyz[:,:2]/xyz[:,2:3].clamp_min(1e-30)+EDGE)/(2*EDGE)*8).long()
        bins=torch.floor(radius/WIDTH).long()
        valid=(xyz[:,2]>0)&(ij>=0).all(1)&(ij<8).all(1)&(bins>=0)&(bins<16)
        ix=(ij[:,1].clamp(0,7)*8+ij[:,0].clamp(0,7))*16+bins.clamp(0,15)
        w=valid*proj.voxel_volume/(SUB**3)/vol[ix]
        source=np.stack([np.bincount(ix.numpy(),weights=(w*mask[q]).numpy(),minlength=1024) for q in range(2)]).reshape(2,8,8,16)
        d=np.load(out/'inputs'/(cell['input']+'.npz'));z=np.maximum(d['z'].mean(0),0);z[...,0]=0
        from cnh_sim2real_frozen import segments
        _,reset=segments(d['seq'],d['ms'])
        starts=np.maximum.accumulate(np.where(reset,np.arange(len(d['z'])),0))
        means=np.stack([d['z'][max(starts[i],i-7):i+1].mean(0) for i in range(len(d['z']))])
        support=source>0;support[...,0]=False
        scores={}
        for minbin in (1,3):
            active=support.copy();active[...,:minbin]=False
            for label,arr in (('single',d['z']),('history8',means)):
                scores[f'{label}_max_bin{minbin}plus']=np.stack([np.max(arr[:,active[q]],axis=1) if active[q].any() else np.zeros(len(arr)) for q in range(2)],axis=1)
                scores[f'{label}_count_gt2_bin{minbin}plus']=np.stack([(arr[:,active[q]]>2).sum(1) for q in range(2)],axis=1)
        np.savez_compressed(out/(cell['id']+'_evidence.npz'),**scores)
        contribution=source*z[None]
        world=R._U@rot.T
        floor_range=np.divide(1.65-h,world[...,1],out=np.full((8,8),np.nan),where=world[...,1]>0)
        floor_bin=np.floor(floor_range/WIDTH)
        narrow=np.abs(np.arange(16)[None,None,:]-floor_bin[...,None])<=1
        np.savez_compressed(out/(cell['id']+'_geometry.npz'),source_weight=source,positive_mean_contribution=contribution,floor_range=floor_range,floor_support=narrow)
        report[cell['id']]=dict(pose=cell['pose'],positive_mass=contribution.sum((1,2,3)).tolist(),
                               positive_mass_by_row=contribution.sum((2,3)).tolist(),
                               positive_mass_by_bin=contribution.sum((1,2)).tolist(),
                               floor_bin_plusminus1_mass=(contribution*narrow).sum((1,2,3)).tolist(),
                               floor_geometry_is_assumption=cell['input'].startswith('dorm'))
    save(out/'geometry.json',report)
    fresh=[]
    for name in ('phone_100754_floor_h','dorm_5_pitch5'):
        cell=next(c for c in cells if c['id']==name)
        for bins in ([1],[1,2]):
            fresh.append(dict(cell,id=name+'_dropbin'+''.join(map(str,bins)),diagnostic=True,drop_bins=bins))
        fresh.append(dict(cell,id=name+'_drop_floor_support',diagnostic=True,drop_support=name+'_geometry.npz'))
    existing={c['id'] for c in cells}
    save(out/'cells.json',cells+[c for c in fresh if c['id'] not in existing])

def summarize_diagnosis(out):
    records=json.loads((out/'joint/index.json').read_text())['records']
    summary={}
    for c in records:
        a=np.load(c['npz']);ready=a['ready'];n=int(ready.sum())
        e=np.load(out/(c['id']+'_evidence.npz'))
        row=dict(ready=n,expect=c['expect'],features={k:np.median(e[k][ready],axis=0).tolist() for k in e.files})
        masked=out/'joint_dropbin1'/(c['id']+'.npz')
        if masked.exists():
            b=np.load(masked)
            assert np.array_equal(ready,b['ready'])
            row['original_strong']=(a['S_grade'][ready]==2).sum(0).tolist()
            row['masked_strong']=(b['S_grade'][ready]==2).sum(0).tolist()
            row['original_light']=(a['S_grade'][ready]==1).sum(0).tolist()
            row['masked_light']=(b['S_grade'][ready]==1).sum(0).tolist()
        summary[c['id']]=row
    save(out/'diagnostic_findings.json',summary)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','replay','geometry','summary'])
    ap.add_argument('--out',type=Path,default=DEFAULT);ap.add_argument('--manifest')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--tag',default='joint')
    ap.add_argument('--only',nargs='*',default=[]);ap.add_argument('--drop-input-bins',nargs='*',type=int,default=[]);a=ap.parse_args()
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='geometry':geometry(a.out)
    elif a.stage=='summary':summarize_diagnosis(a.out)
    else:
        if not a.manifest:ap.error('--manifest required for replay')
        replay(a.out,a.manifest,a.device,a.tag,a.only,a.drop_input_bins)
