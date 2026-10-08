"""Cached finite-ray visibility and horizontal-length description; no inference."""
from pathlib import Path
import csv
import hashlib
import json
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/'artifacts.local/work/cnh-aligned-shapes-dev-20261008'
OUT=ROOT/'artifacts.local/work/cnh-bar-cached-diagnostic-dev-20261008/visibility'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf8')


def summary(x):
    x=np.asarray(x,float)
    return dict(n=int(x.size),min=float(x.min()),median=float(np.median(x)),max=float(x.max())) if x.size else None


def run():
    started=time.monotonic();cpu=time.process_time()
    if (OUT/'result.json').exists():raise FileExistsError('Complete output preserved')
    OUT.mkdir(parents=True,exist_ok=True)
    p=json.loads((SRC/'PLAN.json').read_text());rows=p['scene_rows']
    inputs={str((SRC/n).relative_to(ROOT)):sha(SRC/n) for n in ('PLAN.json','evaluated.npz','physical.npz','geometry.npz')}
    with np.load(SRC/'evaluated.npz') as z:
        cat=z['category'];rays=z['first_target_rays'];timely=z['timely']
    with np.load(SRC/'geometry.npz') as z:sensor=z['sensor']
    with np.load(SRC/'physical.npz') as z:
        mu=z['expectation'];signal=z['target_signal'];ambient=z['ambient']
    assert len(rows)==len(cat)==len(mu)==len(rays)
    assert rays.shape==(len(rows),16)
    assert np.allclose(sensor[:,2,3],np.arange(16)*.16-2.4)
    plan=dict(lane='EXPLORE description of consumed finite shape cache',wall_budget_seconds=30,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},input_sha256=inputs,
        all_scene_denominator=len(rows),body_contact_scene_denominator=int((cat[:,1]=='contact').sum()),
        visibility='Saved first_target_rays: target is first hit, 16x16 rays per each of64 zones; includes background occlusion; no ray recast.',
        vertical_condition='World X right/Y down/Z forward, pitch=-10deg, half-FOV22.5deg. Lowest FOV elevation32.5deg. For box lo_y>0, vertical feasibility requires rear_distance >= lo_y/tan(32.5deg); front-distance exit = lo_y/tan(32.5deg) - depth_extent. Necessary condition only; finite rays, horizontal FOV and occlusion may exit earlier. lo_y<=0 has no lower vertical cutoff in this argument.',
        time='All raw f0..15 retained; front=3.05-.16*f, deadline f13=.97m; f14=.81 and f15=.65 late.',
        pairing='Horizontal30/90cm matched placement, side, y bounds, z bounds, rho, context and background. For side placements inner x fixed, whole-box center shifts30cm; center placements whole-box center fixed0 and inner edge changes. Different length changes spatial distribution, not isolated shape/length causal effect.',
        delta='mu_long-mu_short signed; preserves occlusion/background changes. Target_signal is endpoint-rho target contribution, not target-present minus target-absent expectation. Pair d2=sum(delta^2/(.5*(mu_long+mu_short)+16ambient)) is privileged pair separation proxy, not missing-background evidence or detector performance.',
        exclusions='No model inference, rendering, ray recast, absent-background estimate, threshold or causal conclusion. Full492 scene denominator retained; no fresh confirmation.')
    write('PLAN.json',plan);(OUT/'source.py').write_bytes(Path(__file__).read_bytes())
    body=[]
    tangent=np.tan(np.deg2rad(32.5))
    for i,r in enumerate(rows):
        if cat[i,1]!='contact':continue
        depth=r['hi'][2]-r['lo'][2];upper=r['lo'][1]
        cutoff=upper/tangent-depth if upper>0 else None
        front=r['lo'][2]-sensor[:,2,3];rear=r['hi'][2]-sensor[:,2,3]
        visible=rays[i]>0;frames=np.flatnonzero(visible)
        body.append(dict(scene_id=i,family=r['family'],variant=r['variant'],group=r['group'],placement=r['placement'],side=r['side'],rho=r['rho'],context=r['context'],
            upper_y=upper,depth_extent=depth,vertical_rear_threshold_m=upper/tangent if upper>0 else None,vertical_front_exit_m=cutoff,
            theoretical_front_exit_before_deadline=(cutoff>.97) if cutoff is not None else False,
            last_visible_frame=int(frames[-1]) if len(frames) else -1,last_visible_front_m=float(front[frames[-1]]) if len(frames) else None,
            last_visible_vs_deadline='NEVER' if not len(frames) else 'BEFORE' if frames[-1]<13 else 'AT' if frames[-1]==13 else 'AFTER',
            frame13_seen=bool(visible[13]),frame14_seen=bool(visible[14]),frame15_seen=bool(visible[15]),
            frame13_front_m=float(front[13]),frame13_rear_m=float(rear[13]),
            M3_BODY_timely_replicas=int(timely[i,:,1].sum())))
    groups=[]
    for key in sorted(set((r['family'],r['variant']) for r in body)):
        rs=[r for r in body if (r['family'],r['variant'])==key]
        groups.append(dict(family=key[0],variant=key[1],body_contact_scenes=len(rs),
            theoretical_exit_before_deadline=sum(r['theoretical_front_exit_before_deadline'] for r in rs),
            last_visible_vs_deadline={s:sum(r['last_visible_vs_deadline']==s for r in rs) for s in ('NEVER','BEFORE','AT','AFTER')},
            **{f'frame{f}':dict(seen=sum(r[f'frame{f}_seen'] for r in rs),unseen=sum(not r[f'frame{f}_seen'] for r in rs)) for f in (13,14,15)}))
    pairmap={}
    for i,r in enumerate(rows):
        if r['family']!='horizontal':continue
        key=json.dumps([r[k] for k in ('placement','side','rho','context','background')]+[r['lo'][1:],r['hi'][1:]],sort_keys=True)
        length=round(r['hi'][0]-r['lo'][0],2)
        pairmap.setdefault(key,{})[length]=i
    pairs=[];deltas=[]
    for variants in pairmap.values():
        assert set(variants)=={.30,.90},variants
        short,long=variants[.30],variants[.90];a=rows[short];b=rows[long]
        delta=mu[long]-mu[short];var=.5*(mu[long]+mu[short])+16*ambient[...,None]
        d2=np.divide(delta**2,var,out=np.zeros_like(delta),where=var>0).sum((-3,-2,-1))
        positive=np.maximum(delta,0).sum((-3,-2,-1));negative=np.minimum(delta,0).sum((-3,-2,-1))
        zones=np.abs(delta).sum(-1)
        pairs.append(dict(short_id=short,long_id=long,group=a['group'],placement=a['placement'],side=a['side'],rho=a['rho'],variant_short=a['variant'],variant_long=b['variant'],
            BODY_category_short=cat[short,1],BODY_category_long=cat[long,1],
            center_x_short=(a['lo'][0]+a['hi'][0])/2,center_x_long=(b['lo'][0]+b['hi'][0])/2,
            target_signal_short_predeadline=float(signal[short,:14].sum()),target_signal_long_predeadline=float(signal[long,:14].sum()),
            target_signal_delta_predeadline=float((signal[long,:14]-signal[short,:14]).sum()),
            signed_mu_delta_predeadline=float(delta[:14].sum()),positive_mu_delta_predeadline=float(positive[:14].sum()),negative_mu_delta_predeadline=float(negative[:14].sum()),
            changed_zone_exposures_predeadline=int((zones[:14]>0).sum()),changed_range_bins_predeadline=int((delta[:14]!=0).sum()),
            d_pair_past8_at_deadline=float(np.sqrt(d2[6:14].sum())),
            short_rays_predeadline=int(rays[short,:14].sum()),long_rays_predeadline=int(rays[long,:14].sum())))
        deltas.append(delta)
    pairgroups=[]
    for key in sorted(set((r['group'],r['rho'],r['variant_short']) for r in pairs)):
        rs=[r for r in pairs if (r['group'],r['rho'],r['variant_short'])==key]
        pairgroups.append(dict(group=key[0],rho=key[1],short_variant=key[2],pairs=len(rs),
            target_signal_delta=summary([r['target_signal_delta_predeadline'] for r in rs]),
            signed_mu_delta=summary([r['signed_mu_delta_predeadline'] for r in rs]),
            d_pair=summary([r['d_pair_past8_at_deadline'] for r in rs]),
            positive_target_contribution_pairs=sum(r['target_signal_delta_predeadline']>0 for r in rs),
            negative_mu_pairs=sum(r['negative_mu_delta_predeadline']<0 for r in rs)))
    for name,records in [('body_contacts.csv',body),('length_pairs.csv',pairs)]:
        with (OUT/name).open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    np.savez_compressed(OUT/'pair_deltas.npz',pair_ids=np.array([[r['short_id'],r['long_id']] for r in pairs]),signed_mu_delta=np.stack(deltas))
    elapsed=time.monotonic()-started
    if elapsed>=30:raise TimeoutError('30 second wall budget reached')
    result=dict(status='COMPLETE',all_scenes=len(rows),body_contact_scenes=len(body),body_contact_replica_events=len(body)*timely.shape[1],
        visibility_groups=groups,horizontal_pairs=len(pairs),pair_groups=pairgroups,
        wall_seconds=elapsed,cpu_seconds=time.process_time()-cpu,interpretation=plan['exclusions'])
    write('result.json',result);print((OUT/'result.json').read_text())


if __name__=='__main__':run()
