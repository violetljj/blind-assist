"""Finite-ray geometry maps for aligned straight motion; no model inference.

X right, Y down, Z forward. Cells are finite spherical probes, not a volume
coverage certificate, physical obstacle population, or photon signal model.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import time

import numpy as np
import cnh_coverage_cue_geometry as G

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-aligned-boundary-dev-20261008/geometry'
PITCHES = (-10., -13., -16., -19.)
RADII = (.01, .025, .05)


def visibility(points, pitch, radius):
    """World/travel probe centers -> exact retained finite-ray hit mask."""
    return G.visible_spheres(np.asarray(points) @ G.rotation_yaw_pitch(0., pitch), radius)


def counts(mask, baseline, selection):
    a, b = mask[selection], baseline[selection]
    return dict(n=int(a.size), baseline=int(b.sum()), visible=int(a.sum()),
                gain=int((a & ~b).sum()), loss=int((b & ~a).sum()))


def run(out=OUT, budget_seconds=180.):
    wall, cpu = time.perf_counter(), time.process_time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)

    def check():
        if time.process_time() - cpu >= budget_seconds or time.perf_counter() - wall >= budget_seconds:
            raise TimeoutError('Geometry diagnostic budget reached')

    # Preserve repository HEAD evidence, never import modified geometry WIP.
    old = subprocess.check_output(['git', 'show', 'HEAD:research/active/dtr-r0/nearfield/cnh_track_a_geometry.py'], cwd=ROOT)
    (out / 'track_a_geometry_HEAD.py').write_bytes(old)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (Path(__file__), Path(G.__file__), G.FROZEN_RAY_SOURCE)}
    hashes['track_a_geometry_HEAD'] = hashlib.sha256(old).hexdigest()
    x = np.round(np.arange(-.30, .3001, .05), 8)
    y = np.round(np.arange(-.20, .9001, .05), 8)
    z = np.round(np.arange(.30, 3.0001, .05), 8)
    xyz = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1)
    ranges = 3.05 - .16 * np.arange(16)
    path = np.stack(np.meshgrid(x, y, ranges, indexing='ij'), -1)
    near = (ranges >= .9) & (ranges <= 1.4)
    before = ranges >= .9
    head = y < .42
    body = y >= .42
    plan = dict(lane='EXPLORE finite synthetic geometry', budget_cpu_seconds=budget_seconds,
        coordinates='X right/Y down/Z forward; sensor_to_travel=Ry(0)@Rx(pitch); local=row_world@R',
        pitches_deg=PITCHES, radii_m=RADII, x_m=x.tolist(), y_m=y.tolist(), z_m=z.tolist(),
        ranges_m=ranges.tolist(), head_y_m=[-.20,.42], body_y_m=[.42,.90],
        ray_count=16384, ray_order='8x8 zones each16x16 subrays', fov_deg=45,
        step_m=.16, sample_seconds=.2, straight_near_window_m=[.9,1.4], timely_cutoff_m=.9,
        candidate_rule='For radius=.025: maximize min(HEAD k3 fraction,BODY k3 fraction), require strict improvement over -10 baseline; tie choose closest pitch to -10. HEAD gains/losses and TOP losses separately retained. Selection is geometry-only development, no model/hardware conclusion.',
        primary_probe_denominators=dict(HEAD=int(len(x)*head.sum()),BODY=int(len(x)*body.sum())),
        exclusions=['No ground or other occluders', 'No SNR/reflectance/photon simulation',
                    'No model logits or labels', 'Sphere any-ray hit is not whole-cell volume coverage',
                    'Uniform grid counts are not real obstacle frequencies', 'No pose noise or human motion'],
        sources_sha256=hashes)
    (out/'PLAN.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf8')
    # Existing helper parity against its full 16384-ray reference, only small fixtures.
    rng = np.random.default_rng(2026100809)
    fixture = np.column_stack([rng.uniform(-.5,.5,96), rng.uniform(-.3,1.,96), rng.uniform(.3,3.,96)])
    for r in RADII:
        local = fixture @ G.rotation_yaw_pitch(0., -16.)
        np.testing.assert_array_equal(G.visible_spheres(local,r), G._full_grid(local,r))
        check()
    static = np.empty((len(PITCHES),len(RADII),len(x),len(y),len(z)), bool)
    motion = np.empty((len(PITCHES),len(RADII),len(x),len(y),16), bool)
    for ip,p in enumerate(PITCHES):
        for ir,r in enumerate(RADII):
            static[ip,ir] = visibility(xyz,p,r)
            motion[ip,ir] = visibility(path,p,r)
            check()
    n_near = motion[...,near].sum(-1)
    n_before = motion[...,before].sum(-1)
    k3 = n_near >= 3
    rows=[]
    for ip,p in enumerate(PITCHES):
        for ir,r in enumerate(RADII):
            cell=dict(pitch_deg=p,radius_m=r)
            for name,ymask in [('HEAD',head),('BODY',body),('TOP',y<=-.1),('LOW_BODY',y>=.75)]:
                cell[name+'_k3'] = counts(k3[ip,ir],k3[0,ir], np.broadcast_to(ymask[None,:],k3[ip,ir].shape))
                sel=np.broadcast_to(ymask[None,:,None] & ((z>=.9)&(z<=1.4))[None,None,:],static[ip,ir].shape)
                cell[name+'_static_near'] = counts(static[ip,ir],static[0,ir],sel)
            rows.append(cell)
    baseline=next(r for r in rows if r['radius_m']==.025 and r['pitch_deg']==-10.)
    def worst(r):
        return min(r[g+'_k3']['visible']/r[g+'_k3']['n'] for g in ('HEAD','BODY'))
    eligible = [r for r in rows if r['radius_m']==.025 and r['pitch_deg']!=-10. and worst(r)>worst(baseline)]
    def objective(r):
        return worst(r), -abs(r['pitch_deg']+10.)
    chosen=max(eligible,key=objective) if eligible else None
    np.savez_compressed(out/'maps.npz', pitches=PITCHES,radii=RADII,x=x,y=y,z=z,ranges=ranges,
                        static=static,motion=motion,near_count=n_near,predeadline_count=n_before,k3=k3)
    with (out/'straight_cells.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['pitch_deg','radius_m','x_m','y_down_m','height_band','near_count','predeadline_count','last_visible_range_predeadline_m','k3','gain_vs_minus10','loss_vs_minus10'])
        for ip,p in enumerate(PITCHES):
            for ir,r in enumerate(RADII):
                for ix,xx in enumerate(x):
                    for iy,yy in enumerate(y):
                        hit=ranges[before & motion[ip,ir,ix,iy]]
                        a,b=bool(k3[ip,ir,ix,iy]),bool(k3[0,ir,ix,iy])
                        w.writerow([p,r,xx,yy,'HEAD' if yy<.42 else 'BODY',int(n_near[ip,ir,ix,iy]),int(n_before[ip,ir,ix,iy]),float(hit[-1]) if len(hit) else '',a,a and not b,b and not a])
    witnesses=[]
    for p in PITCHES:
        for yy in (-.20,-.15,-.10,.54,.78,.90):
            pt=np.array([[0.,yy,.97]])
            witnesses.append(dict(pitch_deg=p,x_m=0.,y_down_m=yy,z_m=.97,
                center_in_continuous_fov=bool(abs(np.degrees(np.arctan2(yy,.97))+p)<=22.5),
                sphere_r025_any_finite_ray=bool(visibility(pt,p,.025)[0])))
    result=dict(status='COMPLETE',rows=rows,selected=chosen,selected_objective=objective(chosen)[0] if chosen else None,
                witnesses=witnesses,finite_ray_parity=dict(probes=96,radii=RADII,differences=0),
                static_grid_centers=int(np.prod(xyz.shape[:-1])),straight_probe_centers=int(len(x)*len(y)),
                cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall,
                interpretation='Coverage gain/loss only. A visible sphere may provide too few/weak rays for M3; changes to sensor pitch require paired rendering/model evaluation. The top loss is retained, not dropped from denominator.')
    check()
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--budget-seconds',type=float,default=180.)
    args=ap.parse_args();r=run(args.out,args.budget_seconds)
    print(json.dumps(dict(status=r['status'],selected=r['selected'],cpu_seconds=r['cpu_seconds'],wall_seconds=r['wall_seconds']),indent=2))
