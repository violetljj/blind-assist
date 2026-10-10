"""Expectation-only HEAD bar signal diagnostic on consumed simulated fixtures.

No sampling, fit, model forward, threshold changes or protected input. rho=0
is an opaque foreground, not an absent target. All oracle metrics use authored
target geometry and are descriptive Gaussian noise-standardized proxies.
"""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_data_dev as D

ROOT = C.ROOT
DEFAULT_OUT = ROOT/'artifacts.local/work/cnh-head-thin-signal-oracle-dev-20261010'


def save_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve existing evidence: '+str(path))
    C.save(path, value)


def plan_read(out):
    p = C.read(out/'PLAN.json')
    if not out.resolve().is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Canonical artifact routing required')
    for path, expected in p.get('source_sha256', {}).items():
        source = Path(path)
        if not source.is_absolute():
            source = ROOT/source
        if C.sha(source) != expected:
            raise ValueError('Frozen source changed: '+str(source))
    return p


def make_geometries(rows, thickness):
    selected = [r for r in rows if r['shape_family']=='horizontal' and
                r['group']==0 and r['placement']=='contact' and r['size_variant']==0]
    if len(selected)!=16:
        raise ValueError('Expected all 16 consumed thin HEAD contact worlds')
    result = []
    for row in selected:
        t = row['target_box']
        yc = (t['lo'][1]+t['hi'][1])/2
        for thick in thickness:
            box = dict(lo=[t['lo'][0], yc-thick/2, t['lo'][2]],
                       hi=[t['hi'][0], yc+thick/2, t['lo'][2]+thick])
            result.append(dict(geometry_id=len(result), source_scene_id=row['scene_id'],
                source_scene_uid=row['scene_uid'], background_id=row['background_id'],
                background_family=row['background_family'], source_rho=row['rho'],
                side=row['side'], thickness_m=thick, target_box=box,
                background_boxes=row['background_boxes']))
    return result


def weighted_ray_occupancy(renderer, target):
    """Before pulse/leak, visible quadrature mass by native zone; privileged."""
    cp = renderer.cp
    directions = cp.asnumpy(renderer.directions).reshape(renderer.npose,renderer.nray,3)
    background_distance = cp.asnumpy(renderer.distance).reshape(renderer.npose,renderer.nray)
    poses = cp.asnumpy(renderer.poses)
    origin = poses[:,:3,3][:,None,:]
    lo, hi = np.asarray(target['lo']), np.asarray(target['hi'])
    parallel = np.abs(directions)<1e-14
    safe = np.where(parallel,1.,directions)
    a,b = (lo-origin)/safe,(hi-origin)/safe
    near = np.where(parallel,-np.inf,np.minimum(a,b))
    far = np.where(parallel,np.inf,np.maximum(a,b))
    enter,leave = near.max(-1),far.min(-1)
    distance = np.where(enter>1e-10,enter,leave)
    outside = (parallel & ((origin<lo)|(origin>hi))).any(-1)
    take = (~outside & (leave>=np.maximum(enter,0)) & (distance>1e-10) &
            np.isfinite(distance) & (distance<=background_distance))
    weight = cp.asnumpy(renderer.fractions*renderer.weights).reshape(renderer.nray)
    occupancy=(take*weight[None]).reshape(renderer.npose,8,8,renderer.sub**2).sum(-1)
    import cnh_location_reference_gpu as G
    rawbin=np.floor((np.where(take,distance,0)-renderer.params.range_zero_m)/G.SENSOR.RAW_BIN_M).astype(int)
    visible=take & (rawbin>=0) & (rawbin<128)
    zone=np.broadcast_to(np.arange(renderer.nray)[None]//(renderer.sub**2),visible.shape)
    frame=np.broadcast_to(np.arange(renderer.npose)[:,None],visible.shape)
    direct=np.zeros((renderer.npose,1024),bool)
    cells=frame*1024+zone*16+rawbin//8
    direct.reshape(-1)[cells[visible]]=True
    min_range=np.min(np.where(visible,distance,np.inf),axis=-1)
    return occupancy,direct,visible.sum(-1).astype(np.int32),min_range


def run(out):
    p = plan_read(out)
    began = time.monotonic()
    previous = sum(C.read(x).get('seconds',0) for x in out.glob('run_failure_*.json'))
    cap = p['budgets_wall_seconds']['scientific']
    def check():
        if previous+time.monotonic()-began>=cap:
            raise TimeoutError('Cumulative scientific command wall cap reached')
    old = Path(p['source_evaluation'])
    rows_file = old/'scene_rows.json'
    geometry_file = old/'data/hold/geometry.npz'
    rows = C.read(rows_file)['hold']
    geom = make_geometries(rows,p['thickness_m'])
    with np.load(geometry_file) as z:
        sensor = z['sensor'].copy();query=z['public_query'].copy()
    chain_rows=[r for r in rows if r['shape_family']=='horizontal' and r['group']==0 and r['placement']=='contact']
    if len(chain_rows)!=32:raise ValueError('Expected original thin16 plus thick16 HEAD horizontal contacts')
    dll_handle = None
    engine = None
    env_before={key:os.environ.get(key) for key in ('PATH','CUPY_CACHE_DIR','TEMP','TMP')}
    phase = 'imports'
    try:
        for key in ('CUPY_CACHE_DIR','TEMP','TMP'):
            directory=out/'runtime'/key.lower();directory.mkdir(parents=True,exist_ok=True)
            os.environ[key]=str(directory)
        import torch
        dll = Path(torch.__file__).parent.parent/'nvidia/cublas/bin'
        if os.name=='nt' and dll.is_dir():
            dll_handle = os.add_dll_directory(str(dll))
            os.environ['PATH']=str(dll)+os.pathsep+os.environ.get('PATH','')
        _,R,G,_=D.frozen_imports()
        endpoints=np.empty((len(geom),2,16,8,8,16),np.float64)
        occupancy=np.empty((len(geom),16,8,8),np.float64)
        direct_sweep=np.empty((len(geom),16,1024),bool)
        min_visible_range=np.empty((len(geom),16),np.float64)
        chain_target=np.empty((len(chain_rows),16,1024),np.float64)
        chain_present=np.empty_like(chain_target);chain_absent=np.empty_like(chain_target)
        chain_direct=np.empty_like(chain_target,dtype=bool)
        chain_visible=np.empty((len(chain_rows),16),np.int32)
        chain_occupancy=np.empty((len(chain_rows),16,8,8),np.float64)
        background=np.empty((4,16,8,8,16),np.float64)
        bgids=sorted({r['background_id'] for r in geom})
        bgindex={bg:i for i,bg in enumerate(bgids)}
        groups=defaultdict(list)
        for g in geom:groups[g['background_id']].append(g)
        parity=[];metadata=[]
        for bgid,values in sorted(groups.items()):
            phase=f'background{bgid}';check()
            engine=G.ExpectedRenderer(sensor,values[0]['background_boxes'])
            try:
                background[bgindex[bgid]]=engine.background_expected(return_device=False)
                ambient=engine.ambient.copy()
                metadata.append(dict(background_id=bgid,**engine.metadata))
                for begin,e in engine.iter_render([g['target_box'] for g in values],
                        candidate_batch=4,pose_batch=16,deadline_check=check):
                    for j in range(len(e)):
                        row=values[begin+j];i=row['geometry_id'];endpoints[i]=e[j]
                        occupancy[i],direct_sweep[i],_,min_visible_range[i]=weighted_ray_occupancy(engine,row['target_box'])
                originals=[(i,r) for i,r in enumerate(chain_rows) if r['background_id']==bgid]
                for begin,e in engine.iter_render([r['target_box'] for _,r in originals],
                        candidate_batch=4,pose_batch=16,deadline_check=check):
                    for j in range(len(e)):
                        ci,original=originals[begin+j];rho=original['rho']
                        t=rho/G.ENDPOINT_RHO*(e[j,1]-e[j,0]);present=e[j,0]+t
                        chain_target[ci]=t.reshape(16,1024);chain_present[ci]=present.reshape(16,1024)
                        chain_absent[ci]=background[bgindex[bgid]].reshape(16,1024)
                        chain_occupancy[ci],chain_direct[ci],chain_visible[ci],_=weighted_ray_occupancy(engine,original['target_box'])
                # CPU parity at both thickness extremes, actual original rho,
                # original frame endpoints; geometry probes are prespecified.
                for row in (values[0],values[len(p['thickness_m'])-1]):
                    check();i=row['geometry_id'];rho=row['source_rho'];frames=np.array([0,13,15])
                    expected=endpoints[i,0]+rho/G.ENDPOINT_RHO*(endpoints[i,1]-endpoints[i,0])
                    ref=R.expected(dict(poses=sensor[frames],boxes=[dict(row['target_box'],rho=rho),*row['background_boxes']]))
                    error=float(np.abs(ref['expectation']-expected[frames]).max())
                    np.testing.assert_allclose(ref['expectation'],expected[frames],atol=1e-8,rtol=1e-11)
                    np.testing.assert_array_equal(ref['ambient'],ambient[frames])
                    parity.append(dict(geometry_id=i,max_abs=error))
            finally:
                engine.close();engine=None
            print('ORACLE_BACKGROUND',bgid,'seconds',round(time.monotonic()-began,2),flush=True)
        check();phase='decomposition'
        bg_for_geometry=background[np.array([bgindex[g['background_id']] for g in geom])]
        unit_target=(endpoints[:,1]-endpoints[:,0])/G.ENDPOINT_RHO
        loss=bg_for_geometry-endpoints[:,0]
        if unit_target.min()<-1e-8 or loss.min()<-1e-8:
            raise AssertionError('Target addition/background occlusion must be nonnegative')
        for rho in p['rho_values']:
            present=endpoints[:,0]+rho*unit_target
            np.testing.assert_allclose(present-bg_for_geometry,rho*unit_target-loss,atol=1e-9,rtol=1e-12)
            if present.min()<-1e-8:raise AssertionError('Negative physical mean')
        file=out/'physical.npz'
        if file.exists():raise FileExistsError('Retain partial physical payload')
        np.savez_compressed(file,endpoints=endpoints,background=background,background_ids=np.array(bgids),
            occupancy=occupancy,direct=direct_sweep,min_visible_range=min_visible_range,
            sensor=sensor,public_query=query,ambient=ambient,endpoint_rho=G.ENDPOINT_RHO)
        chain_file=out/'chain_expectations.npz'
        if chain_file.exists():raise FileExistsError('Preserve chain payload')
        np.savez_compressed(chain_file,expected_target=chain_target.reshape(32,16,8,8,16),
            expected_present=chain_present.reshape(32,16,8,8,16),
            expected_absent=chain_absent.reshape(32,16,8,8,16),
            direct=chain_direct.reshape(32,16,8,8,16),visible_counts=chain_visible,
            occupancy=chain_occupancy,scene_ids=np.array([r['scene_id'] for r in chain_rows]),
            sensor=sensor,public_query=query,ambient=ambient)
        save_new(out/'geometries.json',geom)
        save_new(out/'run_receipt.json',dict(status='COMPLETE',seconds=previous+time.monotonic()-began,
            payload_sha256=C.sha(file),geometries_sha256=C.sha(out/'geometries.json'),renderer_parity=parity,
            chain_payload_sha256=C.sha(chain_file),chain_scenes=len(chain_rows),
            input_sha256={str(rows_file):C.sha(rows_file),str(geometry_file):C.sha(geometry_file)},
            backend=metadata,geometry_count=len(geom),expectation_endpoints=2*len(geom)*16,
            decomposition='T=rho/.65*(mu_hi-mu_opaque0); L=mu_absent-mu_opaque0; delta=T-L',
            training=0,sampling=0,threshold_changes=0,model_forward=0,protected_access=0))
    except BaseException as error:
        save_new(out/f'run_failure_{time.time_ns()}.json',dict(error=repr(error),phase=phase,
            seconds=time.monotonic()-began,traceback=traceback.format_exc()))
        raise
    finally:
        if engine is not None:engine.close()
        if dll_handle is not None:dll_handle.close()
        for key,value in env_before.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def frame_metrics(target,present,absent,ambient,loss,occupancy):
    """Returns all16 frames. Var is observation variance, not sample mean zero."""
    shape=target.shape
    t=target.reshape(len(target),-1)
    var=(present+16*ambient[...,None]).reshape(t.shape)
    pairvar=(.5*(present+absent)+16*ambient[...,None]).reshape(t.shape)
    snr=t/np.sqrt(var)
    ix=np.argmax(snr,axis=-1)
    a=np.arange(len(t))
    peak=t.argmax(-1)
    result=dict(target_best_bin_snr=snr[a,ix],target_matched_snr=np.sqrt((t*t/var).sum(-1)),
        pair_separation_proxy=np.sqrt(((present-absent).reshape(t.shape)**2/pairvar).sum(-1)),
        target_counts=t.sum(-1),background_loss_counts=loss.reshape(t.shape).sum(-1),
        signed_delta_counts=(present-absent).reshape(t.shape).sum(-1),
        occupancy_max=occupancy.reshape(len(target),-1).max(-1),
        best_snr_flatbin=ix,target_peak_flatbin=peak,
        target_peak_counts=t[a,peak],background_at_target_peak=absent.reshape(t.shape)[a,peak],
        opaque_background_at_target_peak=(present-target).reshape(t.shape)[a,peak],
        background_max_counts=absent.reshape(t.shape).max(-1),
        target_peak_zone=peak//shape[-1],target_peak_bin=peak%shape[-1])
    return result


def temporal_metrics(target,present,absent,ambient,end,window):
    start=max(0,end-window+1);target=target[start:end+1].reshape(end-start+1,-1)
    present=present[start:end+1].reshape(target.shape);absent=absent[start:end+1].reshape(target.shape)
    var=present+16*np.broadcast_to(ambient[start:end+1,...,None],(len(target),8,8,16)).reshape(target.shape)
    pairvar=.5*(present+absent)+16*np.broadcast_to(ambient[start:end+1,...,None],(len(target),8,8,16)).reshape(target.shape)
    # Fixed native bin summation preserves motion smear. The aligned peak
    # statistic chooses a privileged strongest target bin separately per frame.
    aligned=np.argmax(target/np.sqrt(var),axis=-1);a=np.arange(len(target))
    peak_signal=target[a,aligned].sum();peak_var=var[a,aligned].sum()
    return dict(actual_frames=len(target),
        moving_fixed_bin_snr=float(np.max(target.sum(0)/np.sqrt(var.sum(0)))),
        privileged_moving_peak_snr=float(peak_signal/np.sqrt(peak_var)),
        privileged_moving_matched_snr=float(np.sqrt((target*target/var).sum())),
        privileged_moving_pair_proxy=float(np.sqrt(((present-absent)**2/pairvar).sum())),
        static_repeat_peak_snr=float(np.sqrt(window)*np.max(target[-1]/np.sqrt(var[-1]))),
        static_repeat_matched_snr=float(np.sqrt(window)*np.sqrt((target[-1]*target[-1]/var[-1]).sum())),
        static_repeat_pair_proxy=float(np.sqrt(window)*np.sqrt(((present[-1]-absent[-1])**2/pairvar[-1]).sum())))


def write_csv_new(path,rows):
    if not rows:raise ValueError('No metric rows')
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze(out):
    began=time.monotonic();p=plan_read(out);rr=C.read(out/'run_receipt.json')
    if C.sha(out/'physical.npz')!=rr['payload_sha256']:raise ValueError('Payload changed')
    with np.load(out/'physical.npz') as z:
        endpoints=z['endpoints'];background=z['background'];bgids=z['background_ids']
        occupancy=z['occupancy'];sensor=z['sensor'];ambient=z['ambient'];endpoint_rho=float(z['endpoint_rho'])
        min_visible_range=z['min_visible_range']
    geometries=C.read(out/'geometries.json');bgindex={int(bg):i for i,bg in enumerate(bgids)}
    single=[];temporal=[]
    for g in geometries:
        if time.monotonic()-began>=p['budgets_wall_seconds']['analysis']:raise TimeoutError('Analysis cap reached')
        i=g['geometry_id'];absent=background[bgindex[g['background_id']]]
        unit_target=(endpoints[i,1]-endpoints[i,0])/endpoint_rho
        loss=absent-endpoints[i,0]
        for rho in p['rho_values']:
            target=rho*unit_target;present=endpoints[i,0]+target
            metrics=frame_metrics(target,present,absent,ambient,loss,occupancy[i])
            base={key:g[key] for key in ('geometry_id','source_scene_id','background_id','background_family','side','thickness_m','source_rho')}
            base['rho']=rho;base['original_thin_fixture']=bool(abs(g['thickness_m']-.017)<1e-10 and abs(rho-g['source_rho'])<1e-10)
            for frame in range(16):
                center=(np.asarray(g['target_box']['lo'])+np.asarray(g['target_box']['hi']))/2
                front_center=center.copy();front_center[2]=g['target_box']['lo'][2]
                single.append(dict(**base,frame=frame,front_distance_m=g['target_box']['lo'][2]-sensor[frame,2,3],
                    center_range_m=float(np.linalg.norm(center-sensor[frame,:3,3])),
                    nearest_visible_range_m=(float(min_visible_range[i,frame]) if np.isfinite(min_visible_range[i,frame]) else None),
                    target_front_center_euclidean_m=float(np.linalg.norm(front_center-sensor[frame,:3,3])),
                    **{key:(int(values[frame]) if key in ('best_snr_flatbin','target_peak_flatbin','target_peak_zone','target_peak_bin') else float(values[frame])) for key,values in metrics.items()}))
            for end in p['anchor_frames']:
                for window in p['windows']:
                    temporal.append(dict(**base,end_frame=end,requested_frames=window,
                        front_distance_m=g['target_box']['lo'][2]-sensor[end,2,3],
                        **temporal_metrics(target,present,absent,ambient,end,window)))
    write_csv_new(out/'single_frame.csv',single);write_csv_new(out/'accumulation.csv',temporal)
    groups=[]
    for thick in p['thickness_m']:
        for rho in p['rho_values']:
            for frame in p['anchor_frames']:
                values=[r for r in single if r['thickness_m']==thick and r['rho']==rho and r['frame']==frame]
                group=dict(thickness_m=thick,rho=rho,frame=frame,front_distance_m=values[0]['front_distance_m'],worlds=len(values))
                for name in ('target_best_bin_snr','target_matched_snr','pair_separation_proxy','target_counts','background_loss_counts','occupancy_max'):
                    v=np.array([r[name] for r in values]);group[name+'_min']=float(v.min());group[name+'_median']=float(np.median(v));group[name+'_max']=float(v.max())
                groups.append(group)
    write_csv_new(out/'grouped.csv',groups)
    save_new(out/'analysis_receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,
        single_frame_rows=len(single),temporal_rows=len(temporal),grouped_rows=len(groups),
        definitions=dict(variance='signed Skellam variance mu_present+16ambient',
            target='rho/.65*(endpoint_hi-endpoint0), actual positive target photons after pulse/spatial electronics',
            absent='background_expected without target AABB',loss='mu_absent-mu_opaque_rho0',
            pair='Gaussian proxy sqrt(sum((mu_present-mu_absent)^2/(.5*(mu_present+mu_absent)+16ambient)))',
            temporal='moving fixed native bin sum; privileged strongest-bin per frame; full template matched filter; independent static repeats at current endpoint',
            threshold='SNR is descriptive; z gate and M3 probability thresholds live in different statistic spaces'),
        limits=['Simulator conditional on nominal parameters; no calibrated hardware limit',
            'Authored target/background knowledge; matched filter and pair proxy not deployable detector results',
            'Expectation decomposition must not relabel opaque rho0 as absent',
            'No false-positive accounting or global peak-search cost in these privileged SNR values',
            'Static sqrt(K) uses independent photon noise and identical geometry; moving histories change range/support',
            '16 fixture worlds include repeated source geometry/rho/background factors; Development reuse, not confirmation'],
        outputs_sha256={f.name:C.sha(f) for f in (out/'single_frame.csv',out/'accumulation.csv',out/'grouped.csv')}))
    print('ORACLE_ANALYZED',len(single),len(temporal),'seconds',round(time.monotonic()-began,2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['run','analyze'],required=True)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    (run if args.stage=='run' else analyze)(args.output)
