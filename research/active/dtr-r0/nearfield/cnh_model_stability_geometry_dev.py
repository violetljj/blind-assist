"""New-unit fixed-model stability geometry and observation pose preparation.

CPU NumPy/SciPy only. No scores, predictor weights, torch, rendering or fitting.
"""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work'
sys.path.insert(0,str(WORK/'cnh-direction-information-dev-20261007/runtime'))
import cnh_double_height_geometry_dev as D

G, MC, S = D.G, D.G.MC, D.G.S
from cnh_track_a_readout import noisy_poses

OUT = WORK/'cnh-model-stability-dev-20261007'
UNITS = tuple(range(410004,410052))
TAGS = ('original','HB')
POOL_UNITS = tuple(range(98000,98048))+tuple(range(99000,99096))
POOL = WORK/'cnh-adaptive-query-dev-20261007/units'
ERROR_SEED = 2026100732
NOISE_SEED = 2026100607
PHOTON_SEED = 2026100606
REFLECT = np.diag([-1.,1.,1.,1.])


def save(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(D._finite_json(value),stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def rotation(degrees,axis='y'):
    c,s=np.cos(np.deg2rad(degrees)),np.sin(np.deg2rad(degrees))
    return np.array([[c,0,s],[0,1,0],[-s,0,c]]) if axis=='y' else np.array([[1,0,0],[0,c,-s],[0,s,c]])


def public_query(unit):
    """Known commanded mount schedule, not reconstructed from travel truth."""
    mode=unit%3
    yaw=20*np.sin(np.linspace(-np.pi/2,np.pi/2,16)) if mode==1 else np.full(16,15. if mode==0 else 0.)
    result=np.tile(np.eye(4),(16,1,1))
    for f,angle in enumerate(yaw):result[f,:3,:3]=rotation(angle)@rotation(-10,'x')
    return result


def mirror_pose(poses,mirrored):
    return REFLECT@np.asarray(poses)@REFLECT if mirrored else np.asarray(poses).copy()


def physical_boxes(scene,unit,tag):
    boxes=copy.deepcopy(scene['boxes'])
    if (unit//3)%2:
        for box in boxes:
            lo,hi=box['lo'][0],box['hi'][0]
            box['lo'][0],box['hi'][0]=-hi,-lo
    if tag=='HB':boxes[0]['lo'][1],boxes[0]['hi'][1]=-.1,.84
    elif tag!='original':raise ValueError('Only original/HB tags allowed')
    return boxes


def poses_for(scene,unit,config,error):
    mirrored=bool((unit//3)%2)
    sensor=np.asarray(scene['poses'],float)
    travel=np.asarray(scene['travel'],float)
    seed=int(np.random.SeedSequence([NOISE_SEED,unit,config,0]).generate_state(1)[0])
    noisy=noisy_poses(sensor,seed,dt=.2)
    sensor,travel,nn,pq=[mirror_pose(p,mirrored) for p in (sensor,travel,noisy,public_query(unit))]
    # Evaluator-only parity assertion, never the way public_query is built.
    np.testing.assert_allclose(np.linalg.inv(travel)@sensor,pq,atol=1e-12,rtol=0)
    error=np.asarray(error)
    if error.shape!=(16,) or not np.isfinite(error).all():raise ValueError('16 finite stored head errors required')
    qq=pq.copy()
    for f,angle in enumerate(error):qq[f,:3,:3]=rotation(float(angle))@pq[f,:3,:3]
    photon=int(np.random.SeedSequence([PHOTON_SEED,unit,config,0]).generate_state(1)[0])
    return dict(sensor_center=sensor,travel=travel,nn=nn,qq=qq,public_query=pq,photon_seed=photon)


def evaluate_geometry(boxes,travel):
    poses=np.asarray(travel)[G.FRAMES]
    target=boxes[0]
    overlaps=[j for j,b in enumerate(boxes[1:],1) if np.all(
        np.minimum(target['hi'],b['hi'])-np.maximum(target['lo'],b['lo'])>G.EPS)]
    meshes=[S.box_mesh(b['lo'],b['hi']) for b in boxes]
    allmesh=np.concatenate(meshes)
    ranges,ref=G.deadline_reference(G.corners(target),poses)
    frames=[[G.surface_category(G.transform(allmesh,p),q) for q in (0,1)] for p in poses]
    target_frames=[[G.surface_category(G.transform(meshes[0],p),q) for q in (0,1)] for p in poses]
    refcat=[G.surface_category(G.transform(allmesh,ref['reference_pose']),q) for q in (0,1)] if ref['covered'] else ['censored']*2
    target_ref=[G.surface_category(G.transform(meshes[0],ref['reference_pose']),q) for q in (0,1)] if ref['covered'] else ['censored']*2
    cq=[v in D.CONTACT_CATS for v in refcat]
    clear=(np.asarray(frames)=='clear').all(0).tolist()
    contact,control=any(cq),all(clear)
    reasons=[]
    if overlaps:reasons.append('target_background_overlap')
    if np.any(np.diff(ranges)>1e-10):reasons.append('nonmonotonic_target_front')
    valid=not reasons
    idx=int(np.searchsorted(G.FRAMES,ref['reference_fraction']+1e-10,side='right')-1) if ref['covered'] else None
    role='contact' if contact else 'clear' if control else 'pass' if ref['covered'] else 'censored_nonclear'
    return D._finite_json(dict(boxes=boxes,valid=valid,evaluable=bool(valid and (contact or control)),
        reasons=reasons,contact=bool(contact) if valid else None,control=bool(control) if valid else None,
        physical_contact=bool(contact),physical_control=bool(control),role=role,contact_query=cq,
        target_contact_query=[v in D.CONTACT_CATS for v in target_ref],clear_all=clear,
        frame_category=frames,target_frame_category=target_frames,ref_category=refcat,target_ref_category=target_ref,
        contact_degree=[D.CONTACT_CATS.index(v) if v in D.CONTACT_CATS else None for v in refcat],
        frame_ranges=ranges,deadline_index=idx,target_background_overlaps=overlaps,**ref))


def choose_error(pool,unit,config):
    index=int(np.random.default_rng([ERROR_SEED,unit,config]).integers(len(pool)))
    return index,pool[index].copy()


def source_files():
    paths=[Path(__file__),Path(D.__file__),Path(G.__file__),Path(MC.__file__),
           Path(MC.NR.__file__),Path(MC.SS.__file__),Path(S.__file__),
           Path(sys.modules['cnh_track_a_geometry'].__file__),Path(sys.modules['cnh_track_a_readout'].__file__)]
    return list(dict.fromkeys(paths))


def freeze_geometry_plan(outdir=OUT):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    paths=source_files()+[POOL/f'unit{u}.npz' for u in POOL_UNITS]
    plan=dict(status='frozen',units=list(UNITS),configs=list(range(40)),tags=list(TAGS),
        expected_sequences=3840,geometry_budget_seconds=600,mode_mirror_group_period=12,
        scene_generator='MC.scenes_for(unit,outside=.20); none configs0..19,corner20..39; no rejection/resampling by model score',
        hb_geometry='Only target box0 y=[-.1,.84]; all other geometry and materials fixed within paired tags',
        error_seed=ERROR_SEED,error_pool_units=list(POOL_UNITS),error_pool_key='head_c_err',
        error_sampling='One intact 16-frame trajectory from existing144x40 pool, keyed by new unit/config; no sign or scale change',
        noisy_pose_seed_prefix=NOISE_SEED,photon_seed_prefix=PHOTON_SEED,
        invalid_policy='Retain all rows, invalid contact/control null; render only valid&(contact|control). Censored full-clear remains control.',
        id_audit=dict(candidate=[410004,410051],source='Current nearfield cnh Python sources and510 related cnh plan/result/request/manifest/receipt JSON files inspected',
            candidate_matches=0,rejected_candidate=[220000,220047],rejected_training_overlap=31,
            note='Bounded authority-file audit, not an exhaustive historical binary inventory'),
        no_scores_or_models_read=True,no_render_or_training=True,
        scope='New units in the same synthetic generator, not new real heading-error participants or safety evidence',
        hashes={p.relative_to(ROOT).as_posix():D.sha(p) for p in paths})
    save(outdir/'GEOMETRY_PLAN.json',plan)
    return plan


def build_manifest(outdir=OUT,budget_seconds=600):
    outdir=Path(outdir);start=time.monotonic();plan=D.read(outdir/'GEOMETRY_PLAN.json')
    budget_seconds=min(float(budget_seconds),float(plan['geometry_budget_seconds']))
    if not np.isfinite(budget_seconds) or budget_seconds<=0:raise ValueError('Positive frozen budget required')
    if (outdir/'GEOMETRY_STARTED.json').exists():raise FileExistsError('Preserve prior geometry run')
    save(outdir/'GEOMETRY_STARTED.json',dict(unix=time.time(),budget_seconds=budget_seconds))
    def check():
        if time.monotonic()-start>budget_seconds:raise TimeoutError('600s geometry budget reached')
    try:
        if plan['units']!=list(UNITS) or plan['tags']!=list(TAGS):raise ValueError('Frozen units/tags changed')
        for path,expected in plan['hashes'].items():
            check()
            if D.sha(ROOT/path)!=expected:raise ValueError(f'Frozen source changed: {path}')
        pool=[]
        for unit in POOL_UNITS:
            with np.load(POOL/f'unit{unit}.npz',allow_pickle=False) as saved:errors=saved['head_c_err']
            if errors.shape!=(40,16) or not np.isfinite(errors).all():raise ValueError('Invalid old head-error pool')
            pool.extend(errors)
        pool=np.asarray(pool)
        rows=[];draws=[];posehashes={};(outdir/'poses').mkdir(exist_ok=True)
        for ui,unit in enumerate(UNITS):
            check();scenes=MC.scenes_for(unit);nn=[];qq=[];errors=[];su=[];sc=[];seeds=[]
            if len(scenes)!=40:raise ValueError('Expected original40 configurations')
            mirrored=bool((unit//3)%2)
            for config,scene in enumerate(scenes):
                check();index,error=choose_error(pool,unit,config)
                oldunit,oldconfig=POOL_UNITS[index//40],index%40
                obs=poses_for(scene,unit,config,error)
                nn.append(obs['nn']);qq.append(obs['qq']);errors.append(error);su.append(oldunit);sc.append(oldconfig);seeds.append(obs['photon_seed'])
                anchor_id=f'u{unit}_c{config}'
                draws.append(dict(unit=unit,config=config,pool_index=index,source_unit=oldunit,source_config=oldconfig))
                pair=[]
                for tag in TAGS:
                    physical=evaluate_geometry(physical_boxes(scene,unit,tag),obs['travel'])
                    row=dict(anchor_id=anchor_id,row_id=f'{anchor_id}_{tag}',unit=unit,config=config,tag=tag,
                        family=scene['family'],mode=unit%3,mirror=mirrored,
                        turn=('left' if mirrored else 'right') if unit%3==2 else 'none',
                        source_group=int(scene['group']),poses_path=f'poses/unit{unit}.npz',
                        pose_config_index=config,head_error_source_unit=oldunit,head_error_source_config=oldconfig,
                        photon_seed=obs['photon_seed'],**physical)
                    pair.append(row);rows.append(row)
                if pair[0]['covered']!=pair[1]['covered'] or pair[0]['censor_reason']!=pair[1]['censor_reason']:
                    raise ValueError('Height-only intervention changed reference coverage')
                np.testing.assert_allclose(pair[0]['frame_ranges'],pair[1]['frame_ranges'],atol=1e-10,rtol=0)
                if pair[0]['covered']:
                    if abs(pair[0]['reference_fraction']-pair[1]['reference_fraction'])>1e-10 or pair[0]['deadline_index']!=pair[1]['deadline_index']:
                        raise ValueError('Height-only intervention changed causal reference')
            path=outdir/f'poses/unit{unit}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream,configs=np.arange(40),sensor_center=obs['sensor_center'],travel=obs['travel'],
                    public_query=obs['public_query'],nn=np.asarray(nn),qq=np.asarray(qq),head_c_err=np.asarray(errors),
                    head_error_source_unit=np.asarray(su),head_error_source_config=np.asarray(sc),photon_seed=np.asarray(seeds,dtype=np.uint32))
            posehashes[path.relative_to(outdir).as_posix()]=D.sha(path)
            print(f'Geometry {ui+1}/48 units; {time.monotonic()-start:.1f}s',flush=True)
        check();counts={}
        for tag in TAGS:
            group=[r for r in rows if r['tag']==tag]
            counts[tag]=dict(total=len(group),valid=sum(r['valid'] for r in group),evaluable=sum(r['evaluable'] for r in group),
                contacts=sum(r['contact'] is True for r in group),controls=sum(r['control'] is True for r in group),
                by_family={family:dict(total=sum(r['family']==family for r in group),
                    contacts=sum(r['family']==family and r['contact'] is True for r in group),
                    controls=sum(r['family']==family and r['control'] is True for r in group)) for family in ('none','corner')},
                physical_roles=dict(Counter(r['role'] for r in group)),invalid_reasons=dict(Counter(reason for r in group for reason in r['reasons'])))
        manifest=dict(status='COMPLETE',rows=rows,units=list(UNITS),tags=list(TAGS),counts=counts,
            requested_sequences=3840,query_order=['HEAD','BODY'],output_frames=G.FRAMES.tolist(),
            poses_sha256=posehashes,geometry_plan_sha256=D.sha(outdir/'GEOMETRY_PLAN.json'),
            source_hashes=plan['hashes'],no_model_scores_read=True,no_render_or_training=True,
            seconds=time.monotonic()-start,budget_seconds=budget_seconds,scope=plan['scope'])
        save(outdir/'head_error_draws.json',dict(seed=ERROR_SEED,pool_sequences=5760,draws=draws,
            pool_source_hashes={k:v for k,v in plan['hashes'].items() if '/units/unit' in k},
            note='Existing real error trajectories reused intact; no new participant/error evidence'))
        save(outdir/'geometry_manifest.json',manifest)
        save(outdir/'GEOMETRY_TERMINAL.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
        return manifest
    except BaseException as exc:
        save(outdir/'GEOMETRY_TERMINAL.json',dict(status='FAILED',seconds=time.monotonic()-start,error=repr(exc)))
        raise


def load_source(unit,config,outdir=OUT):
    with np.load(Path(outdir)/f'poses/unit{int(unit)}.npz',allow_pickle=False) as z:
        ids=np.flatnonzero(z['configs']==int(config))
        if len(ids)!=1:raise ValueError('Missing or duplicate config')
        i=int(ids[0])
        return dict(unit=int(unit),config=int(config),sensor_center=z['sensor_center'],
                    nn=z['nn'][i],qq=z['qq'][i],photon_seed=int(z['photon_seed'][i]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    value=freeze_geometry_plan() if args.stage=='freeze' else build_manifest()
    print(json.dumps({k:value[k] for k in ('status','counts') if k in value},indent=2))
