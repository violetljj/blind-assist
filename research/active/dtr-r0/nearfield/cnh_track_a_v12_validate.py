"""Independent read-only v1.2 mesh/label/identity/quota validator.

Consumes only emitted unit JSON, never changes or completes a cohort.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from cnh_track_a_generate import BOXES, PATTERNS, check_unit
from cnh_track_a_geometry import signed_margin, clip_triangles
from cnh_track_a_validate import exact_voxels

FAMILY='cnh-track-a-v12-20260925'
PURPOSES=('layout','trajectory','size_placement','material','noise','ego_motion')


def expected_seed(unit,config,purpose,candidate=0):
    return int(hashlib.sha256(f'{FAMILY}|{unit}|{config}|{purpose}|{candidate}'.encode()).hexdigest()[:16],16)


def seed_errors(config,unit):
    errors=[];ci=config['config'];candidate=config.get('candidate',-1);seeds=config.get('seeds',{})
    if not isinstance(candidate,int) or not 0<=candidate<32:return ['candidate_index']
    if set(seeds)!=set(PURPOSES):errors.append('purpose_seed_keys')
    for purpose in PURPOSES:
        attempt=candidate if purpose in ('size_placement','material') else 0
        seed_config=-1 if purpose=='layout' else ci
        if seeds.get(purpose)!=expected_seed(unit,seed_config,purpose,attempt):errors.append('seed_'+purpose)
    if config.get('seed')!=expected_seed(unit,ci,'trajectory'):errors.append('trajectory_seed')
    return errors


def schedule(unit):
    common=[(0,0),(1,0),(5,0),(2,0),(0,1),(0,5),(0,4)]
    tail=([(3,3),(1,2),(2,1),(4,5),(5,4),(3,6),(6,3)] if unit%2==0 else
          [(6,6),(1,4),(4,1),(2,5),(5,2),(3,2),(2,3)])
    return (common+tail)*2


def different_query_objects(contributors):
    return any(a!=b for q,ids in enumerate(contributors) for r,others in enumerate(contributors)
               if q!=r for a in ids for b in others)


def recompute_config(config):
    """All margin->label identities; exact geometry audit at frames0/7/11.

    Geometry helper is unchanged and separately tested. Do not repeat the full
    generator LP workload; the deterministic three-frame sample is disclosed.
    """
    objects=config['objects'];poses=np.asarray(config['world_from_Q'],float)
    ids=[o['id'] for o in objects]
    margins=[];contributors=[];witness=[]
    for ti,pose in enumerate(poses):
        if ti not in (0,7,11):
            mm=np.asarray(config['margins'][ti],float)
            margins.append(mm)
            contributors.append([[oid for oid,value in zip(ids,mm[:,q]) if value>=-1e-10] for q in range(6)])
            witness.append(config['witness_z'][ti])
            continue
        local=[(np.asarray(o['triangles_world'],float)-pose[:3,3])@pose[:3,:3] for o in objects]
        mm=np.array([[signed_margin(mesh,*box) for box in BOXES] for mesh in local])
        margins.append(mm)
        contributors.append([[oid for oid,value in zip(ids,mm[:,q]) if value>=-1e-10] for q in range(6)])
        zs=[]
        for q,box in enumerate(BOXES):
            candidates=[]
            for mesh,m in zip(local,mm[:,q]):
                if m>=-1e-10:
                    clipped=clip_triangles(mesh,*box)
                    z=float(clipped[...,2].min()) if len(clipped) else float(signed_margin(mesh,*box,return_witness=True)[1][2])
                    candidates.append(z)
            zs.append(min(candidates) if candidates else None)
        witness.append(zs)
    margins=np.array(margins)
    labels=(margins.max(1)>=-1e-10).astype(int)
    boundary=(np.abs(margins)<.05).any((1,2))
    main=(np.arange(12)>=3)&~boundary
    return dict(**{k:v for k,v in config.items() if k not in ('margins','labels','contributors','witness_z','boundary','main')},
                margins=margins.tolist(),labels=labels.tolist(),contributors=contributors,witness_z=witness,
                boundary=boundary.tolist(),main=main.tolist())


def validate_config(config,unit):
    errors=seed_errors(config,unit);ci=config['config'];prefix=f'{unit}/{ci}'
    poses=np.asarray(config['world_from_Q'],float);angles=np.asarray(config['angles'],float)
    objects=config['objects'];ids=[o['id'] for o in objects]
    if poses.shape!=(12,4,4) or angles.shape!=(12,3):
        raise ValueError(prefix+': requires12poses/angles')
    if len(ids)!=len(set(ids)) or ids!=config['object_ids']:errors.append('object_identity')
    stored_margin=np.asarray(config['margins'],float)
    if stored_margin.shape!=(12,len(objects),6) or not np.isfinite(stored_margin).all():
        raise ValueError(prefix+': malformed/nonfinite margins')
    if not np.isfinite(poses).all() or not np.isfinite(angles).all():errors.append('nonfinite_pose')
    if not np.allclose(poses[:,3],[0,0,0,1],atol=1e-10):errors.append('homogeneous_pose')
    rotations=poses[:,:3,:3]
    if not np.allclose(rotations@rotations.transpose(0,2,1),np.eye(3),atol=1e-8):errors.append('rotation')
    if not np.allclose(np.linalg.det(rotations),1,atol=1e-8):errors.append('rotation_determinant')
    if not np.allclose(rotations[:,1,:],[0,1,0],atol=1e-8):errors.append('Q_gravity_axis')
    if not .8<=config['speed']<=1.4 or not np.allclose(np.diff(poses[:,2,3]),config['speed']*.2,atol=1e-6):errors.append('speed')
    if not 1.45<=config['height']<=1.75:errors.append('height')
    if not ((angles>=[-15,-10,-30])&(angles<=[10,10,30])).all() or np.abs(np.diff(angles[:,2])).max()>12+1e-9:errors.append('head_pose')
    related=[]
    for obj in objects:
        mesh=np.asarray(obj['triangles_world'],float)
        if mesh.ndim!=3 or mesh.shape[1:]!=(3,3) or not len(mesh) or not np.isfinite(mesh).all():
            raise ValueError(prefix+': malformed mesh')
        if obj['category'] in ('HEAD','BODY'):
            related.append((mesh-poses[7,:3,3])@rotations[7])
    rebuilt=recompute_config(config)
    for key in ('labels','boundary','main'):
        if not np.array_equal(rebuilt[key],config[key]):errors.append(key)
    if not np.allclose(rebuilt['margins'],config['margins'],rtol=0,atol=1e-4):errors.append('exact_margin')
    if rebuilt['contributors']!=config['contributors']:errors.append('contributors')
    for actual,saved in zip(rebuilt['witness_z'],config['witness_z']):
        for a,b in zip(actual,saved):
            if (a is None)!=(b is None) or (a is not None and abs(a-b)>1e-4):errors.append('witness_z')
    for labels,witness in zip(rebuilt['labels'],config['witness_z']):
        for label,z in zip(labels,witness):
            if label and (z is None or not np.isfinite(z) or not .3-1e-8<=z<=3+1e-8):errors.append('witness_schema')
            if not label and z is not None:errors.append('negative_witness')
    y=np.array(rebuilt['labels']);main=np.array(rebuilt['main'])
    multi_frames=sum(bool(m) and different_query_objects(cs) for m,cs in zip(main,rebuilt['contributors']))
    target_frames=None
    if config.get('planned_multi')!=(ci<28 and ci%14>=7):errors.append('planned_multi_flag')
    if ci<28:
        pair=schedule(unit)[ci]
        wanted=np.array([PATTERNS[pair[q%2]][q//2] for q in range(6)])
        if not np.array_equal(config['target_labels'],wanted):errors.append('target_plan')
        target_frames=int((main&(y==wanted).all(1)).sum())
        if target_frames<2:errors.append('planned_target_less_than2_main')
        if ci%14>=7 and multi_frames<6:errors.append('planned_multi_less_than6_main')
        if config.get('boundary_plan') is not None:errors.append('unexpected_boundary_plan')
    else:
        plan=config.get('boundary_plan')
        if not isinstance(plan,dict) or set(plan)!= {'object_id','query','target_margin'}:
            errors.append('boundary_plan_schema')
        else:
            target=plan['target_margin'];query=plan['query']
            expected_positive=ci in (28,29)
            if (not .005<abs(target)<.04) or (target>0)!=expected_positive:errors.append('boundary_margin_plan')
            if plan['object_id'] not in ids or query not in range(6):errors.append('boundary_witness_identity')
            else:
                actual=np.asarray(rebuilt['margins'])[7,ids.index(plan['object_id']),query]
                if abs(actual-target)>1e-4:errors.append('boundary_exact_selected_margin')
    fingerprint=exact_voxels(np.concatenate(related)) if related else set()
    return rebuilt,dict(config=ci,errors=sorted(set(errors)),N_main=int(main.sum()),target_frames=target_frames,
                        planned_multi=ci<28 and ci%14>=7,multi_distinct_query_frames=int(multi_frames)),fingerprint


def run(folder):
    start=time.monotonic();folder=Path(folder);errors=[];fingerprints=[];units=[];hashes={};g2={};details={};split_combos={};seed_authorities={}
    for path in sorted(folder.glob('unit[0-9][0-9].json')):
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        saved=json.loads(path.read_text(encoding='utf-8'));u=saved['unit'];units.append(u)
        split='train' if u<6 else 'calib' if u<8 else 'audit'
        if u not in range(12) or saved['split']!=split:errors.append(f'u{u}:split')
        if saved.get('layout_seed')!=expected_seed(u,-1,'layout'):errors.append(f'u{u}:layout_seed')
        trajectory_path=folder/f'trajectories-unit{u:02d}.json'
        if trajectory_path.is_file():
            hashes[trajectory_path.name]=hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
            paths=json.loads(trajectory_path.read_text())['paths']
            if len(paths)!=32:errors.append(f'u{u}:32_prior_trajectories')
        else:paths=[];errors.append(f'u{u}:missing_prior_trajectories')
        configs=saved['configs']
        if [c['config'] for c in configs]!=list(range(32)):errors.append(f'u{u}:32_ordered_configs')
        rebuilt=[];detail=[]
        for config in configs:
            try:
                ci=config['config']
                for purpose,value in config.get('seeds',{}).items():
                    authority=(u,-1 if purpose=='layout' else ci,purpose,
                               config.get('candidate') if purpose in ('size_placement','material') else 0)
                    if value in seed_authorities and seed_authorities[value]!=authority:errors.append(f'u{u}/c{ci}:seed_collision')
                    seed_authorities[value]=authority
                if ci<len(paths):
                    for key in ('speed','angles','world_from_Q'):
                        if not np.array_equal(config[key],paths[ci][key]):errors.append(f'u{u}/c{ci}:changed_frozen_trajectory_{key}')
                corrected,receipt,fingerprint=validate_config(config,u)
                rebuilt.append(corrected);detail.append(receipt)
                errors.extend(f"u{u}/c{config['config']}:{e}" for e in receipt['errors'])
                if fingerprint:fingerprints.append((u,config['config'],fingerprint))
            except Exception as exc:
                errors.append(f"u{u}/c{config.get('config')}:validation_exception:{exc}")
        if sum(d['planned_multi'] and d['multi_distinct_query_frames']>=6 for d in detail)<14:
            errors.append(f'u{u}:fewer_than14_multi_plans_realized')
        details[str(u)]=detail
        if rebuilt:
            g2[str(u)]=check_unit(rebuilt)
            combos=split_combos.setdefault(split,{})
            for c in rebuilt:
                for row in np.asarray(c['labels'])[c['main']]:
                    combos.setdefault(''.join(map(str,row)),set()).add((u,c['config']))
        else:g2[str(u)]={'pass_gate':False,'failures':['no_valid_configs']}
    duplicates=[];near={}
    for i,(ua,ca,fa) in enumerate(fingerprints):
        for ub,cb,fb in fingerprints[i+1:]:
            inter=len(fa&fb);jac=inter/max(1,len(fa)+len(fb)-inter)
            if fa==fb:duplicates.append([ua,ca,ub,cb])
            if ua!=ub and jac>=.98:
                near.setdefault((ua,ub),set()).add(ca);near.setdefault((ub,ua),set()).add(cb)
    near_failed=[dict(unit=a,other=b,matching_configs=len(cs)) for (a,b),cs in near.items()
                 if len(cs)>=.9*sum(u==a for u,_,_ in fingerprints)]
    split_results={}
    for split,wanted in [('train',set(range(6))),('calib',{6,7}),('audit',set(range(8,12)))]:
        combos=split_combos.get(split,{})
        eligible=sum(len(c)>=2 for c in combos.values())
        complete=wanted.issubset(set(units)) and all(len(details.get(str(u),[]))==32 for u in wanted)
        split_results[split]=dict(complete=complete,eligible_combinations=eligible,required=20,
                                 status=('PASS' if eligible>=20 else 'FAIL') if complete else 'NOT_EVALUABLE_PARTIAL',
                                 combinations={key:[list(v) for v in sorted(value)] for key,value in sorted(combos.items())})
    complete=len(units)==12 and set(units)==set(range(12)) and all(len(v)==32 for v in details.values())
    if len(set(units))!=len(units):errors.append('duplicate_units')
    if not units:errors.append('no_units')
    return dict(schema='cnh.track-a.v12.readonly-audit.v1',cohort_complete=complete,
                encountered_unit_files=units,completed_units=[u for u in units if len(details.get(str(u),[]))==32],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes=hashes,
                G0=dict(pass_gate=not errors,errors=errors),G1=dict(pass_gate=not duplicates and not near_failed,
                nonempty_configs=len(fingerprints),exact_duplicates=duplicates,near_failed=near_failed),
                G2=dict(units=g2,splits=split_results,pass_gate=complete and all(x['pass_gate'] for x in g2.values()) and all(x['status']=='PASS' for x in split_results.values())),
                verification_scope={'all_frames':'margin->labels/boundary/main/contributors, witness schema, poses, G2',
                                    'exact_geometry_frames_per_config':[0,7,11],
                                    'fingerprints':'all nonempty HEAD/BODY anchor meshes, exact1cm SAT'},
                config_checks=details,wall_s=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    receipt=run(args.input);args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('cohort_complete','completed_units','G0','G1','wall_s')}))
