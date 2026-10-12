"""Original-only geometry for a new shared strict-clear calibration batch.

Reuse the frozen stability generator/pose helpers without changing their globals.
No new scores, model weights, rendering, fitting or threshold selection here.
"""
import argparse
from collections import Counter
import copy
from pathlib import Path
import time

import numpy as np
import cnh_model_stability_geometry_dev as M

ROOT,WORK = M.ROOT,M.WORK
OUT = WORK/'cnh-shared-calibration-dev-20261007'
UNITS = tuple(range(410052,410100))
TAGS = ('original',)


def unit_contract():
    """New scene-unit separation and the deterministic motion/mirror balance."""
    excluded={'stability_evaluation':set(M.UNITS),'original_tree_units':set(M.POOL_UNITS),
              'm3_training':set(range(93000,93096))}
    overlaps={key:sorted(set(UNITS)&ids) for key,ids in excluded.items()}
    if any(overlaps.values()):raise ValueError(f'Shared calibration units overlap: {overlaps}')
    balance=Counter((u%3,bool((u//3)%2)) for u in UNITS)
    if len(UNITS)!=48 or set(balance.values())!={8} or len(balance)!=6:
        raise ValueError('Expected eight units in each mode x mirror cell')
    return dict(overlaps=overlaps,mode_mirror_counts={f'mode{mode}/mirror{int(mirror)}':n
                for (mode,mirror),n in sorted(balance.items())})


def calibration_geometry(physical):
    """Preserve physical labels but restrict eligibility to strict-clear rows.

    A censored target-front crossing does not invalidate a sequence whose two
    queries are clear at every output. Censored nonclear rows are ineligible.
    """
    result=copy.deepcopy(physical)
    strict_clear=bool(np.asarray(result['frame_category']).shape==(13,2)
                      and (np.asarray(result['frame_category'])=='clear').all())
    eligible=bool(result['valid'] and result['control'] is True and strict_clear
                  and result['contact'] is False)
    result['geometry_evaluable']=bool(result['evaluable'])
    result['evaluable']=eligible
    result['eligible_control']=eligible
    result['calibration_exclusion']=None if eligible else (
        'invalid_geometry' if not result['valid'] else 'contact' if result['contact'] else
        'censored_nonclear' if not result['covered'] else 'pass_or_nonclear')
    if result['valid'] and result['control'] is True and not strict_clear:
        raise ValueError('Control label contradicts all-frame strict-clear geometry')
    return result


def freeze_geometry_plan(outdir=OUT):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    contract=unit_contract()
    paths=[Path(__file__),*M.source_files(),M.OUT/'GEOMETRY_PLAN.json',
           *[M.POOL/f'unit{u}.npz' for u in M.POOL_UNITS]]
    plan=dict(status='frozen',units=list(UNITS),configs=list(range(40)),tags=list(TAGS),
        expected_sequences=1920,geometry_budget_seconds=300,unit_contract=contract,
        scene_generator='Frozen M.MC.scenes_for(unit), original40 configurations; none20/corner20; no height edit',
        selection='Keep every row. Render and calibrate only eligible_control: valid, no contact, both queries clear at all13 outputs',
        censoring='Unchanged strict-clear controls may have no0.9m crossing; censored_nonclear is never a calibration negative',
        error_seed=M.ERROR_SEED,error_pool_units=list(M.POOL_UNITS),error_pool_key='head_c_err',
        error_sampling='Intact16-frame old5760-pool sequence sampled by seed/unit/config; no new sign, scale or error evidence',
        noisy_pose_seed_prefix=M.NOISE_SEED,photon_seed_prefix=M.PHOTON_SEED,
        no_scores_or_models_read=True,no_render_or_training=True,
        evaluation_separation='New calibration units410052..410099; fixed existing stability evaluation410004..410051 reused only downstream',
        scope='Shared original-only synthetic calibration; same generator and reused real error pool, not new participant evidence',
        hashes={p.relative_to(ROOT).as_posix():M.D.sha(p) for p in paths})
    M.save(outdir/'GEOMETRY_PLAN.json',plan)
    return plan


def summarize(rows):
    return dict(total=len(rows),valid=sum(r['valid'] for r in rows),
        evaluable=sum(r['evaluable'] for r in rows),eligible_controls=sum(r['eligible_control'] for r in rows),
        contacts=sum(r['contact'] is True for r in rows),controls=sum(r['control'] is True for r in rows),
        physical_roles=dict(Counter(r['role'] for r in rows)),
        calibration_exclusions=dict(Counter(r['calibration_exclusion'] for r in rows if not r['eligible_control'])),
        invalid_reasons=dict(Counter(reason for r in rows for reason in r['reasons'])))


def build_manifest(outdir=OUT,budget_seconds=300):
    outdir=Path(outdir);start=time.monotonic();plan=M.D.read(outdir/'GEOMETRY_PLAN.json')
    budget_seconds=min(float(budget_seconds),float(plan['geometry_budget_seconds']))
    if not np.isfinite(budget_seconds) or budget_seconds<=0:raise ValueError('Positive finite frozen budget required')
    for name in ('GEOMETRY_STARTED.json','geometry_manifest.json'):
        if (outdir/name).exists():raise FileExistsError(outdir/name)
    M.save(outdir/'GEOMETRY_STARTED.json',dict(unix=time.time(),budget_seconds=budget_seconds))
    def check():
        if time.monotonic()-start>budget_seconds:raise TimeoutError('300s shared calibration geometry budget reached')
    try:
        if plan['units']!=list(UNITS) or plan['tags']!=list(TAGS) or plan['configs']!=list(range(40)):
            raise ValueError('Frozen shared calibration identities changed')
        contract=unit_contract()
        if plan['unit_contract']!=contract:raise ValueError('Frozen unit separation contract changed')
        for path,expected in plan['hashes'].items():
            check()
            if M.D.sha(ROOT/path)!=expected:raise ValueError(f'Frozen source changed: {path}')
        pool=[]
        for unit in M.POOL_UNITS:
            check()
            with np.load(M.POOL/f'unit{unit}.npz',allow_pickle=False) as saved:errors=saved['head_c_err']
            if errors.shape!=(40,16) or not np.isfinite(errors).all():raise ValueError('Invalid existing error pool')
            pool.extend(errors)
        pool=np.asarray(pool)
        rows=[];draws=[];posehashes={};(outdir/'poses').mkdir(exist_ok=True)
        for ui,unit in enumerate(UNITS):
            check();scenes=M.MC.scenes_for(unit)
            if len(scenes)!=40:raise ValueError('Original generator must yield40 configurations')
            nn=[];qq=[];errors=[];su=[];sc=[];seeds=[];shared=None
            mirrored=bool((unit//3)%2)
            for config,scene in enumerate(scenes):
                check();index,error=M.choose_error(pool,unit,config)
                oldunit,oldconfig=M.POOL_UNITS[index//40],index%40
                obs=M.poses_for(scene,unit,config,error)
                if shared is None:shared={k:obs[k] for k in ('sensor_center','travel','public_query')}
                else:
                    for key,value in shared.items():np.testing.assert_array_equal(obs[key],value)
                nn.append(obs['nn']);qq.append(obs['qq']);errors.append(error);su.append(oldunit);sc.append(oldconfig);seeds.append(obs['photon_seed'])
                draws.append(dict(unit=unit,config=config,pool_index=index,source_unit=oldunit,source_config=oldconfig))
                physical=calibration_geometry(M.evaluate_geometry(M.physical_boxes(scene,unit,'original'),obs['travel']))
                anchor_id=f'u{unit}_c{config}'
                rows.append(dict(anchor_id=anchor_id,row_id=anchor_id+'_original',unit=unit,config=config,tag='original',
                    family=scene['family'],mode=unit%3,mirror=mirrored,
                    turn=('left' if mirrored else 'right') if unit%3==2 else 'none',source_group=int(scene['group']),
                    poses_path=f'poses/unit{unit}.npz',pose_config_index=config,
                    head_error_source_unit=oldunit,head_error_source_config=oldconfig,photon_seed=obs['photon_seed'],**physical))
            path=outdir/f'poses/unit{unit}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream,configs=np.arange(40),**shared,nn=np.asarray(nn),qq=np.asarray(qq),head_c_err=np.asarray(errors),
                    head_error_source_unit=np.asarray(su),head_error_source_config=np.asarray(sc),photon_seed=np.asarray(seeds,dtype=np.uint32))
            posehashes[path.relative_to(outdir).as_posix()]=M.D.sha(path)
            if (ui+1)%6==0:print(f'Calibration geometry {ui+1}/48 units; {time.monotonic()-start:.1f}s',flush=True)
        check()
        counts=summarize(rows)
        counts['by_family']={family:summarize([r for r in rows if r['family']==family]) for family in ('none','corner')}
        counts['by_mode_mirror']={f'mode{mode}/mirror{int(mirror)}':summarize([r for r in rows if r['mode']==mode and r['mirror']==mirror])
            for mode in range(3) for mirror in (False,True)}
        manifest=dict(status='COMPLETE',rows=rows,units=list(UNITS),tags=list(TAGS),counts={'original':counts},
            requested_sequences=1920,query_order=['HEAD','BODY'],output_frames=M.G.FRAMES.tolist(),
            eligibility_contract='evaluable equals eligible_control; calibration only, not event evaluation',
            poses_sha256=posehashes,geometry_plan_sha256=M.D.sha(outdir/'GEOMETRY_PLAN.json'),
            source_hashes=plan['hashes'],no_model_scores_read=True,no_render_or_training=True,
            seconds=time.monotonic()-start,budget_seconds=budget_seconds,scope=plan['scope'])
        M.save(outdir/'head_error_draws.json',dict(seed=M.ERROR_SEED,pool_sequences=5760,draws=draws,
            pool_source_hashes={k:v for k,v in plan['hashes'].items() if '/units/unit' in k},
            note='Existing real error trajectories reused intact; no new participant/error evidence'))
        M.save(outdir/'geometry_manifest.json',manifest)
        M.save(outdir/'GEOMETRY_TERMINAL.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
        return manifest
    except BaseException as exc:
        M.save(outdir/'GEOMETRY_TERMINAL.json',dict(status='FAILED',seconds=time.monotonic()-start,error=repr(exc)))
        raise


def load_source(unit,config,outdir=OUT):
    return M.load_source(unit,config,outdir)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    result=freeze_geometry_plan() if args.stage=='freeze' else build_manifest()
    print(result['status'])
