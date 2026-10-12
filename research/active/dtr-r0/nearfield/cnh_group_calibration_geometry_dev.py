"""Original/HB calibration groups on stored shared-calibration poses.

Keep all original records exactly and compute HB eligibility independently.
No new scene sampling, poses, errors, scores, fitting or rendering.
"""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import cnh_shared_calibration_geometry_dev as C

M=C.M
ROOT,WORK=M.ROOT,M.WORK
SOURCE_OUT=C.OUT
OUT=WORK/'cnh-group-calibration-dev-20261007'
UNITS=C.UNITS
TAGS=('original','HB')


def original_manifest():
    source=M.D.read(SOURCE_OUT/'geometry_manifest.json')
    if source['status']!='COMPLETE':raise ValueError('Original calibration geometry incomplete')
    rows=source['rows']
    expected=[(u,c,'original') for u in UNITS for c in range(40)]
    if [(r['unit'],r['config'],r['tag']) for r in rows]!=expected:
        raise ValueError('Expected all1920 original records in frozen unit/config order')
    for row in rows:
        strict=bool(row['valid'] and row['control'] is True and row['contact'] is False
                    and (np.asarray(row['frame_category'])=='clear').all())
        if row['eligible_control']!=strict or row['evaluable']!=strict:
            raise ValueError('Original strict-clear eligibility mismatch')
    return source


def hb_record(original,travel):
    """Independent HB geometry, including original-ineligible configurations."""
    boxes=copy.deepcopy(original['boxes'])
    boxes[0]['lo'][1],boxes[0]['hi'][1]=-.1,.84
    geometry=C.calibration_geometry(M.evaluate_geometry(boxes,travel))
    # Y-only edits cannot change the original yaw-only target-front crossing.
    if geometry['covered']!=original['covered'] or geometry['censor_reason']!=original['censor_reason']:
        raise ValueError('HB height edit changed source reference coverage')
    np.testing.assert_allclose(geometry['frame_ranges'],original['frame_ranges'],atol=1e-10,rtol=0)
    if geometry['covered']:
        if abs(geometry['reference_fraction']-original['reference_fraction'])>1e-10 or geometry['deadline_index']!=original['deadline_index']:
            raise ValueError('HB height edit changed source causal deadline')
        np.testing.assert_allclose(geometry['reference_pose'],original['reference_pose'],atol=1e-10,rtol=0)
    result=copy.deepcopy(original)
    result.update(geometry)
    result.update(tag='HB',row_id=original['anchor_id']+'_HB')
    return result


def freeze_geometry_plan(outdir=OUT):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    source=original_manifest();contract=C.unit_contract()
    paths=[Path(__file__),Path(C.__file__),*M.source_files(),
           SOURCE_OUT/'GEOMETRY_PLAN.json',SOURCE_OUT/'geometry_manifest.json',
           SOURCE_OUT/'GEOMETRY_TERMINAL.json',SOURCE_OUT/'head_error_draws.json']
    paths.extend(SOURCE_OUT/path for path in source['poses_sha256'])
    for path,expected in source['poses_sha256'].items():
        if M.D.sha(SOURCE_OUT/path)!=expected:raise ValueError('Stored source pose hash mismatch')
    plan=dict(status='frozen',units=list(UNITS),configs=list(range(40)),tags=list(TAGS),
        expected_sequences=3840,geometry_budget_seconds=300,unit_contract=contract,
        original='Retain all1920 original rows exactly, including labels and missingness; original observations reused downstream',
        hb='Recompute all1920 physical HB rows by changing only target y to[-.1,.84], using stored travel',
        eligibility='Each group uses its own valid strict-clear controls. HB is not filtered by original eligibility or score.',
        new_render_scope='Only HB eligible controls; original observations fully reused',
        no_new_poses_or_errors=True,no_scores_or_models_read=True,no_render_or_training=True,
        poses_root=SOURCE_OUT.relative_to(ROOT).as_posix(),poses_sha256=source['poses_sha256'],
        scope='Consumed shared calibration units and existing error draws; group calibration on synthetic structures, no independent evidence',
        hashes={p.relative_to(ROOT).as_posix():M.D.sha(p) for p in dict.fromkeys(paths)})
    M.save(outdir/'GEOMETRY_PLAN.json',plan)
    return plan


def build_manifest(outdir=OUT,budget_seconds=300):
    outdir=Path(outdir);start=time.monotonic();plan=M.D.read(outdir/'GEOMETRY_PLAN.json')
    budget_seconds=min(float(budget_seconds),float(plan['geometry_budget_seconds']))
    if not np.isfinite(budget_seconds) or budget_seconds<=0:raise ValueError('Positive finite frozen budget required')
    for name in ('GEOMETRY_STARTED.json','geometry_manifest.json'):
        if (outdir/name).exists():raise FileExistsError(outdir/name)
    M.save(outdir/'GEOMETRY_STARTED.json',dict(unix=time.time(),budget_seconds=budget_seconds))
    def check():
        if time.monotonic()-start>budget_seconds:raise TimeoutError('300s group calibration geometry budget reached')
    try:
        if plan['units']!=list(UNITS) or plan['configs']!=list(range(40)) or plan['tags']!=list(TAGS):
            raise ValueError('Frozen identities changed')
        if plan['unit_contract']!=C.unit_contract():raise ValueError('Unit separation changed')
        for path,expected in plan['hashes'].items():
            check()
            if M.D.sha(ROOT/path)!=expected:raise ValueError(f'Frozen source changed: {path}')
        source=original_manifest();originals=source['rows'];rows=[]
        for ui,unit in enumerate(UNITS):
            check()
            with np.load(SOURCE_OUT/f'poses/unit{unit}.npz',allow_pickle=False) as poses:
                travel=poses['travel']
                np.testing.assert_array_equal(poses['configs'],np.arange(40))
            for config in range(40):
                check();original=originals[ui*40+config]
                if (original['unit'],original['config'])!=(unit,config):raise ValueError('Original row order changed')
                rows.extend((copy.deepcopy(original),hb_record(original,travel)))
            if (ui+1)%6==0:print(f'Group calibration HB geometry {ui+1}/48 units; {time.monotonic()-start:.1f}s',flush=True)
        if rows[::2]!=originals:raise ValueError('Original records must reproduce exactly')
        counts={}
        for tag in TAGS:
            group=[row for row in rows if row['tag']==tag]
            counts[tag]=C.summarize(group)
            counts[tag]['by_family']={family:C.summarize([row for row in group if row['family']==family]) for family in ('none','corner')}
        a=np.asarray([r['eligible_control'] for r in rows[::2]])
        b=np.asarray([r['eligible_control'] for r in rows[1::2]])
        check()
        manifest=dict(status='COMPLETE',rows=rows,units=list(UNITS),tags=list(TAGS),counts=counts,
            requested_sequences=3840,query_order=['HEAD','BODY'],output_frames=M.G.FRAMES.tolist(),
            eligible_pair_counts=dict(both=int((a&b).sum()),original_only=int((a&~b).sum()),hb_only=int((~a&b).sum()),neither=int((~a&~b).sum())),
            original_records_exact=True,original_manifest_sha256=M.D.sha(SOURCE_OUT/'geometry_manifest.json'),
            original_observations_root=SOURCE_OUT.relative_to(ROOT).as_posix(),
            poses_root=SOURCE_OUT.relative_to(ROOT).as_posix(),poses_sha256=source['poses_sha256'],
            head_error_draws_source=(SOURCE_OUT/'head_error_draws.json').relative_to(ROOT).as_posix(),
            geometry_plan_sha256=M.D.sha(outdir/'GEOMETRY_PLAN.json'),source_hashes=plan['hashes'],
            no_new_poses_or_errors=True,no_model_scores_read=True,no_render_or_training=True,
            eligibility_contract='Own-group eligible_control/evaluable only; HB independent of original eligibility',
            seconds=time.monotonic()-start,budget_seconds=budget_seconds,scope=plan['scope'])
        M.save(outdir/'geometry_manifest.json',manifest)
        M.save(outdir/'GEOMETRY_TERMINAL.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
        return manifest
    except BaseException as exc:
        M.save(outdir/'GEOMETRY_TERMINAL.json',dict(status='FAILED',seconds=time.monotonic()-start,error=repr(exc)))
        raise


def load_source(unit,config,outdir=OUT):
    """Always read shared calibration poses; outdir kept for driver API parity."""
    return C.load_source(unit,config,SOURCE_OUT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    value=freeze_geometry_plan() if args.stage=='freeze' else build_manifest()
    print(value['status'])
