"""Causal signed radial features from observations and noisy rotation only.

No labels, M3 scores, translation, true pose, travel or exact query are inputs.
Sector order is top-left to bottom-right: (vertical=0..1,horizontal=0..3).
Each sector contains rows [4*v,4*v+4), columns [2*h,2*h+2).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
OUT = WORK / 'cnh-raw-radial-information-dev-20261007'
LEDGER = WORK / 'cnh-querywise-calibration-dev-20261007/ledger.npz'
HB = WORK / 'cnh-double-height-dev-20261007'
FRAMES = np.arange(3, 16)
LAGS = (1, 3, 5)
FOV_DEG = 45.0
GRID_TOL = 1e-10
EDGE = np.tan(np.deg2rad(FOV_DEG / 2))
CENTERS = -EDGE + (np.arange(8) + .5) * (2 * EDGE / 8)
_X, _Y = np.meshgrid(CENTERS, CENTERS)
RAYS = np.stack((_X, _Y, np.ones_like(_X)), axis=-1).reshape(64, 3)
SECTORS = np.array([[r * 8 + c for r in range(4*v, 4*v+4)
                     for c in range(2*h, 2*h+2)]
                    for v in range(2) for h in range(4)])
SCHEMA = dict(version=1,native_frames=FRAMES.tolist(),lags=list(LAGS),
    shapes=dict(current=[13,136],support=[13,24],temporal_delta=[13,384]),
    dtype='float32',fov_deg=FOV_DEG,grid_tolerance=GRID_TOL,
    sectors='2 vertical x 4 horizontal, each 4 rows x 2 columns, row-major',
    current='8 sector-major signed z1 mean vectors of 16 bins, then 8 mean(log1p ambient)',
    temporal='lag-major then sector-major then bin-major',
    warp='Rp.T @ Rc; positive z and center-grid [0,7]; bilinear past signed z1',
    missing='delta NaN, support 0',pose='noisy rotation only; translation ignored')


def extract_features(z1, ambient, noisy_pose):
    """Return float32 current(13,136), support(13,24), delta(13,384).

    Current: 8 sector means x 16 signed bins, then 8 means of log1p(ambient).
    Support/delta: lag (1,3,5), then sector, then radial bin (delta only).
    Backward warp maps CURRENT rays to PAST coordinates using Rp.T @ Rc.
    Only past center-grid coordinates in [0,7] with positive depth contribute;
    tolerance clips roundoff at the boundary, never fills missing FOV zones.
    Missing past frames/empty sectors have support=0 and all-NaN delta.
    """
    z = np.asarray(z1, dtype=np.float64)
    a = np.asarray(ambient, dtype=np.float64)
    pose = np.asarray(noisy_pose, dtype=np.float64)
    if z.shape != (16,8,8,16) or a.shape != (16,8,8) or pose.shape != (16,4,4):
        raise ValueError('Expected z1(16,8,8,16), ambient(16,8,8), pose(16,4,4)')
    # Deliberately validate only the rotation block: translations are not read.
    rotation = pose[:, :3, :3]
    if not np.isfinite(z).all() or not np.isfinite(a).all() or not np.isfinite(rotation).all():
        raise ValueError('Observation and rotation inputs must be finite')
    if (a < 0).any():
        raise ValueError('Ambient must be nonnegative')
    flat = z.reshape(16,64,16)
    now = flat[FRAMES]
    current = np.concatenate((now[:,SECTORS].mean(axis=2).reshape(13,128),
        np.log1p(a[FRAMES].reshape(13,64))[:,SECTORS].mean(axis=2)), axis=1)

    past = FRAMES[None,:] - np.asarray(LAGS)[:,None]
    safe_past = np.maximum(past, 0)
    relative = np.swapaxes(rotation[safe_past], -1, -2) @ rotation[FRAMES][None]
    q = np.einsum('lfij,zj->lfzi', relative, RAYS)
    positive = q[...,2] > 0
    xy = np.divide(q[...,:2], q[...,2,None], out=np.zeros_like(q[...,:2]),
                   where=positive[...,None])
    grid = (xy + EDGE) * (8 / (2*EDGE)) - .5
    valid = (positive & (past >= 0)[...,None]
             & (grid >= -GRID_TOL).all(axis=-1)
             & (grid <= 7 + GRID_TOL).all(axis=-1))
    grid = np.clip(grid, 0, 7)
    col, row = grid[...,0], grid[...,1]
    c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
    c1, r1 = np.minimum(c0+1,7), np.minimum(r0+1,7)
    wc, wr = (col-c0)[...,None], (row-r0)[...,None]
    pi = safe_past[...,None]
    warped = ((1-wr)*((1-wc)*z[pi,r0,c0] + wc*z[pi,r0,c1])
              + wr*((1-wc)*z[pi,r1,c0] + wc*z[pi,r1,c1]))
    common = valid[...,SECTORS]
    counts = common.sum(axis=-1)
    difference = now[None,:,SECTORS] - warped[...,SECTORS,:]
    summed = np.where(common[...,None], difference, 0).sum(axis=-2)
    delta = np.full(summed.shape, np.nan)
    np.divide(summed, counts[...,None], out=delta, where=counts[...,None]>0)
    return dict(current=current.astype(np.float32),
                support=(counts.transpose(1,0,2).reshape(13,24)/8).astype(np.float32),
                temporal_delta=delta.transpose(1,0,2,3).reshape(13,384).astype(np.float32))


def observation_path(unit):
    unit = int(unit)
    if 98000 <= unit < 98048:
        return WORK / f'cnh-extrinsic-aug-20261006/observations/calibration/unit{unit}.npz'
    if 99000 <= unit < 99096:
        return WORK / f'cnh-extrinsic-aug-20261006/continuation-r1/observations/evaluation/unit{unit}.npz'
    raise ValueError(f'Unsupported source unit {unit}')


def extract_all(outdir=OUT, budget_seconds=900):
    """Extract original + physical HB observations, only after PLAN.json exists.

    Every source NPZ is read once into bytes, hashed and decoded once. Original
    unit/config row order comes from the existing querywise ledger; HB order is
    manifest anchors x (H,B,HB). Manifest contents only supply identity/path.
    No full extraction runs at import time. Existing output files are preserved.
    """
    outdir = Path(outdir)
    if not (outdir/'PLAN.json').is_file():
        raise FileNotFoundError('Freeze PLAN.json before starting extraction')
    plan = json.loads((outdir/'PLAN.json').read_text(encoding='utf8'))
    budget_seconds = min(float(budget_seconds),float(plan['budget_extract_wall_seconds']))
    if not np.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError('A positive finite frozen extraction budget is required')
    outputs = ('features.npz','hb_features.npz','features_metadata.json')
    for name in outputs:
        if (outdir/name).exists():
            raise FileExistsError(outdir/name)
    start = time.perf_counter()
    hashes = {}

    def budget():
        if time.perf_counter()-start > budget_seconds:
            raise TimeoutError(f'Feature extraction exceeded {budget_seconds}s budget')

    def read_bytes(path):
        budget()
        payload = Path(path).read_bytes()
        hashes[Path(path).relative_to(ROOT).as_posix()] = hashlib.sha256(payload).hexdigest()
        return payload

    def allocate(n):
        return {k:np.empty((n,13,d),np.float32) for k,d in
                (('current',136),('support',24),('temporal_delta',384))}

    with np.load(io.BytesIO(read_bytes(LEDGER)), allow_pickle=False) as ledger:
        unit = ledger['unit'].astype(np.int64)
        config = ledger['config'].astype(np.int64)
    if unit.shape != (5760,) or config.shape != unit.shape or len(set(zip(unit,config))) != 5760:
        raise ValueError('Expected 5760 unique unit/config rows')
    expected_units = set(range(98000,98048)) | set(range(99000,99096))
    if set(unit) != expected_units:
        raise ValueError('Expected the existing 144 Development units')
    original = allocate(len(unit))
    original.update(unit=unit,config=config)
    for completed,u in enumerate(dict.fromkeys(unit.tolist()),start=1):
        indices = np.flatnonzero(unit == u)
        with np.load(io.BytesIO(read_bytes(observation_path(u))), allow_pickle=False) as obs:
            z, a, poses, configs = obs['z1'][0],obs['ambient'][0],obs['noisy_center'],obs['configs']
        mapping = {int(c):i for i,c in enumerate(configs)}
        if len(mapping) != 40 or len(indices) != 40 or set(config[indices]) != set(mapping):
            raise ValueError(f'Unit {u} does not contain exactly the requested 40 configs')
        for index in indices:
            ci = mapping[int(config[index])]
            features = extract_features(z[ci], a[ci], poses[ci])
            for key,value in features.items():
                original[key][index] = value
        budget()
        if completed % 24 == 0:
            print(f'Original observations: {completed}/144 units; '
                  f'{time.perf_counter()-start:.1f}s elapsed',flush=True)
    manifest = json.loads(read_bytes(HB/'geometry_manifest.json'))
    identities = [(str(anchor['anchor_id']),int(anchor['unit']),int(anchor['config']))
                  for anchor in manifest['anchors']]
    if len(identities) != 72 or len({v[0] for v in identities}) != 72:
        raise ValueError('Expected the existing 72 unique physical anchors')
    hb = allocate(216)
    ids, tags, units, configs = [], [], [], []
    for aid,u,c in identities:
        for tag in ('H','B','HB'):
            path = HB/'anchors'/aid/f'{tag}.npz'
            with np.load(io.BytesIO(read_bytes(path)), allow_pickle=False) as obs:
                features = extract_features(obs['z1'],obs['ambient'],obs['nn'])
            index = len(ids)
            for key,value in features.items():
                hb[key][index] = value
            ids.append(aid);tags.append(tag);units.append(u);configs.append(c)
    hb.update(anchor_id=np.asarray(ids),tag=np.asarray(tags),
              unit=np.asarray(units,np.int64),config=np.asarray(configs,np.int64))
    extraction_seconds = time.perf_counter()-start
    budget()
    for name,arrays in (('features.npz',original),('hb_features.npz',hb)):
        with (outdir/name).open('xb') as stream:
            np.savez_compressed(stream,**arrays)
        budget()
    metadata = dict(status='complete',schema=SCHEMA,native_frames=FRAMES.tolist(),lags=list(LAGS),
        fov_deg=FOV_DEG,grid_tolerance=GRID_TOL,sector_zone_indices=SECTORS.tolist(),
        current_layout='sector-major 8x16 signed mean z1; then 8 mean(log1p ambient)',
        temporal_layout='lag-major, sector-major, radial-bin-major',
        missing='delta NaN and support 0 for empty common FOV or unavailable past',
        pose_contract='Only noisy rotation; Rp.T@Rc backward warp; translation ignored',
        original_order='querywise ledger unit/config rows',hb_order='manifest anchors x H,B,HB',
        original_sequences=len(unit),hb_sequences=len(ids),observation_npz_read_count=144+216,
        source_sha256=hashes,extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256((outdir/'PLAN.json').read_bytes()).hexdigest(),
        shapes={name:{k:list(v.shape) for k,v in arrays.items()} for name,arrays in
                (('features.npz',original),('hb_features.npz',hb))},
        feature_dtype='float32',identity_dtype='int64 or unicode',
        partial_resume_supported=False,
        partial_output_policy='Partial extraction cannot resume; preserve files and report failure',
        extraction_seconds=extraction_seconds,total_seconds=time.perf_counter()-start,
        budget_seconds=budget_seconds)
    with (outdir/'features_metadata.json').open('x',encoding='utf8') as stream:
        json.dump(metadata,stream,indent=2,ensure_ascii=False,allow_nan=False)
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=OUT)
    parser.add_argument('--budget-seconds',type=float,default=900)
    args = parser.parse_args()
    result = extract_all(args.outdir,args.budget_seconds)
    print(json.dumps({key:result[key] for key in
          ('status','original_sequences','hb_sequences','total_seconds')},indent=2))
