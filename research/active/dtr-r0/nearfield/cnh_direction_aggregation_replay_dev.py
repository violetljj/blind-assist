"""Observation-only absolute-yaw frozen readout replay, without training.

Native rows stay scene -> replica -> frame3..15. Geometry is shared among the
six frozen networks; compact FP16 features are transient. Entire-cohort 0/+3
parity precedes publication of any new angle. Original saved centers remain
authoritative. No CUDA import or work occurs during CPU preparation.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-direction-aggregation-dev-20261009'
DATA = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009/data'
PASS = ROOT/'artifacts.local/work/cnh-pass-boundary-dev-20261009'
SEEDS = (2026100955, 2026100956, 2026100957)
ARMS = ('control', 'weak_pass')
ANGLES = {'cal': (-3, 0, 3), 'validation': (-3, 0, 3, 6)}
SHAPE = (384, 4, 13, 2)
NROWS = 384*4*13
ANGLE_NAMES = {-3: 'minus3', 0: 'zero', 3: 'plus3', 6: 'plus6'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path)
    return dict(path=str(path), resolved_path=str(path.resolve()), bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns, sha256=sha(path))


def rotation(angle):
    """Absolute left sensor-to-query rotation, evaluated in native float64."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = ((c, 0, s), (0, 1, 0), (-s, 0, c))
    return result


def absolute_transforms(native, lengths, angle):
    result = rotation(angle) @ np.asarray(native, np.float64)
    valid = np.arange(8)[None] >= 8-np.asarray(lengths)[:, None]
    # Left padding remains native identity/NaN; the inherited geometry masks it.
    return np.where(valid[:, :, None, None], result, native)


def deduplicate(transforms, lengths):
    lookup, unique_t, unique_l, index = {}, [], [], []
    for transform, length in zip(transforms, lengths):
        key = (int(length), np.asarray(transform, np.float32).tobytes())
        if key not in lookup:
            lookup[key] = len(unique_t)
            unique_t.append(np.asarray(transform, np.float32))
            unique_l.append(int(length))
        index.append(lookup[key])
    return (np.array(unique_t, np.float32), np.array(unique_l, np.int64),
            np.array(index, np.int64))


class Budget:
    """All recorded command-stage wall is charged, including failures and I/O."""
    def __init__(self, phase):
        self.start = time.monotonic()
        self.kind = 'cpu' if phase == 'prepare' else 'gpu'
        self.phase = phase
        self.identifier = time.time_ns()
        self.plan = read(OUT/'PLAN.json')
        self.folder = OUT/'stage_receipts'
        self.folder.mkdir(parents=True, exist_ok=True)
        records = [read(p) for p in self.folder.glob('*.json')]
        # Root and agents use budget_kind; recognize the older generic kind too.
        same = [r for r in records if r.get('budget_kind', r.get('kind')) == self.kind
                or (self.kind == 'gpu' and str(r.get('stage', '')).startswith('runstage_gpu'))]
        self.previous = sum(float(r['seconds']) for r in same)
        self.cap = self.plan[self.kind+'_stage_cap_seconds']
        own = [r for r in same if r.get('stage') == 'replay_prepare']
        self.own_previous = sum(float(r['seconds']) for r in own)
        self.own_cap = self.plan['allocations_cpu']['replay_preparation'] if phase == 'prepare' else self.cap
        self.check()

    def check(self):
        elapsed = time.monotonic()-self.start
        if self.previous+elapsed >= self.cap or self.own_previous+elapsed >= self.own_cap:
            raise TimeoutError('Direction replay cumulative stage wall cap reached')

    def finish(self, status, **details):
        receipt = dict(stage='replay_'+self.phase, budget_kind=self.kind, status=status,
                       seconds=time.monotonic()-self.start, previous_seconds=self.previous,
                       cap_seconds=self.cap, plan_sha256=sha(OUT/'PLAN.json'),
                       source_sha256=sha(__file__), **details)
        save(self.folder/f'runstage_{self.kind}_replay_{self.identifier}.json', receipt)
        return receipt


def sources():
    here = Path(__file__).parent
    return [Path(__file__), here/'cnh_boundary_token_model.py',
            here/'cnh_counterfactual_train_dev.py', here/'cnh_cvr_projection.py']


def check_plan(plan):
    if (tuple(plan['seeds']) != SEEDS or tuple(plan['arms']) != ARMS
            or plan['replay_angles'] != {k:list(v) for k,v in ANGLES.items()}):
        raise ValueError('Frozen replay arm/seed/angle scope differs')


def saved_path(split, arm, seed, angle):
    return PASS/'scores'/f'{arm}_seed{seed}_{split}{"_yaw_plus3" if angle == 3 else ""}.npz'


def load_saved(split, arm, seed, angle):
    with np.load(saved_path(split, arm, seed, angle), allow_pickle=False) as payload:
        raw = payload['raw']
        branch = 'yaw+3' if angle == 3 else 'ideal'
        if (str(payload['arm'].item()) != arm or int(payload['seed'].item()) != seed
                or str(payload['split'].item()) != split or str(payload['branch'].item()) != branch):
            raise ValueError('Saved center identity differs')
        if raw.shape != SHAPE or not np.isfinite(raw).all():
            raise ValueError('Complete finite saved center required')
        return raw.reshape(NROWS, 2).copy()


def prepare():
    budget = Budget('prepare')
    try:
        check_plan(budget.plan)
        if (OUT/'prepare_manifest.json').exists() or (OUT/'prepared').exists():
            raise FileExistsError('Preserve previous complete or partial CPU preparation')
        folder = OUT/'prepared'
        folder.mkdir()
        input_files, geometry, checks = [], [], []
        for split, angles in ANGLES.items():
            arrays = {name: np.load(DATA/split/(name+'.npy'), mmap_mode='r', allow_pickle=False)
                      for name in ('histories', 'transforms', 'transforms_yaw3', 'length')}
            if (arrays['histories'].shape != (NROWS,8,8,8,16)
                    or arrays['transforms'].shape != (NROWS,8,4,4)
                    or arrays['transforms_yaw3'].shape != (NROWS,8,4,4)
                    or arrays['length'].shape != (NROWS,)):
                raise ValueError('Native observation axes differ')
            lengths = np.asarray(arrays['length'])
            if not np.array_equal(lengths, np.tile(np.minimum(np.arange(3,16)+1,8),384*4)):
                raise ValueError('Native scene/replica/frame row binding differs')
            valid = np.arange(8)[None] >= 8-lengths[:,None]
            rebuilt3 = absolute_transforms(arrays['transforms'], lengths, 3)
            difference = np.abs(rebuilt3-arrays['transforms_yaw3'])[valid]
            maximum = float(difference.max())
            if not np.isfinite(maximum) or maximum > 1e-12:
                raise ValueError('Stored +3 differs from absolute public query rotation')
            checks.append(dict(split=split, absolute_yaw3_transform_max_abs=maximum,
                               valid_only=True, padded_transform_policy='retain native padding'))
            for angle in angles:
                budget.check()
                native = arrays['transforms_yaw3'] if angle == 3 else arrays['transforms'] if angle == 0 else absolute_transforms(arrays['transforms'], lengths, angle)
                t, le, index = deduplicate(native, lengths)
                path = folder/f'{split}_geometry_{ANGLE_NAMES[angle]}.npz'
                np.savez_compressed(path, transforms=t, lengths=le, geometry_index=index)
                geometry.append(dict(split=split, angle_degrees=angle, rows=NROWS,
                                     unique_geometries=len(t), **identity(path)))
            for name in arrays:
                budget.check()
                input_files.append(dict(role='observation', split=split, name=name,
                                        **identity(DATA/split/(name+'.npy'))))
        for seed in SEEDS:
            for arm in ARMS:
                budget.check()
                input_files.append(dict(role='checkpoint', arm=arm, seed=seed,
                                        **identity(PASS/'models'/f'{arm}_seed{seed}.pt')))
                for split in ANGLES:
                    for angle in (0,3):
                        load_saved(split,arm,seed,angle)
                        input_files.append(dict(role='saved_center', split=split, angle_degrees=angle,
                                                arm=arm, seed=seed, **identity(saved_path(split,arm,seed,angle))))
        declaration = dict(status='PREPARED', plan_sha256=sha(OUT/'PLAN.json'),
                           source_files=[identity(path) for path in sources()], inputs=input_files,
                           geometry=geometry, transform_checks=checks,
                           arm_names=list(ARMS), seed_ids=list(SEEDS), angles_by_split={k:list(v) for k,v in ANGLES.items()},
                           original_shape=list(SHAPE), raw_shape=[NROWS,2], native_row_order='scene -> replica -> frame3..15',
                           inference_batch=256, geometry_batch=64, feature_roundtrip='native15 -> kept9 FP16 -> restore15 FP32; signed faces6:12 exact0',
                           label_access=False, model_inputs=['histories','transforms','length'],
                           centers='0/+3 saved raw authoritative after full-cohort parity maxabs<=1e-5',
                           angles='Ry(absolute_angle) @ original sensor-to-query transform, float64 before float32 geometry; +3 uses verified original payload',
                           durable_cache='deduplicated public geometry mappings only, no durable photon or compact feature cache')
        save(OUT/'prepare_manifest.json', declaration)
        budget.check()
        return budget.finish('COMPLETE', rows_per_split=NROWS, unique_geometry_counts=[dict(split=g['split'], angle=g['angle_degrees'], count=g['unique_geometries']) for g in geometry], cpu_only=True)
    except Exception as error:
        budget.finish('FAILED', error=repr(error), cpu_only=True)
        raise


def verify_prepared(manifest, budget):
    if manifest['plan_sha256'] != sha(OUT/'PLAN.json'):
        raise ValueError('Frozen PLAN changed after preparation')
    for item in manifest['source_files']+manifest['inputs']+manifest['geometry']:
        budget.check()
        path = Path(item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Hash-bound replay source/input changed: '+str(path))


def output_record(split, angle, arm, seed, raw, authoritative):
    path = OUT/'scores'/f'{arm}_seed{seed}_{split}_angle_{ANGLE_NAMES[angle]}.npz'
    if path.exists():
        raise FileExistsError('Preserve replay score output')
    checkpoint_sha = sha(PASS/'models'/f'{arm}_seed{seed}.pt')
    np.savez_compressed(path, raw=raw, arm=np.array(arm), seed=np.array(seed),
                        split=np.array(split), angle_degrees=np.array(angle), original_shape=np.array(SHAPE),
                        checkpoint_sha256=np.array(checkpoint_sha))
    return dict(arm=arm, seed=seed, split=split, angle_degrees=angle, path=str(path),
                shape=list(raw.shape), original_shape=list(SHAPE), sha256=sha(path),
                checkpoint_sha256=checkpoint_sha, authoritative_saved_center=authoritative, label_access=False)


def run():
    budget = Budget('run')
    torch = None
    networks = {}
    geometries = h = le = geo = compact = features = gpu_lengths = None
    outputs, parity = [], []
    try:
        check_plan(budget.plan)
        if (OUT/'scores').exists():
            raise FileExistsError('Prior replay outputs or partials preserved; do not blindly rerun parity')
        (OUT/'scores').mkdir()
        manifest = read(OUT/'prepare_manifest.json')
        verify_prepared(manifest, budget)
        import torch
        import cnh_boundary_token_model as M
        import cnh_counterfactual_train_dev as T
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for absolute-angle replay')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode():
            for seed in SEEDS:
                for arm in ARMS:
                    state = torch.load(PASS/'models'/f'{arm}_seed{seed}.pt', map_location='cpu', weights_only=True)
                    if state['steps'] != 2808:
                        raise ValueError('Checkpoint is not frozen continuation final step')
                    network = M.BoundaryTokenReadout().cuda().eval()
                    network.load_state_dict(state['state_dict'])
                    networks[arm,seed] = network
            def infer_angle(split, angle):
                nonlocal geometries, h, le, geo, compact, features, gpu_lengths
                item = next(g for g in manifest['geometry'] if g['split']==split and g['angle_degrees']==angle)
                with np.load(item['path'], allow_pickle=False) as native:
                    transforms, lengths, index = (native[k] for k in ('transforms','lengths','geometry_index'))
                parts = []
                for start in range(0,len(transforms),64):
                    budget.check()
                    t = torch.as_tensor(transforms[start:start+64], device='cuda')
                    ll = torch.as_tensor(lengths[start:start+64], device='cuda')
                    n = len(t)
                    if n < 64:
                        t = torch.cat((t,t[-1:].expand(64-n,-1,-1,-1)))
                        ll = torch.cat((ll,ll[-1:].expand(64-n)))
                    parts.append(M.feature_geometry(t,ll,'center')[:n])
                geometries = torch.cat(parts)
                del parts
                histories = np.load(DATA/split/'histories.npy', mmap_mode='r', allow_pickle=False)
                native_lengths = np.load(DATA/split/'length.npy', mmap_mode='r', allow_pickle=False)
                gpu_lengths = torch.as_tensor(np.array(native_lengths),device='cuda')
                raw = {key:np.empty((NROWS,2),np.float32) for key in networks}
                for start in range(0,NROWS,256):
                    budget.check()
                    stop = min(start+256,NROWS)
                    h = torch.as_tensor(np.array(histories[start:stop]),device='cuda')
                    le = gpu_lengths[start:stop]
                    # Build short64 batches exactly as the inherited feature pipeline.
                    compact_parts = []
                    for offset in range(0,len(h),64):
                        geo = geometries[torch.as_tensor(index[start+offset:min(start+offset+64,stop)],device='cuda')]
                        compact_parts.append(M.build_features(h[offset:offset+64],geo,le[offset:offset+64])[:,:,:,list(T.KEPT_CHANNELS)].half())
                    compact = torch.cat(compact_parts)
                    del compact_parts
                    if not bool(torch.isfinite(compact).all()):
                        raise ValueError('Nonfinite compact FP16 features')
                    features = T.restore_features(compact)
                    for key, network in networks.items():
                        raw[key][start:stop] = network(features,le).cpu().numpy()
                    if start % (256*20) == 0:
                        print('REPLAY',split,angle,start,'/',NROWS,flush=True)
                if not all(np.isfinite(value).all() for value in raw.values()):
                    raise ValueError('Nonfinite frozen raw response')
                geometries = h = le = geo = compact = features = gpu_lengths = None
                torch.cuda.empty_cache()
                return raw

            # No new-angle scores published until every original center is checked.
            for split in ANGLES:
                for angle in (0,3):
                    raw = infer_angle(split,angle)
                    for (arm,seed), value in raw.items():
                        saved = load_saved(split,arm,seed,angle)
                        difference = np.abs(value.astype(np.float64)-saved.astype(np.float64))
                        row = dict(split=split,angle_degrees=angle,arm=arm,seed=seed,
                                   scalar_count=value.size,max_abs_difference=float(difference.max()),
                                   unequal_scalars=int(np.count_nonzero(difference)), tolerance=1e-5)
                        parity.append(row)
                    save(OUT/'parity_progress.json',dict(status='IN_PROGRESS',checks=parity))
                    # Preserve parity arrays on observed failure, without making a score branch.
                    if any(p['max_abs_difference']>1e-5 for p in parity):
                        np.savez_compressed(OUT/f'failed_parity_{split}_{ANGLE_NAMES[angle]}.npz',
                                            **{f'{arm}_seed{seed}':v for (arm,seed),v in raw.items()})
                        raise ValueError('Full-cohort frozen center score parity failed')
                    del raw
            save(OUT/'parity.json',dict(status='PASS',checks=parity,total_scalar_count=sum(r['scalar_count'] for r in parity),
                                      max_abs_difference=max(r['max_abs_difference'] for r in parity)))
            for split in ANGLES:
                for angle in (0,3):
                    for arm,seed in networks:
                        outputs.append(output_record(split,angle,arm,seed,load_saved(split,arm,seed,angle),True))
            save(OUT/'scores/manifest.json',dict(status='CENTERS_PARITY_PASS',outputs=outputs,
                     arm_names=list(ARMS),seed_ids=list(SEEDS),angles_by_split={k:list(v) for k,v in ANGLES.items()},original_shape=list(SHAPE)))
            for split, angles in ANGLES.items():
                for angle in angles:
                    if angle in (0,3):
                        continue
                    raw = infer_angle(split,angle)
                    for (arm,seed), value in raw.items():
                        outputs.append(output_record(split,angle,arm,seed,value,False))
                    save(OUT/'scores/manifest.json',dict(status='IN_PROGRESS',outputs=outputs,
                             arm_names=list(ARMS),seed_ids=list(SEEDS),angles_by_split={k:list(v) for k,v in ANGLES.items()},original_shape=list(SHAPE)))
                    del raw
        budget.check()
        save(OUT/'scores/manifest.json',dict(status='COMPLETE',outputs=outputs,arm_names=list(ARMS),seed_ids=list(SEEDS),
                 angles_by_split={k:list(v) for k,v in ANGLES.items()},original_shape=list(SHAPE),
                 parity_path=str(OUT/'parity.json'),prepared_manifest_sha256=sha(OUT/'prepare_manifest.json')))
        return budget.finish('COMPLETE', outputs=len(outputs), native_feature_rows=NROWS*7,
                             total_replay_score_scalars=NROWS*2*6*7,parity_checks=len(parity),label_access=False)
    except Exception as error:
        budget.finish('FAILED',error=repr(error), completed_outputs=len(outputs),parity_checks=len(parity))
        raise
    finally:
        networks.clear()
        network = state = None
        geometries = h = le = geo = compact = features = gpu_lengths = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare','run'), required=True)
    args = parser.parse_args()
    print(json.dumps(prepare() if args.stage=='prepare' else run(),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
