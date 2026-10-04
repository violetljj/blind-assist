"""Pilot observations, evaluator truth and lazy input caches remain separate.

Generation and materialisation require the frozen pilot PLAN. No automatic run
on import; voxel materialisation is an explicitly requested CUDA stage.
"""
import argparse
import copy
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import cnh_displacement_ceiling as D
import cnh_displacement_ceiling_render as R
import cnh_near_range as NR
import cnh_margin_confirm as MC
import cnh_proposal_attribution_scenes as S
import cnh_sequence_observed_geometry as GEOM
import cnh_structure_space as SS

OUT = SS.WORK/'cnh-temporal-readout-20261004'
FRAMES = np.arange(3, 16)
DELTAS = D.INTRUSION_CM.copy()
SCENE_SEED = 2026100405
PHOTON_SEED = 2026100406
POSE_SEED = 2026100407


def common():
    import cnh_temporal_readout as C
    return C


def frozen():
    c = common()
    p = c.load_plan()
    c.check_budget()
    required = dict(scene_seed_prefix=SCENE_SEED, photon_seed_prefix=PHOTON_SEED,
                    noise_seed_prefix=POSE_SEED, train_K=2, eval_K=4,
                    frames=FRAMES.tolist(), eval_delta_cm=DELTAS.tolist())
    if any(p.get(k) != v for k,v in required.items()):
        raise ValueError('Data seeds, photon replicas or frame/delta cohort differ from PLAN')
    return p, c.plan_sha()


def categories(boxes, travel):
    tri = np.concatenate([S.box_mesh(b['lo'], b['hi']) for b in boxes])
    return np.asarray([[GEOM.surface_category((tri-p[:3, 3])@p[:3, :3], q)
                        for q in (0, 1)] for p in travel])


def scenes(unit, index, split):
    """Fixed delta, independently redrawn nuisance if actual boxes intersect."""
    from cnh_expected_separability import target_at
    rng = np.random.default_rng([SCENE_SEED, int(unit)])
    delta = np.asarray([rng.uniform(-25., 10.)]) if split == 'train' else DELTAS
    ref = S.make_scenes(unit)[0]
    assert ref['mode'] in (0, 1)
    group, context = (index//2) % 2, 'none' if (index//4) % 2 == 0 else 'panel'
    rejected = 0
    while True:
        target = NR._target(rng, -.01, group)
        dz = .65-target['z']
        target['box']['lo'][2] += dz
        target['box']['hi'][2] += dz
        target['z'] = .65
        if context == 'none':
            base = NR._none_scene(S, unit, 0, ref, target)
        else:
            shape = dict(SS.draw_shape(rng), same_side=1, gap=float(rng.uniform(.34, .40)))
            base = SS.panel_scene(unit, 0, ref, target, shape, float(rng.uniform(SS.EDGES[0], SS.EDGES[5])))
        variants = []
        collision = False
        for penetration in delta:
            sc = copy.deepcopy(base)
            sc['boxes'][0] = target_at(target, -float(penetration)/100)
            for b in sc['boxes'][1:]:
                overlap = np.minimum(sc['boxes'][0]['hi'], b['hi'])-np.maximum(sc['boxes'][0]['lo'], b['lo'])
                collision |= bool((overlap > 0).all())
            variants.append(sc)
        if not collision:
            return variants, group, context, delta, rejected
        rejected += 1
        if rejected > 1000:
            raise RuntimeError('Nuisance rejection exhausted; do not change delta')


def generate_one(args):
    split, unit, index = args
    p, digest = frozen()
    start = time.monotonic()
    obs_path = OUT/'observations'/split/f'unit{unit}.npz'
    truth_path = OUT/'truth'/split/f'unit{unit}.json'
    receipt_path = obs_path.with_suffix('.json')
    if receipt_path.exists():
        r = json.loads(receipt_path.read_text())
        assert r['plan_sha256'] == digest
        assert all(common().sha(OUT/k) == v for k,v in r['outputs'].items())
        return r
    variants, group, context, delta, rejected = scenes(unit, index, split)
    rendered = [R.expected(sc) for sc in variants]
    expected = np.stack([r['expectation'] for r in rendered])
    ambient = rendered[0]['ambient']
    K = 2 if split == 'train' else 4
    hist = np.empty((len(variants), K, *expected.shape[1:]), np.int32)
    for v in range(len(variants)):
        for k in range(K):
            seed = int(np.random.SeedSequence([PHOTON_SEED, unit, v, k]).generate_state(1)[0])
            hist[v,k] = R.sample(expected[v], ambient, seed)[0]
    sensor, travel = variants[0]['poses'], variants[0]['travel']
    from cnh_track_a_readout import noisy_poses
    noisy = np.stack([noisy_poses(sensor, int(np.random.SeedSequence([POSE_SEED, unit, k]).generate_state(1)[0]), dt=.2)
                      for k in range(K)])
    cats = [categories(sc['boxes'], travel).tolist() for sc in variants]
    truth = dict(unit=unit, group=group, mode=unit%3, context=context,
                 front_range_m=(.65-travel[:,2,3]).tolist(), intrusion_cm=delta.tolist(),
                 boxes=[sc['boxes'] for sc in variants], categories=cats,
                 nuisance_rejections=rejected, fixed_delta_during_rejection=True)
    for path in (obs_path, truth_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(obs_path, hist=hist, ambient=ambient, sensor=sensor, travel=travel, noisy=noisy)
    common().save(truth_path, truth)
    paths = [obs_path, truth_path]
    if split != 'train':
        template_path = OUT/'templates'/split/f'unit{unit}.npz'
        template_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(template_path, expected=expected,
                            object_id=np.stack([r['object_id'] for r in rendered]).astype(np.int8))
        paths.append(template_path)
    r = dict(status='COMPLETE', unit=unit, split=split, plan_sha256=digest,
             K=K, nuisance_rejections=rejected, elapsed_s=time.monotonic()-start,
             outputs={str(x.relative_to(OUT)).replace('\\','/'):common().sha(x) for x in paths})
    common().save(receipt_path, r)
    return r


def generate(split, workers=4):
    p, digest = frozen()
    units = p['train_units'] if split == 'train' else p['eval_units'] if split == 'evaluation' else [p['smoke_unit']]
    args = [(split, int(u), i) for i,u in enumerate(units)]
    start = time.monotonic()
    if workers == 1:
        receipts = [generate_one(a) for a in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            receipts = []
            for r in pool.map(generate_one, args, chunksize=1):
                receipts.append(r)
                if len(receipts)%24 == 0 or len(receipts)==len(units):
                    print('generate', split, len(receipts), '/', len(units), round(time.monotonic()-start, 1), flush=True)
    common().save(OUT/f'generation_{split}.json', dict(status='COMPLETE', plan_sha256=digest,
        units=units, elapsed_s=time.monotonic()-start, nuisance_rejections=sum(r['nuisance_rejections'] for r in receipts),
        nuisance_draws=len(units)+sum(r['nuisance_rejections'] for r in receipts),
        source_sha256=common().sha(Path(__file__))))


def normalized_z(hist, ambient):
    """Match frozen CUDA float32 arithmetic then float16 cache exactly."""
    bias = np.load(D.BIAS).astype(np.float32)
    numerator = np.asarray(hist, np.float32)-bias
    denominator = np.sqrt(np.maximum(16*np.asarray(ambient, np.float32)[...,None]+np.maximum(bias, 0), 1e-9))
    return (numerator/denominator).astype(np.float16)


def natural_sequences(split, units):
    import cnh_cvr_pilot as CP
    is_train = split == 'train'
    folder = NR.OUT/'features'/'train' if is_train else MC.OUT/'features'/('calib' if split=='calibration' else 'evaluation')
    for unit in units:
        with np.load(folder/f'unit{unit}.npz', allow_pickle=False) as d:
            z, ids, frame = d['z1'], d['scene'], d['frame']
        scenes_ = NR.scenes_for(unit) if is_train else MC.scenes_for(unit)
        for sc in scenes_:
            config = sc['config']
            sel = np.flatnonzero(ids == config)
            assert np.array_equal(frame[sel], np.arange(16)) and z.dtype == np.float16
            sensor, travel, noisy = CP.motion_metadata(unit, config)
            assert np.allclose(sensor, sc['poses'], atol=1e-9) and np.allclose(travel, sc['travel'], atol=1e-9)
            cats = categories(sc['boxes'], travel)
            ambient = np.full((16,8,8), S.nominal_parameters()[0].ambient_counts, np.float32)
            front = np.asarray([GEOM.target_front(GEOM.corners(sc['boxes'][0]), t) for t in travel])
            yield dict(unit=unit, config=config, variant=0, replica=0, domain=0, group=sc['group'],
                       z=z[sel], ambient=ambient, sensor=sensor, travel=travel, noisy=noisy,
                       categories=cats, front=front)


def fresh_sequences(split, units):
    folder = 'train' if split=='train' else 'smoke' if split=='smoke' else 'evaluation'
    for unit in units:
        with np.load(OUT/'observations'/folder/f'unit{unit}.npz', allow_pickle=False) as d:
            hist, ambient, sensor, travel, noisy = (d[k] for k in ('hist','ambient','sensor','travel','noisy'))
        truth = json.loads((OUT/'truth'/folder/f'unit{unit}.json').read_text())
        z = normalized_z(hist, ambient)
        for v in range(len(hist)):
            for k in range(hist.shape[1]):
                yield dict(unit=unit, config=v, variant=v, replica=k, domain=1, group=truth['group'],
                           z=z[v,k], ambient=ambient, sensor=sensor, travel=travel, noisy=noisy[k],
                           categories=np.asarray(truth['categories'][v]), front=np.asarray(truth['front_range_m']))


def inputs(split):
    import cnh_cvr_pilot as CP
    p, digest = frozen()
    start = time.monotonic()
    fresh = p['train_units'] if split=='train' else p['eval_units'] if split=='fresh_evaluation' else [p['smoke_unit']] if split=='smoke' else []
    natural = p['natural_train_units'] if split=='train' else p['natural_calibration_units'] if split=='calibration' else p['natural_evaluation_units'] if split=='evaluation' else []
    n = (len(natural)*(22 if split=='train' else 40) + len(fresh)*(2 if split=='train' else 7*4))*len(FRAMES)
    folder = OUT/'inputs'/split
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'receipt.json').exists():
        r=json.loads((folder/'receipt.json').read_text()); assert r['plan_sha256']==digest and r['samples']==n
        return
    arrays = dict(histories=np.lib.format.open_memmap(folder/'histories.npy', mode='w+', dtype=np.float16, shape=(n,8,8,8,16)),
                  ambient=np.lib.format.open_memmap(folder/'ambient.npy', mode='w+', dtype=np.float32, shape=(n,8,8,8)),
                  transforms=np.lib.format.open_memmap(folder/'transforms.npy', mode='w+', dtype=np.float64, shape=(n,8,4,4)),
                  length=np.lib.format.open_memmap(folder/'length.npy', mode='w+', dtype=np.int64, shape=(n,)),
                  labels=np.lib.format.open_memmap(folder/'labels.npy', mode='w+', dtype=np.float32, shape=(n,2)),
                  mask=np.lib.format.open_memmap(folder/'mask.npy', mode='w+', dtype=np.float32, shape=(n,2)),
                  weights=np.lib.format.open_memmap(folder/'weights.npy', mode='w+', dtype=np.float32, shape=(n,2)))
    metadata = {k:[] for k in ('unit','config','variant','replica','frame','domain','group','front','category')}
    offset = 0
    from itertools import chain
    for sc in chain(natural_sequences(split, natural), fresh_sequences(split, fresh)):
        for f in FRAMES:
            begin=max(0,int(f)-7); length=int(f)-begin+1; pad=8-length
            arrays['histories'][offset] = 0
            arrays['histories'][offset,pad:] = sc['z'][begin:f+1]
            arrays['ambient'][offset] = 0
            arrays['ambient'][offset,pad:] = sc['ambient'][begin:f+1]
            arrays['transforms'][offset] = np.eye(4)
            arrays['transforms'][offset,pad:] = CP.relative_transforms(sc['sensor'],sc['travel'],sc['noisy'],int(f))
            arrays['length'][offset] = length
            cats = sc['categories'][f]
            y = np.asarray([c.startswith('contact') for c in cats],np.float32)
            mask = np.asarray([c!='pass0-10cm' for c in cats],np.float32)
            arrays['labels'][offset],arrays['mask'][offset] = y,mask
            far = p['far_weight'] if 1.6 <= sc['front'][f] < 2.6 else 1.
            arrays['weights'][offset] = mask*far
            for key in metadata:
                metadata[key].append(cats.tolist() if key=='category' else int(f) if key=='frame' else float(sc['front'][f]) if key=='front' else sc[key])
            offset += 1
        if offset%3380 == 0:
            print('inputs',split,offset,'/',n,round(time.monotonic()-start,1),flush=True)
    assert offset==n
    md={k:np.asarray(v) for k,v in metadata.items()}
    raw_sums={}
    for domain in np.unique(md['domain']):
        sel=md['domain']==domain
        total=float(arrays['weights'][sel].sum(dtype=np.float64))
        assert total>0
        target=n/2 if split=='train' else n
        arrays['weights'][sel] *= np.float32(target/total)
        raw_sums[str(domain)]=total
    for a in arrays.values():a.flush()
    del arrays
    np.savez_compressed(folder/'rows.npz',**md)
    common().save(folder/'receipt.json',dict(status='COMPLETE',split=split,samples=n,plan_sha256=digest,
        elapsed_s=time.monotonic()-start,raw_loss_mass=raw_sums,source_sha256=common().sha(Path(__file__)),
        weight_rule='far*valid; each train domain sums N/2, no class balancing; nontraining one domain sums N',
        inputs_only='histories,ambient,transforms,length; rows/labels/mask/weights never network input'))


def native_voxel_index(split):
    roots = [NR.OUT/'data'/'train'] if split=='train' else [MC.OUT/'data'/n for n in
              (('calib','calib_early') if split=='calibration' else ('evaluation','evaluation_early'))] if split in ('calibration','evaluation') else []
    arrays, lookup = [], {}
    for root in roots:
        for path in sorted(root.glob('features_c*.npy')):
            a=np.load(path,mmap_mode='r'); ai=len(arrays);arrays.append(a)
            with np.load(path.with_name(path.name.replace('features_','metadata_').replace('.npy','.npz'))) as m:
                assert len(a)==len(m['unit']) and a.shape[1:]==(3,24,17,33) and a.dtype==np.float16
                for j,(u,c,f) in enumerate(zip(m['unit'],m['config'],m['frame'])):
                    key=(int(u),int(c),int(f)); assert key not in lookup
                    lookup[key]=(ai,j)
    return arrays,lookup


def voxels(split):
    import torch
    from cnh_cvr_v2_materialize import BatchedProjector
    p,digest=frozen();folder=OUT/'inputs'/split
    if (folder/'voxels_receipt.json').exists():
        assert json.loads((folder/'voxels_receipt.json').read_text())['plan_sha256']==digest
        return
    start=time.monotonic();torch.set_num_threads(2)
    assert torch.cuda.is_available()
    with np.load(folder/'rows.npz') as d: rows={k:d[k] for k in ('unit','config','frame','domain')}
    z=np.load(folder/'histories.npy',mmap_mode='r');ts=np.load(folder/'transforms.npy',mmap_mode='r');lens=np.load(folder/'length.npy',mmap_mode='r')
    dest=np.lib.format.open_memmap(folder/'voxels.npy',mode='w+',dtype=np.float16,shape=(len(z),3,24,17,33))
    native,lookup=native_voxel_index(split);projector=BatchedProjector();checked=0;reused=0
    for i in range(len(z)):
        key=tuple(int(rows[k][i]) for k in ('unit','config','frame'))
        if rows['domain'][i]==0:
            ai,j=lookup[key]; dest[i]=native[ai][j];reused+=1
            if checked<2:
                L=int(lens[i]); fresh=projector.sequence(z[i,-L:].copy(),ts[i,-L:]).cpu().numpy().astype(np.float16)
                assert np.array_equal(fresh,dest[i]),('Native cache projector parity',key,float(np.max(np.abs(fresh-dest[i]))))
                checked+=1
        else:
            L=int(lens[i]); dest[i]=projector.sequence(z[i,-L:].copy(),ts[i,-L:]).cpu().numpy().astype(np.float16)
        if (i+1)%2600==0:
            common().check_budget(); print('voxels',split,i+1,'/',len(z),round(time.monotonic()-start,1),flush=True)
    dest.flush(); del dest,projector
    torch.cuda.empty_cache()
    common().save(folder/'voxels_receipt.json',dict(status='COMPLETE',split=split,plan_sha256=digest,
        samples=len(z),reused_native_rows=reused,native_parity_rows=checked,elapsed_s=time.monotonic()-start,
        source_sha256=common().sha(Path(__file__))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=('generate','inputs','voxels'))
    parser.add_argument('--split',required=True,choices=('train','evaluation','smoke','calibration','fresh_evaluation'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='generate':
        assert args.split in ('train','evaluation','smoke')
        generate(args.split,args.workers)
    else:
        {'inputs':inputs,'voxels':voxels}[args.stage](args.split)
