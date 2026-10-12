"""Fresh evaluation inputs and reference-only reuse for the second readout pilot.

Observed inputs never consult truth. Generation saves evaluator truth separately.
Seven lateral variants share estimated poses, allowing one CUDA projection batch
per replica/anchor instead of repeating geometry seven times.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import time

import numpy as np
import cnh_temporal_readout_data as PREVIOUS
import cnh_displacement_ceiling as D
import cnh_displacement_ceiling_render as R
import cnh_cvr_pilot as CP
import cnh_readout_pilot2 as C

OUT, OLD = C.OUT, C.OLD
EVAL_UNITS, SMOKE_UNIT = C.EVAL_UNITS, C.SMOKE_UNIT
FRAMES = np.arange(3, 16)
DELTAS = D.INTRUSION_CM.copy()
SCENE_SEED, PHOTON_SEED, POSE_SEED = 2026100405, 2026100406, 2026100407


def frozen():
    p = C.load_plan()
    required = dict(eval_units=EVAL_UNITS, smoke_unit=SMOKE_UNIT,
                    scene_seed_prefix=SCENE_SEED, photon_seed_prefix=PHOTON_SEED,
                    noise_seed_prefix=POSE_SEED, frames=FRAMES.tolist(),
                    eval_delta_cm=DELTAS.tolist(), eval_K=4)
    if any(p.get(k) != v for k,v in required.items()):
        raise ValueError('Second pilot evaluation cohort/seeds differ from frozen PLAN')
    return p, C.plan_sha()


def input_folder(split):
    """Path resolver: existing train/natural arrays remain owned by pilot1."""
    if split in ('train', 'calibration', 'evaluation'):
        return OLD/'inputs'/split
    if split in ('fresh_evaluation', 'smoke'):
        return OUT/'inputs'/split
    raise ValueError('Unknown input split '+split)


def references():
    _, digest = frozen()
    bindings = {}
    for split in ('train', 'calibration', 'evaluation'):
        folder = input_folder(split)
        receipt = folder/'receipt.json'
        voxel_receipt = folder/'voxels_receipt.json'
        assert receipt.exists() and voxel_receipt.exists()
        assert json.loads(receipt.read_text())['status'] == 'COMPLETE'
        assert json.loads(voxel_receipt.read_text())['status'] == 'COMPLETE'
        bindings[split] = dict(input_folder=str(folder), reuse='read-only direct references; no copies or junctions',
            receipt_sha256=C.sha(receipt), voxel_receipt_sha256=C.sha(voxel_receipt),
            files={name:str(folder/f'{name}.npy') for name in
                   ('histories','ambient','transforms','length','labels','mask','weights','voxels')},
            rows_path=str(folder/'rows.npz'))
    C.save(OUT/'input_references.json', dict(plan_sha256=digest, old_plan_sha256=C.sha(OLD/'PLAN.json'),
        bindings=bindings, old_observations_train=str(OLD/'observations'/'train'),
        old_truth_train=str(OLD/'truth'/'train')))
    return bindings


def id_audit():
    """Check numeric reservations and exact filenames without opening scores."""
    wanted=set(EVAL_UNITS+[SMOKE_UNIT]);hits=[];roots=0;plans=0
    for folder in OUT.parent.glob('cnh*'):
        if folder == OUT or not folder.is_dir():
            continue
        roots += 1
        for name in ('PLAN.json','plan.json','request.json'):
            path=folder/name
            if path.exists():
                plans += 1
                ids={int(n) for n in re.findall(r'(?<![\d.])\d+(?![\d.])',path.read_text(encoding='utf8'))}
                if wanted&ids:
                    hits.append(dict(path=str(path),ids=sorted(wanted&ids),kind='numeric reservation'))
        for path in folder.rglob('unit231*.*'):
            match=re.fullmatch(r'unit(\d+)',path.stem)
            if match and int(match[1]) in wanted:
                hits.append(dict(path=str(path),ids=[int(match[1])],kind='exact unit filename'))
    result=dict(status='PASS' if not hits else 'COLLISION', roots=roots,plans=plans,
                evaluation_units=EVAL_UNITS,smoke_unit=SMOKE_UNIT,hits=hits,
                exclusion='This task root is reserved separately; score contents not opened')
    C.save(OUT/'fresh_id_audit.json',result)
    if hits:
        raise ValueError('Fresh second-pilot IDs are already reserved: '+str(hits))
    print('Fresh ID audit PASS',len(EVAL_UNITS),'evaluation units; smoke',SMOKE_UNIT,flush=True)
    return result


def generate_one(args):
    split,unit,index=args
    _,digest=frozen();started=time.monotonic()
    observation=OUT/'observations'/split/f'unit{unit}.npz'
    truth_path=OUT/'truth'/split/f'unit{unit}.json'
    template=OUT/'templates'/split/f'unit{unit}.npz'
    receipt=observation.with_suffix('.json')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        assert old['plan_sha256']==digest
        assert all(C.sha(OUT/k)==v for k,v in old['outputs'].items())
        return old
    # Reuse a pure scene constructor; its OUT/PLAN and generators are untouched.
    variants,group,context,delta,rejected=PREVIOUS.scenes(unit,index,'evaluation')
    rendered=[R.expected(sc) for sc in variants]
    expected=np.stack([r['expectation'] for r in rendered])
    ambient=rendered[0]['ambient']
    assert all(np.array_equal(ambient,r['ambient']) for r in rendered)
    hist=np.empty((7,4,*expected.shape[1:]),np.int32)
    for variant in range(7):
        for replica in range(4):
            seed=int(np.random.SeedSequence([PHOTON_SEED,unit,variant,replica]).generate_state(1)[0])
            hist[variant,replica]=R.sample(expected[variant],ambient,seed)[0]
    sensor,travel=variants[0]['poses'],variants[0]['travel']
    from cnh_track_a_readout import noisy_poses
    noisy=np.stack([noisy_poses(sensor,int(np.random.SeedSequence([POSE_SEED,unit,k]).generate_state(1)[0]),dt=.2)
                    for k in range(4)])
    truth=dict(unit=unit,group=group,mode=unit%3,context=context,
        front_range_m=(.65-travel[:,2,3]).tolist(),intrusion_cm=delta.tolist(),
        boxes=[sc['boxes'] for sc in variants],
        categories=[PREVIOUS.categories(sc['boxes'],travel).tolist() for sc in variants],
        nuisance_rejections=rejected,fixed_delta_during_rejection=True,
        role='Fresh random evaluation Development; not used for training or recipe selection')
    for path in (observation,truth_path,template):path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(observation,hist=hist,ambient=ambient,sensor=sensor,travel=travel,noisy=noisy)
    C.save(truth_path,truth)
    np.savez_compressed(template,expected=expected,
                        object_id=np.stack([r['object_id'] for r in rendered]).astype(np.int8))
    paths=(observation,truth_path,template)
    result=dict(status='COMPLETE',split=split,unit=unit,plan_sha256=digest,
        seconds=time.monotonic()-started,nuisance_rejections=rejected,
        outputs={str(p.relative_to(OUT)).replace('\\','/'):C.sha(p) for p in paths})
    C.save(receipt,result)
    return result


def generate(split,workers=4):
    _,digest=frozen()
    assert split in ('evaluation','smoke')
    units=EVAL_UNITS if split=='evaluation' else [SMOKE_UNIT]
    jobs=[(split,int(unit),i) for i,unit in enumerate(units)]
    started=time.monotonic()
    if workers==1:
        records=[generate_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records=[]
            for result in pool.map(generate_one,jobs,chunksize=1):
                records.append(result)
                if len(records)%12==0 or len(records)==len(units):
                    print('generate',split,len(records),'/',len(units),round(time.monotonic()-started,1),flush=True)
    result=dict(status='COMPLETE',split=split,units=units,plan_sha256=digest,
                elapsed_s=time.monotonic()-started,nuisance_rejections=sum(r['nuisance_rejections'] for r in records),
                data_source_sha256=C.sha(Path(__file__)),old_scene_source_sha256=C.sha(PREVIOUS.__file__))
    C.save(OUT/f'generation_{split}.json',result)
    return result


def inputs(split):
    """Only observation files enter deployment input construction."""
    _,digest=frozen();assert split in ('fresh_evaluation','smoke')
    rawsplit='evaluation' if split=='fresh_evaluation' else 'smoke'
    units=EVAL_UNITS if split=='fresh_evaluation' else [SMOKE_UNIT]
    folder=input_folder(split);folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt.json'
    if receipt.exists():
        prior=json.loads(receipt.read_text());assert prior['plan_sha256']==digest
        return prior
    started=time.monotonic();n=len(units)*7*4*13
    arrays={name:np.lib.format.open_memmap(folder/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)
            for name,dtype,shape in [('histories',np.float16,(n,8,8,8,16)),('ambient',np.float32,(n,8,8,8)),
                                    ('transforms',np.float64,(n,8,4,4)),('length',np.int64,(n,))]}
    # Routing identifiers only. Evaluator-only group/range/categories written in
    # a separate second pass, after input arrays are completed and closed.
    rows={key:[] for key in ('unit','config','variant','replica','frame','domain')}
    observations={};offset=0
    for unit in units:
        path=OUT/'observations'/rawsplit/f'unit{unit}.npz'
        with np.load(path,allow_pickle=False) as obs:
            hist,ambient,sensor,travel,noisy=(obs[k] for k in ('hist','ambient','sensor','travel','noisy'))
        assert hist.shape==(7,4,16,8,8,16)
        z=PREVIOUS.normalized_z(hist,ambient)
        observations[str(path)]=C.sha(path)
        for variant in range(7):
            for replica in range(4):
                for frame in FRAMES:
                    first=max(0,int(frame)-7);L=int(frame)-first+1;pad=8-L
                    arrays['histories'][offset]=0;arrays['histories'][offset,pad:]=z[variant,replica,first:frame+1]
                    arrays['ambient'][offset]=0;arrays['ambient'][offset,pad:]=ambient[first:frame+1]
                    arrays['transforms'][offset]=np.eye(4)
                    arrays['transforms'][offset,pad:]=CP.relative_transforms(sensor,travel,noisy[replica],int(frame))
                    arrays['length'][offset]=L
                    for key,value in dict(unit=unit,config=variant,variant=variant,replica=replica,
                                          frame=int(frame),domain=1).items():rows[key].append(value)
                    offset+=1
        if offset%4368==0 or offset==n:
            print('inputs',split,offset,'/',n,round(time.monotonic()-started,1),flush=True)
    assert offset==n
    for a in arrays.values():a.flush()
    del arrays
    rows={k:np.asarray(v) for k,v in rows.items()}
    evaluator_rows={key:[] for key in ('group','front','category')}
    truth_sources={}
    for unit in units:
        path=OUT/'truth'/rawsplit/f'unit{unit}.json'
        truth=json.loads(path.read_text());truth_sources[str(path)]=C.sha(path)
        for variant in range(7):
            for replica in range(4):
                for frame in FRAMES:
                    evaluator_rows['group'].append(truth['group'])
                    evaluator_rows['front'].append(truth['front_range_m'][int(frame)])
                    evaluator_rows['category'].append(truth['categories'][variant][int(frame)])
    rows.update({k:np.asarray(v) for k,v in evaluator_rows.items()})
    np.savez_compressed(folder/'rows.npz',**rows)
    if split=='smoke':
        # Engineering gradient checks alone receive separate supervision. Fresh
        # scientific evaluation never gets these training-loss input files.
        labels=np.asarray([[str(c).startswith('contact') for c in pair] for pair in rows['category']],np.float32)
        mask=np.asarray(rows['category']!='pass0-10cm',np.float32)
        p=C.load_plan()
        far=np.where((rows['front']>=1.6)&(rows['front']<2.6),p['far_weight'],1.).astype(np.float32)
        weights=mask*far[:,None]
        weights *= np.float32(n/weights.sum(dtype=np.float64))
        for name,array in [('labels',labels),('mask',mask),('weights',weights)]:
            np.save(folder/f'{name}.npy',array)
    result=dict(status='COMPLETE',split=split,samples=n,plan_sha256=digest,elapsed_s=time.monotonic()-started,
                source_sha256=C.sha(Path(__file__)),observation_sha256=observations,
                evaluator_metadata_sha256=truth_sources,
                inputs_only='histories,ambient,transforms,length generated from observations before opening truth; no labels/masks in fresh evaluation input folder; smoke labels separate')
    C.save(receipt,result)
    return result


def voxels(split):
    """Batch all seven variants sharing one replica/anchor geometric operator."""
    import torch
    from cnh_cvr_v2_materialize import BatchedProjector
    _,digest=frozen();assert split in ('fresh_evaluation','smoke')
    folder=input_folder(split);receipt=folder/'voxels_receipt.json'
    if receipt.exists():
        prior=json.loads(receipt.read_text());assert prior['plan_sha256']==digest
        return prior
    started=time.monotonic();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    hist=np.load(folder/'histories.npy',mmap_mode='r');trans=np.load(folder/'transforms.npy',mmap_mode='r')
    length=np.load(folder/'length.npy',mmap_mode='r')
    with np.load(folder/'rows.npz') as metadata:
        # Only routing identifiers; evaluator group/range/categories not read.
        rows={k:metadata[k] for k in ('unit','variant','replica','frame')}
    units=EVAL_UNITS if split=='fresh_evaluation' else [SMOKE_UNIT]
    assert len(hist)==len(units)*364
    dest=np.lib.format.open_memmap(folder/'voxels.npy',mode='w+',dtype=np.float16,shape=(len(hist),3,24,17,33))
    projector=BatchedProjector();checks=[];projection_calls=0
    try:
        for ui,unit in enumerate(units):
            base=ui*364
            for replica in range(4):
                for fi,frame in enumerate(FRAMES):
                    ids=np.asarray([base+(variant*4+replica)*13+fi for variant in range(7)])
                    assert np.array_equal(rows['unit'][ids],np.full(7,unit))
                    assert np.array_equal(rows['variant'][ids],np.arange(7))
                    assert np.array_equal(rows['replica'][ids],np.full(7,replica))
                    assert np.array_equal(rows['frame'][ids],np.full(7,frame))
                    L=int(length[ids[0]])
                    assert np.array_equal(length[ids],np.full(7,L))
                    t=np.asarray(trans[ids[0],-L:])
                    assert np.array_equal(trans[ids,-L:],np.broadcast_to(t,(7,*t.shape)))
                    native=np.array(hist[ids,-L:],copy=True)
                    result=D.project_many(projector,native,t)
                    if ui==0 and replica==0 and int(frame) in (3,15):
                        serial=projector.sequence(native[0],t)
                        error=float((serial-result[0]).abs().max())
                        assert torch.equal(serial,result[0]),('Batched/serial projector parity',int(frame),error)
                        checks.append(dict(frame=int(frame),max_absolute_error=error,bitwise_equal=True))
                    assert bool(torch.isfinite(result).all())
                    dest[ids]=result.half().cpu().numpy();projection_calls+=1
            dest.flush()
            if (ui+1)%12==0 or ui+1==len(units):
                print('voxels',split,ui+1,'/',len(units),round(time.monotonic()-started,1),flush=True)
        dest.flush()
    finally:
        del dest,projector
        torch.cuda.empty_cache()
    result=dict(status='COMPLETE',split=split,samples=len(hist),plan_sha256=digest,
                elapsed_s=time.monotonic()-started,projection_calls=projection_calls,
                geometric_reuse_factor=7,projector_checks=checks,source_sha256=C.sha(Path(__file__)),
                operator_source_sha256=C.sha(D.__file__),inputs_receipt_sha256=C.sha(folder/'receipt.json'))
    C.save(receipt,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=('id-audit','refs','generate','inputs','voxels'))
    parser.add_argument('--split',default='evaluation',choices=('evaluation','smoke','fresh_evaluation'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='id-audit':id_audit()
    elif args.stage=='refs':references()
    elif args.stage=='generate':generate(args.split,args.workers)
    else:{'inputs':inputs,'voxels':voxels}[args.stage](args.split)
