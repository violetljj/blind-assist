"""Materialize sanitized City RGB and separate supervision without training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'research/active/dtr-r0/nearfield'))
from city_data import pool_support


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def checked(root,relative,digest):
    path=(root/relative).resolve(strict=True)
    if not path.is_relative_to(root) or sha(path)!=digest:
        raise ValueError('Input path/hash mismatch: '+relative)
    return path


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def build(capture,split_path,output):
    start=time.perf_counter()
    root=capture.resolve(strict=True); out=output.resolve()
    artifacts=(REPO/'artifacts.local').resolve()
    if not root.is_relative_to(artifacts) or not out.is_relative_to(artifacts) or out==artifacts or out.exists():
        raise ValueError('Artifact-owned input and fresh output required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for native mask pooling')
    split=read(split_path)
    if split['schema']!='city-region-development-split-v1':
        raise ValueError('Unsupported split contract')
    for rel,digest in split['input_hashes'].items():
        checked(root,rel,digest)
    world=read(root/'world-verification.json'); source=read(root/'evaluator/spec.json')
    if world.get('status')!='PASS' or world.get('mask_order')!=['BODY','HEAD'] or world.get('query_range_m')!=3.:
        raise ValueError('Expected accepted BODY/HEAD 3m visible-support contract')
    rows=split['frames']; ids=[r['sample_index'] for r in rows]
    if sorted(ids)!=list(range(len(world['rows']))):
        raise ValueError('Split must cover every accepted frame exactly once')
    grouped={}
    for r in rows:
        i=r['sample_index']; role=r['split']
        if role not in ('train','val','test') or r['group_id']!=source['cases'][i]['group_id']:
            raise ValueError('Invalid group or partition')
        if r['region']!=r['group_id'].split('__')[0] or split['assignment'].get(r['region'])!=role:
            raise ValueError('Declared region split changed')
        if grouped.setdefault(r['group_id'],role)!=role:
            raise ValueError('Group crosses partitions')
    out.mkdir(parents=True)
    manifest=dict(schema='city-training-cache-v1',status='BUILDING',partitions={},
        input_contract='Only rgb is passed to DecoupledModel; uint8 BOX resize -> float [0,1]; model normalizes',
        supervision_contract='All visible scene surfaces; BODY/HEAD 3m near unchanged; pooled -1/0/1 support',
        support_pooling='20x20 positive wins; otherwise any UNKNOWN -> -1; else 0',
        approaching='UNKNOWN; not trained',distance='Evaluator-only static states; no WARNING/DANGER model head',
        source_split_sha256=sha(split_path),source_hashes=split['input_hashes'])
    try:
        for role in ('train','val','test'):
            selected=[r for r in rows if r['split']==role]
            if not selected:
                continue
            dest=out/'model'/role; dest.mkdir(parents=True)
            labeldir=out/('evaluator' if role=='test' else 'supervision')/role
            labeldir.mkdir(parents=True)
            n=len(selected)
            rgb=np.lib.format.open_memmap(dest/'rgb.npy',mode='w+',dtype=np.uint8,shape=(n,144,256,3))
            near=np.lib.format.open_memmap(labeldir/'near.npy',mode='w+',dtype=np.float32,shape=(n,2))
            support=np.lib.format.open_memmap(labeldir/'support.npy',mode='w+',dtype=np.int8,shape=(n,2,18,32))
            distance=[]
            for begin in range(0,n,32):
                native=[]
                for offset,r in enumerate(selected[begin:begin+32],begin):
                    i=r['sample_index']; w=world['rows'][i]
                    if w['sample_index']!=i or r['rgb_sha256']!=w['rgb_sha256'] or r['mask_sha256']!=w['mask_sha256'] or r['mask_path']!=w['mask_path']:
                        raise ValueError('World/split row mismatch')
                    image_path=checked(root,r['rgb_path'],r['rgb_sha256'])
                    mask_path=checked(root,r['mask_path'],r['mask_sha256'])
                    with Image.open(image_path) as im:
                        if im.size!=(640,360): raise ValueError('Unexpected RGB dimensions')
                        rgb[offset]=np.asarray(im.convert('RGB').resize((256,144),Image.Resampling.BOX))
                    mask=np.load(mask_path,allow_pickle=False)
                    if mask.dtype!=np.int8 or mask.shape!=(2,360,640): raise ValueError('Native mask shape/type')
                    target=np.asarray(w['body_head_visible_targets'])
                    if not np.array_equal(target,(np.sum(mask==1,axis=(1,2))>=3).astype(int)):
                        raise ValueError('Near label/native support mismatch')
                    near[offset]=target
                    native.append(mask)
                    distance.append(dict(sample_index=i,evidence=w['distance_evidence']))
                support[begin:begin+len(native)]=pool_support(torch.from_numpy(np.stack(native)).cuda()).cpu().numpy()
            for array in (rgb,near,support): array.flush()
            del rgb,near,support
            def record(path): return dict(path=path.relative_to(out).as_posix(),sha256=sha(path))
            sample_ids=[r['sample_index'] for r in selected]
            manifest['partitions'][role]=dict(sample_indices=sample_ids,rgb=record(dest/'rgb.npy'))
            labels=dict(sample_indices=sample_ids,near=record(labeldir/'near.npy'),support=record(labeldir/'support.npy'))
            write(out/('evaluator' if role=='test' else 'supervision')/f'{role}.json',labels)
            write(out/'evaluator'/f'{role}-distance.json',distance)
        torch.cuda.synchronize()
        manifest.update(status='PASS',backend='CUDA',device=torch.cuda.get_device_name(),
            elapsed_s=time.perf_counter()-start,
            code_sha256={p.name:sha(p) for p in (Path(__file__),REPO/'research/active/dtr-r0/nearfield/city_data.py')})
        write(out/'manifest.json',manifest)
        return manifest
    except BaseException as exc:
        write(out/'failure.json',dict(status='FAIL',error=str(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','split','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); r=build(a.capture,a.split,a.output)
    print(json.dumps(dict(status=r['status'],backend=r['backend'],elapsed_s=r['elapsed_s'],
                         frames={k:len(v['sample_indices']) for k,v in r['partitions'].items()})))
