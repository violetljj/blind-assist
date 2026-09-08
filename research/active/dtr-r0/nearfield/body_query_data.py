"""Hash-bound RGB cache plus separate visible-count supervision/evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from body_query_labels import labels
from body_query_model import CALIBRATION
from city_data import pool_support


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def fresh_output(path):
    path=Path(path).resolve()
    root=Path(os.environ.get('BLINDASSIST_ARTIFACTS',Path(__file__).resolve().parents[4]/'artifacts.local')).resolve()
    if path.exists() or path==root or not path.is_relative_to(root):
        raise ValueError('Fresh output under canonical artifacts required: '+str(path))
    path.mkdir(parents=True)
    return path


def checked_array(root,record):
    p=(root/record['path']).resolve()
    if not p.is_relative_to(root.resolve()) or sha(p)!=record['sha256']:
        raise ValueError('Cache path/hash mismatch')
    return np.load(p,allow_pickle=False)


class QueryRGB:
    """Does not open supervision/evaluator, including during inference."""
    def __init__(self,cache,role):
        self.root=Path(cache).resolve();manifest=read(self.root/'manifest.json')
        if manifest['status']!='PASS' or manifest['calibration']!=CALIBRATION:
            raise ValueError('Accepted fixed-camera cache required')
        row=manifest['partitions'][role]
        if not row['rgb']['path'].startswith('model/'):
            raise ValueError('RGB must reside under model/')
        self.rgb=checked_array(self.root,row['rgb']);self.ids=row['sample_indices']
        if self.rgb.dtype!=np.uint8 or self.rgb.shape!=(len(self.ids),144,256,3):
            raise ValueError('Unexpected RGB shape/dtype')

    def tensor(self,indices,device):
        return torch.from_numpy(self.rgb[indices].copy()).to(device).permute(0,3,1,2).float()/255.


def truth(cache,role,training=False):
    if training and role!='train':raise ValueError('Only TRAIN supervision can enter optimizer')
    root=Path(cache).resolve();folder='supervision' if role=='train' else 'evaluator'
    record=read(root/folder/f'{role}.json')
    arrays={k:checked_array(root,record[k]) for k in ('near','support','counts')}
    return record,arrays


def build(capture,output):
    start=time.perf_counter();root=Path(capture).resolve()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for native geometry materialization')
    world=read(root/'world-verification.json');spec=read(root/'source/spec.json')
    if world['status']!='PASS' or world['query_range_m']!=3.:
        raise ValueError('Native world verification not accepted')
    if len(world['rows'])!=len(spec['cases']):raise ValueError('Capture frame coverage mismatch')
    out=fresh_output(output)
    manifest=dict(schema='body-query-cache-v1',status='BUILDING',calibration=CALIBRATION,
        source_spec_sha256=sha(root/'source/spec.json'),world_verification_sha256=sha(root/'world-verification.json'),
        source_capture=str(root),partitions={},groups={},input='RGB only, fixed calibration',
        label_semantics='Observed native visible-pixel counts capped3; zero is not certified empty space',
        scope='Synthetic same-world Development; source roles fixed before capture; shared assets/background possible')
    write(out/'manifest.json',manifest)
    seen_rgb={}
    try:
        for role in ('train','dev','eval'):
            indices=[i for i,c in enumerate(spec['cases']) if c['source_role']==role.upper()+'_ONLY']
            if not indices:raise ValueError('Empty role '+role)
            rgb=[];near=[];support=[];counts=[];records=[];raw_counts=[]
            for i in indices:
                c=spec['cases'][i];row=world['rows'][i]
                native_path=root/f'evaluator/native/{i:04d}.npy';image_path=root/f'model/sample/{i:04d}.png'
                if sha(native_path)!=row['native_sha256'] or sha(image_path)!=row['rgb_sha256']:
                    raise ValueError('Raw capture hash mismatch')
                if seen_rgb.setdefault(row['rgb_sha256'],role)!=role:
                    raise ValueError('Identical RGB crosses source roles')
                if manifest['groups'].setdefault(c['group_id'],role)!=role:
                    raise ValueError('Parent group crosses split')
                data=labels(torch.from_numpy(np.load(native_path,allow_pickle=False)).cuda(),c['camera'],c['floor_z_m'])
                expected=np.load(root/row['mask_path'],allow_pickle=False)
                if sha(root/row['mask_path'])!=row['mask_sha256'] or not np.array_equal(data['support'].cpu().numpy(),expected):
                    raise ValueError('Independent partition differs from native support')
                if data['near'].tolist()!=row['body_head_visible_targets']:
                    raise ValueError('Near labels changed')
                with Image.open(image_path) as im:
                    rgb.append(np.asarray(im.convert('RGB').resize((256,144),Image.Resampling.BOX)))
                near.append(data['near'].cpu().numpy());counts.append(data['counts'].cpu().numpy())
                raw_counts.append(data['raw_counts'].cpu().tolist())
                support.append(pool_support(data['support'][None]).cpu().numpy()[0])
                condition=c.get('condition',{})
                records.append(dict(sample_index=i,name=c['name'],group_id=c['group_id'],
                    condition=condition.get('query_control',c.get('variant_id',condition.get('desired_relation','UNKNOWN'))),
                    family=condition.get('family','unknown'),site=c.get('source_site_id','unknown'),
                    intended_relation=condition.get('desired_relation'),actual_near=data['near'].tolist(),
                    source_rgb_sha256=row['rgb_sha256'],source_native_sha256=row['native_sha256']))
            def save(relative,array):
                path=out/relative;path.parent.mkdir(parents=True,exist_ok=True)
                np.save(path,np.asarray(array),allow_pickle=False)
                return dict(path=relative,sha256=sha(path))
            manifest['partitions'][role]=dict(rgb=save(f'model/{role}/rgb.npy',np.asarray(rgb,dtype=np.uint8)),sample_indices=indices)
            folder='supervision' if role=='train' else 'evaluator'
            rec=dict(records=records,sample_indices=indices,
                near=save(f'{folder}/{role}/near.npy',np.asarray(near,dtype=np.int64)),
                support=save(f'{folder}/{role}/support.npy',np.asarray(support,dtype=np.int8)),
                counts=save(f'{folder}/{role}/counts.npy',np.asarray(counts,dtype=np.int64)))
            write(out/folder/f'{role}.json',rec)
            # Demo metadata duplicate is evaluator-side, never read by model loader.
            if role=='train':write(out/'evaluator/train.json',rec)
            write(out/f'evaluator/{role}-raw-counts.json',dict(records=records,raw_counts=raw_counts))
            manifest['partitions'][role].update(frames=len(indices),groups=len({r['group_id'] for r in records}),
                near_positives=np.asarray(near).sum(0).tolist(),query_class_counts=np.bincount(np.asarray(counts).flatten(),minlength=4).tolist(),
                unknown_support_fraction=float((np.asarray(support)<0).mean()))
        torch.cuda.synchronize()
        manifest.update(status='PASS',seconds=time.perf_counter()-start,backend='CUDA',device=torch.cuda.get_device_name(),
            source_sha256={p:sha(Path(__file__).with_name(p)) for p in ('body_query_data.py','body_query_labels.py','body_query_model.py')})
        write(out/'manifest.json',manifest)
        return manifest
    except Exception as exc:
        write(out/'failure.json',dict(error=repr(exc)));raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.capture,a.output)))


if __name__=='__main__':main()
