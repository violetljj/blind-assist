"""Build a role-separated 64-frame City cache from completed native evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def build(capture, output, role, admission_path):
    started = time.perf_counter()
    spec = read(capture/'source/spec.json')
    source_role = {'train':'TRAIN_ONLY', 'eval':'EVAL_ONLY'}[role]
    truth = read(capture/'world-verification.json')
    admission = read(admission_path)
    assert admission['status'] == 'PASS'
    assert admission['minimum_train_eval_camera_distance_m'] >= 30
    assert admission['minimum_eval_prior_train_dev_camera_distance_m'] >= 30
    assert truth['mask_order'] == ['BODY','HEAD'] and truth['query_range_m'] == 3.
    assert truth['status'] == read(capture/'completion.json')['status'] == 'PASS'
    assert read(capture/'source-integrity.json')['unchanged']
    release = read(capture/'process-release.json')
    assert release['released'] and not release['survivors']
    assert len(spec['cases']) == len(truth['rows'])
    selected = [(i,c,r) for i,(c,r) in enumerate(zip(spec['cases'],truth['rows']))
                if c['source_role'] == source_role]
    assert len(selected) == 64 and len({c['group_id'] for _,c,_ in selected}) == 16
    own_groups = {c['group_id'] for _,c,_ in selected}
    assert not own_groups.intersection(c['group_id'] for c in spec['cases'] if c['source_role'] != source_role)
    assert torch.cuda.is_available()
    rgb, near, support, records = [], [], [], []
    for i, case, row in selected:
        assert case['name'] == row['name']
        ip, mp = capture/f'model/sample/{i:04d}.png', capture/row['mask_path']
        assert sha(ip) == row['rgb_sha256'] and sha(mp) == row['mask_sha256']
        with Image.open(ip) as image:
            assert image.size == (640,360)
            rgb.append(np.asarray(image.convert('RGB').resize((256,144),Image.Resampling.BOX)).copy())
        mask = np.load(mp,allow_pickle=False)
        assert mask.shape == (2,360,640) and np.isin(mask,[-1,0,1]).all()
        cells = torch.as_tensor(mask,device='cuda').reshape(2,18,20,32,20)
        positive = (cells == 1).any(dim=4).any(dim=2)
        unknown = (cells == -1).any(dim=4).any(dim=2)
        pooled = torch.where(positive,1,torch.where(unknown,-1,0)).to(torch.int8)
        support.append(pooled.cpu().numpy())
        near.append(row['body_head_visible_targets'])
        records.append(dict(sample_index=i,group_id=case['group_id'],source_site_id=case['source_site_id'],
            family=case['condition']['family'],intended_relation=case['condition']['desired_relation'],
            native_near=row['body_head_visible_targets'],rgb_sha256=sha(ip),mask_sha256=sha(mp),camera=case['camera']))
    y = np.array(near,np.float32)
    assert y.shape == (64,2) and np.isin(y,[0,1]).all()
    prefix = 'supervision' if role == 'train' else 'evaluator'
    output.mkdir(parents=True,exist_ok=False)
    (output/'source').mkdir()
    shutil.copy2(capture/'source/spec.json',output/'source/spec.json')
    shutil.copy2(admission_path,output/'source/capture-admission.json')
    arrays = {}
    for name, array in [('rgb',np.stack(rgb)),('near',y),('support',np.stack(support))]:
        rel = f'{"model" if name == "rgb" else prefix}/{role}/{name}.npy'
        path = output/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        np.save(path,array,allow_pickle=False)
        arrays[name] = dict(path=rel,sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype),bytes=path.stat().st_size)
    def write(rel,value):
        path=output/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2),encoding='utf-8')
    ids=[i for i,_,_ in selected]
    write(f'{prefix}/{role}.json',dict(sample_indices=ids,group_ids=[r['group_id'] for r in records],
        source_site_ids=[r['source_site_id'] for r in records],near=arrays['near'],support=arrays['support']))
    write('evaluator/groups.json',records)
    manifest=dict(schema='city-training-cache-v1',status='PASS',role=source_role,
        admission=admission,
        source_spec=dict(path='source/spec.json',sha256=sha(output/'source/spec.json')),
        capture_admission=dict(path='source/capture-admission.json',sha256=sha(output/'source/capture-admission.json')),
        partitions={role:dict(sample_indices=ids,rgb=arrays['rgb'])},source_capture=str(capture),
        source_spec_sha256=sha(capture/'source/spec.json'),native_verification_sha256=sha(capture/'world-verification.json'),
        source_sha256=sha(Path(__file__)),backend='CUDA',device=torch.cuda.get_device_name(),
        seconds=time.perf_counter()-started,native_positive_counts=y.sum(axis=0).astype(int).tolist(),
        scope='Same-world new regional source; native gaps preserved; EVAL labels never placed under supervision')
    write('manifest.json',manifest)
    return manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--role',choices=['train','eval'],required=True)
    parser.add_argument('--admission',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.capture,args.output,args.role,args.admission)))
