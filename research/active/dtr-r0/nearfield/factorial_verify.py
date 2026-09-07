"""Reuse native factorial checks, then publish only train/validation supervision."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from grounding_verify import verify

def materialize(root):
    root=root.resolve();verify(root)
    data=json.loads((root/'model/dataset.json').read_text(encoding='utf-8-sig'))
    spec=json.loads((root/'evaluator/spec.json').read_text(encoding='utf-8-sig'))
    if spec['preflight']:return
    groups={k:{s['group_id'] for s in data['samples'] if s['split']==k} for k in ('train','val','test')}
    assert [len(groups[k]) for k in ('train','val','test')]==[40,12,12]
    assert not (groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    ids={s['sample_id'] for s in data['samples'] if s['split']!='test'}
    labels=json.loads((root/'evaluator/labels.json').read_text())['targets']
    train=root/'training';train.mkdir(exist_ok=False)
    (train/'labels.json').write_text(json.dumps(dict(targets={k:labels[k] for k in ids}),indent=2),encoding='utf-8')
    with np.load(root/'evaluator/support.npz') as masks:np.savez_compressed(train/'support.npz',**{k:masks[k] for k in ids})
    variants={c['clip_id']:c['variant'] for c in spec['clips']}
    (train/'partition.json').write_text(json.dumps(dict(variants={s['sample_id']:variants[s['clip_id']] for s in data['samples'] if s['split']=='train'}),indent=2),encoding='utf-8')
    (train/'receipt.json').write_text(json.dumps(dict(status='PASS',train_groups=40,val_groups=12,test_groups=12,
        supervision_ids=len(ids),test_supervision_published=False,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);materialize(p.parse_args().capture)
