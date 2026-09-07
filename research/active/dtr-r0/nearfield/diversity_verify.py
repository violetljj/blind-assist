"""Native verification plus TRAIN/VAL-only supervision for G10."""
import argparse
import json
from pathlib import Path
import numpy as np
from grounding_verify import verify, read, sha


def materialize(root):
    root=root.resolve();verify(root)
    data=read(root/'model/dataset.json');spec=read(root/'evaluator/spec.json')
    if spec['preflight']:return
    groups={k:{s['group_id'] for s in data['samples'] if s['split']==k} for k in ('train','val','test')}
    assert [len(groups[k]) for k in groups]==[160,24,32]
    assert not (groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    ids=sorted(s['sample_id'] for s in data['samples'] if s['split']!='test')
    labels=read(root/'evaluator/labels.json')['targets'];train=root/'training';train.mkdir(exist_ok=False)
    (train/'labels.json').write_text(json.dumps(dict(targets={k:labels[k] for k in ids}),indent=2),encoding='utf-8')
    with np.load(root/'evaluator/support.npz',allow_pickle=False) as masks:np.savez_compressed(train/'support.npz',**{k:masks[k] for k in ids})
    variants={c['clip_id']:c['variant'] for c in spec['clips']}
    (train/'partition.json').write_text(json.dumps(dict(variants={s['sample_id']:variants[s['clip_id']] for s in data['samples'] if s['split']=='train'}),indent=2),encoding='utf-8')
    (train/'receipt.json').write_text(json.dumps(dict(status='PASS',supervision_ids=len(ids),test_supervision_published=False,code_sha256=sha(Path(__file__))),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);materialize(p.parse_args().capture)
