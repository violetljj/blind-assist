"""Append echo-independent native cell supervision without rewriting a source ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(capture, dependencies, runtime, output):
    sys.path.insert(0, str(runtime/'research/active/dtr-r0/nearfield'))
    sys.path.insert(0, str(dependencies))
    from mz9_contributors import _cached_geometry
    from mz8_attribution import query_membership
    output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    keep, rays, factor, zone, cell = _cached_geometry(device)
    address = (zone*49 + cell)[None]
    spec = json.loads((capture/'source/spec.json').read_text(encoding='utf-8-sig'))
    presence, known, inputs = [], [], []
    torch.set_num_threads(1)
    with torch.inference_mode():
        for index, case in enumerate(spec['cases']):
            path = capture/f'evaluator/native/{index:04d}.npy'
            native = torch.from_numpy(np.load(path, allow_pickle=False)).to(device)
            axial = native.flatten()[keep][None].double()
            radial = axial*factor
            valid = torch.isfinite(axial) & (axial>0) & (axial<100) & (radial<=4)
            points = axial[..., None]*rays
            points[..., 2] += 1.7
            member = query_membership(points)
            count = torch.zeros(1,64*49,dtype=torch.int64,device=device)
            count.scatter_add_(1,address,valid.long())
            known.append((count>0).reshape(64,49).cpu().numpy())
            events = []
            for query in range(4):
                count = torch.zeros(1,64*49,dtype=torch.int64,device=device)
                count.scatter_add_(1,address,(valid & member[...,query]).long())
                events.append((count>0).reshape(64,49))
            presence.append(torch.stack(events,-1).cpu().numpy())
            inputs.append(dict(index=index,frame_id=case['name'],native_sha256=sha(path)))
    presence, known = np.stack(presence), np.stack(known)
    assert not (presence & ~known[...,None]).any()
    target = output/'native-cell-supervision.npz'
    np.savez_compressed(target,cell_event_presence=presence,cell_known=known)
    receipt = dict(status='PASS',schema='mz48-echo-independent-native-supervision-v1',
        frames=len(inputs),shapes=dict(cell_event_presence=list(presence.shape),cell_known=list(known.shape)),
        semantics='Any valid native point in angular cell inside query; independent of echo selection and packet condition; absence in unobserved cells is UNKNOWN.',
        input_authority='Evaluator/training-only; never predictor input',
        code_sha256=sha(Path(__file__)),source_spec_sha256=sha(capture/'source/spec.json'),
        native_inputs=inputs,output_sha256=sha(target),output_bytes=target.stat().st_size,
        geometry_dependencies={name:sha(dependencies/name) for name in ('mz9_contributors.py','mz8_attribution.py','multizone64_observation.py','contact_retina_spec.py')},
        backend='CUDA',device=torch.cuda.get_device_name(),model_inference_frames=0,training_steps=0,
        original_source_mutations=0,original_zip_mutations=0)
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:receipt[k] for k in ('status','frames','shapes','output_sha256','output_bytes')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('capture','dependencies','runtime','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args()
    build(a.capture,a.dependencies,a.runtime,a.output)
