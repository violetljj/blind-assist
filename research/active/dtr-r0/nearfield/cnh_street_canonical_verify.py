"""Verify exact depth and declared float16 storage; preserve audit receipts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import OpenEXR
from PIL import Image
from cnh_route_insert_launch import compose_instance_ids
from cnh_street_storage_plan import crosscheck_ids


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(capture,output):
    root=Path(capture).resolve(strict=True)
    spec=json.loads((root/'source/spec.json').read_text(encoding='utf-8-sig'))
    if spec.get('scope')!='STREET_DEVELOPMENT_PILOT_NOT_BENCHMARK':
        raise ValueError('Only current Development capture is eligible')
    manifest=json.loads((root/'raw-manifest.json').read_text())
    old=json.loads((root/'format-receipt.json').read_text())
    if old['status']!='PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY':raise ValueError('Transport incomplete')
    historical={r['id']:r['hashes'] for r in old['frames']};samples=crosscheck_ids(manifest['frames'])
    entries=[];delete=[];max_error={'normal':0.,'albedo':0.}
    for row in manifest['frames']:
        folder=(root/row['folder']).resolve(strict=True)
        if not folder.is_relative_to(root):raise ValueError('Escaping frame directory')
        for side in ('left','right'):
            raw=np.load(folder/f'depth_{side}.transport.npy',allow_pickle=False)
            mask=np.load(folder/f'depth_{side}_valid.npy',allow_pickle=False)
            with OpenEXR.File(str(folder/f'depth_{side}.exr'),separate_channels=True) as f:
                depth=f.channels()['Z'].pixels
            expected=np.isfinite(raw)&(raw>0)&(raw<100)
            if not np.array_equal(mask,expected) or not np.array_equal(raw[mask],depth[mask]) or not np.isnan(depth[~mask]).all():
                raise ValueError('Nonexact depth EXR roundtrip: '+row['id'])
        for kind in ('normal','albedo'):
            raw=np.load(folder/f'{kind}_left.transport.npy',allow_pickle=False)
            canonical=np.load(folder/f'{kind}_left.npy',allow_pickle=False)
            if canonical.dtype!=np.float16 or not np.array_equal(canonical,raw.astype(np.float16),equal_nan=True):
                raise ValueError('Float16 conversion differs: '+row['id'])
            valid=np.load(folder/f'{kind}_left_valid.npy',allow_pickle=False)
            if valid.any():max_error[kind]=max(max_error[kind],float(np.max(np.abs(canonical[valid].astype(float)-raw[valid]))))
        depth=np.load(folder/'depth_left.transport.npy',allow_pickle=False)
        valid=np.isfinite(depth)&(depth>0)&(depth<100)
        ids=np.where(valid,0,65535).astype(np.uint16);owners=np.zeros(depth.shape,dtype=np.uint8);ambiguous=np.zeros(depth.shape,dtype=bool)
        for identifier in (1,254):
            candidate=compose_instance_ids(depth,np.load(folder/f'isolated_depth_{identifier}.transport.npy',allow_pickle=False),identifier)
            mask=candidate==identifier;owners[mask]+=1;ids[mask]=identifier;ambiguous|=candidate==65535
        ids[ambiguous|(owners>1)]=65535
        with Image.open(folder/'instance_left.png') as f:
            if not np.array_equal(ids,np.asarray(f)):raise ValueError('ID encoding replay differs')
        for name in ('depth_left.transport.npy','depth_right.transport.npy','normal_left.transport.npy','albedo_left.transport.npy',
                     *(() if row['id'] in samples else ('isolated_depth_1.transport.npy','isolated_depth_254.transport.npy'))):
            path=folder/name;digest=sha(path)
            if digest!=historical[row['id']][name]:raise ValueError('Transport hash changed')
            delete.append(dict(path=str(path),sha256=digest,bytes=path.stat().st_size))
        entries.append(dict(frame_id=row['id'],layout_id=row['layout_id'],canonical_depth='EXACT_VALID_FLOAT32',
            canonical_attributes='EXACT_DECLARED_FLOAT16_CONVERSION',id_encoding='EXACT_REPLAY_NOT_INDEPENDENT_LABEL_PRECISION',
            retained_crosscheck=row['id'] in samples))
    receipt=dict(schema='cnh-street-canonical-verification-v1',status='PASS_CANONICAL_ENCODING',capture=str(root),
        frame_count=len(entries),frames=entries,max_attribute_quantization_error=max_error,
        original_transport_receipt_sha256=sha(root/'format-receipt.json'),manifest_sha256=sha(root/'raw-manifest.json'),
        code_sha256=sha(__file__),delete_candidates=delete,candidate_bytes=sum(r['bytes'] for r in delete),
        label_precision='NOT_ESTABLISHED_BY_ENCODING_REPLAY',files_deleted=0,
        retained_evidence='All source/evaluator/metadata and first/last isolated depths per clip; original hashes preserved')
    with Path(output).open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('frames','delete_candidates')}))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--capture',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.capture,a.output)
