"""Descriptive CUDA native-depth / exported LOD0 comparison; never admission."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from cnh_route_capture import camera_record


def world_triangles(folder, receipt, section_filter=None):
    triangles, owners, paths, missing = [], [], [], []
    cached = {}
    for instance in receipt['instances']:
        descriptor = instance['mesh']
        if descriptor.get('status') != 'EXPORTED_LOD0_NOT_RENDER_VERIFIED':
            missing.append(dict(id=instance['id'], mesh=descriptor.get('asset_path')))
            continue
        path = (folder/descriptor['path']).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('Mesh path escapes scene probe')
        if path not in cached:
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != descriptor['sha256']:
                raise ValueError('Mesh hash mismatch: '+str(path))
            cached[path] = json.loads(payload)
        x,y,z,w = instance['actual_rotation_quaternion']
        rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]], dtype=np.float64)
        owner = len(paths); paths.append(descriptor['asset_path'])
        for section in cached[path]['sections']:
            if section_filter is not None and not section_filter(instance,section):
                continue
            vertices = np.asarray(section['vertices_m'], dtype=np.float64)*instance['actual_scale']
            vertices = vertices@rotation.T + instance['actual_translation_m']
            tri = vertices[np.asarray(section['triangles']).reshape(-1,3)]
            if not np.isfinite(tri).all():
                raise ValueError('Nonfinite exported geometry')
            triangles.append(tri); owners.extend([owner]*len(tri))
    return (np.concatenate(triangles) if triangles else np.empty((0,3,3)),
            np.asarray(owners, dtype=np.int64), paths, missing)


def trace_cuda(origin, directions, triangles):
    """Two-sided rays; unnormalised directions retain optical-axis depth units."""
    device = torch.device('cuda')
    rays = torch.as_tensor(directions, dtype=torch.float32, device=device)
    # Translate in float64 before conversion to avoid large-world cancellation.
    tri = np.asarray(triangles-origin, dtype=np.float32)
    nearest = torch.full((len(rays),), float('inf'), device=device)
    winner = torch.full((len(rays),), -1, dtype=torch.long, device=device)
    for start in range(0, len(tri), 16384):
        block = torch.as_tensor(tri[start:start+16384], device=device)
        edge1, edge2, offset = block[:,1]-block[:,0], block[:,2]-block[:,0], -block[:,0]
        q = torch.cross(offset, edge1, dim=-1)
        numerator = (edge2*q).sum(-1)
        for rstart in range(0, len(rays), 64):
            direction = rays[rstart:rstart+64]
            p = torch.cross(direction[:,None,:], edge2[None,:,:], dim=-1)
            determinant = (p*edge1).sum(-1)
            inverse = torch.where(determinant.abs()>1e-9, 1/determinant, 0.)
            u = (offset*p).sum(-1)*inverse
            v = (direction[:,None,:]*q).sum(-1)*inverse
            distance = numerator*inverse
            valid = (determinant.abs()>1e-9)&(u>=0)&(v>=0)&(u+v<=1)&(distance>0)
            distance = torch.where(valid, distance, float('inf'))
            best, index = distance.min(dim=1)
            old = nearest[rstart:rstart+len(direction)]
            update = best < old
            winner[rstart:rstart+len(direction)] = torch.where(update, index+start,
                winner[rstart:rstart+len(direction)])
            nearest[rstart:rstart+len(direction)] = torch.minimum(best, old)
    return nearest.cpu().numpy(), winner.cpu().numpy()


def audit(capture):
    if not torch.cuda.is_available():
        raise RuntimeError('Actual CUDA required; no CPU fallback')
    started = time.monotonic(); capture = Path(capture)
    spec_path = capture/'source/spec.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for folder in sorted((capture/'evaluator').glob('cnh-scene-*')):
        index = int(folder.name.rsplit('-', 1)[1])
        receipt_path = folder/'scene-probe.json'
        receipt = json.loads(receipt_path.read_text())
        depth_path = capture/f'evaluator/native/{index:04d}.npy'
        native = np.load(depth_path, allow_pickle=False)
        height, width = native.shape
        if (height, width) != (360, 640):
            raise ValueError('Only frozen 640x360 HFOV100 captures supported')
        camera = camera_record(spec['cases'][index]['camera'],
            dict(width=width, height=height, hfov_deg=100, baseline_m=.06))
        matrix = np.asarray(camera['T_world_camera']); k = np.asarray(camera['K'])
        yy, xx = np.meshgrid(np.arange(8,height,16), np.arange(8,width,16), indexing='ij')
        optical = np.stack([(xx.ravel()-k[0,2])/k[0,0],
                            (yy.ravel()-k[1,2])/k[1,1], np.ones(xx.size)], axis=-1)
        directions = optical@matrix[:3,:3].T
        triangles, owners, paths, missing = world_triangles(folder, receipt)
        expected, winner = trace_cuda(matrix[:3,3], directions, triangles)
        observed = native[yy,xx].ravel(); norm = np.linalg.norm(optical, axis=1)
        native_near = np.isfinite(observed)&(observed>0)&(observed*norm<=5.)
        expected_near = np.isfinite(expected)&(expected>0)&(expected*norm<=5.)
        common = native_near&expected_near
        error = np.abs(expected[common]-observed[common])
        attribution = {}
        for i in np.where(expected_near)[0]:
            key = paths[owners[winner[i]]]
            attribution[key] = attribution.get(key, 0)+1
        rows.append(dict(index=index, selected_rays=len(observed), native_near=int(native_near.sum()),
            native_near_mask=native_near.tolist(), predicted_near_mask=expected_near.tolist(),
            predicted_near=int(expected_near.sum()), near_mask_disagreement=int((native_near!=expected_near).sum()),
            near_mask_disagreement_fraction=float(np.mean(native_near!=expected_near)),
            native_only_near=int((native_near&~expected_near).sum()),
            predicted_only_near=int((expected_near&~native_near).sum()), common_near=int(common.sum()),
            axial_abs_error_m=dict(median=float(np.median(error)) if len(error) else None,
                p95=float(np.quantile(error,.95)) if len(error) else None,
                maximum=float(error.max()) if len(error) else None),
            predicted_attribution_by_mesh=attribution, attribution_authority='LOD0_RAYCAST_NOT_RENDERED_INSTANCE_ID',
            missing_export_instances=len(missing), missing_exports=missing, triangles=len(triangles),
            unsupported_primitives=len(receipt['unsupported_primitives']),
            probe_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            native_depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest()))
    if not rows:
        raise ValueError('No scene probe receipts found')
    return dict(status='NOT_VERIFIED', scope='DESCRIPTIVE_SPARSE_NEARFIELD_LOD0_NATIVE_DEPTH',
        backend='CUDA', device=torch.cuda.get_device_name(), peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        pixel_grid=dict(stride=16, centre_offset=8), radial_limit_m=5., depth_units='AXIAL_METRES',
        caveat='No rendered instance ID, material equivalence or full-scene completeness established',
        spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(), frames=rows,
        wall_s=time.monotonic()-started)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.capture)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
