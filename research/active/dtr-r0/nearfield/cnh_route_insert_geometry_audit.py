"""Independent CUDA geometry diagnostic for a consumed inserted-asset canary.

Fixed stride-four rays; descriptive measurements only, never source admission.
"""
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
from cnh_route_scene_depth_audit import trace_cuda


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def world_triangles(geometry):
    x,y,z,w = geometry['actual_rotation_quaternion']
    quaternion = np.asarray([x,y,z,w],dtype=np.float64)
    if not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion)-1) > 1e-5:
        raise ValueError('Expected finite unit quaternion in XYZW order')
    rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float64)
    chunks=[]
    for section in geometry['sections']:
        vertices=np.asarray(section['vertices_m'],dtype=np.float64)
        indices=np.asarray(section['triangles'],dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or indices.size % 3 or not indices.size:
            raise ValueError('Malformed exported geometry')
        if indices.min() < 0 or indices.max() >= len(vertices):
            raise ValueError('Triangle indices outside vertex buffer')
        vertices=(vertices*np.asarray(geometry['actual_scale']))@rotation.T+np.asarray(geometry['actual_translation_m'])
        chunks.append(vertices[indices.reshape(-1,3)])
    if not chunks:
        raise ValueError('Missing target sections')
    triangles=np.concatenate(chunks)
    if not np.isfinite(triangles).all():
        raise ValueError('Nonfinite geometry')
    return triangles


def stats(values):
    return dict(count=int(len(values)),median=float(np.median(values)) if len(values) else None,
        p95=float(np.quantile(values,.95)) if len(values) else None,
        maximum=float(np.max(values)) if len(values) else None)


def audit(capture):
    journal=os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    if not journal or json.loads(Path(journal).read_text(encoding='utf-8-sig')).get('state') != 'running':
        raise RuntimeError('Use governed research-ue execution')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    capture=Path(capture).resolve(strict=True)
    receipt=json.loads((capture/'terminal.json').read_text())
    if receipt.get('status') != 'PASS_INSERTED_ASSET_ID_CANARY':
        raise ValueError('A completed successful inserted-asset native canary is required')
    manifest=json.loads((capture/'raw-manifest.json').read_text())
    rows=manifest['frames']
    if len(rows) != 6 or {(r['asset_id'],r['occlusion']) for r in rows} != {(a,c) for a in (1,254) for c in ('visible','partial','occluded')}:
        raise ValueError('Six fixed control frames required')
    start=time.monotonic(); reports=[]
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        folder=(capture/row['folder']).resolve(strict=True)
        if not folder.is_relative_to(capture) or folder == capture:
            raise ValueError('Frame escapes capture tree')
        geometry=json.loads((folder/'target-geometry.json').read_text())
        camera=json.loads((folder/'camera.json').read_text())
        if geometry['inserted_id'] != row['asset_id']:
            raise ValueError('Geometry owner differs from manifest')
        isolated=np.load(folder/'isolated_depth.transport.npy',allow_pickle=False)
        scene=np.load(folder/'depth_left.transport.npy',allow_pickle=False)
        with Image.open(folder/'instance_left.png') as image:
            ids=np.asarray(image).copy()
        if isolated.shape != (360,640) or scene.shape != isolated.shape or ids.shape != isolated.shape:
            raise ValueError('Frozen 640x360 canary required')
        if isolated.dtype != np.dtype('<f4') or scene.dtype != np.dtype('<f4'):
            raise ValueError('Native float32 axial metres required')
        matrix=np.asarray(camera['T_world_camera'],dtype=np.float64)
        k=np.asarray(camera['K'],dtype=np.float64)
        if matrix.shape != (4,4) or k.shape != (3,3) or not np.isfinite(matrix).all() or not np.isfinite(k).all():
            raise ValueError('Invalid camera calibration')
        yy,xx=np.meshgrid(np.arange(2,360,4),np.arange(2,640,4),indexing='ij')
        optical=np.stack([(xx.ravel()-k[0,2])/k[0,0],(yy.ravel()-k[1,2])/k[1,1],np.ones(xx.size)],axis=-1)
        triangles=world_triangles(geometry)
        expected,_=trace_cuda(matrix[:3,3],optical@matrix[:3,:3].T,triangles)
        observed=isolated[yy,xx].ravel(); scene_values=scene[yy,xx].ravel()
        norm=np.linalg.norm(optical,axis=-1)
        native_hit=np.isfinite(observed)&(observed>0)&(observed<100)
        mesh_hit=np.isfinite(expected)&(expected>0)&(expected<100)
        common=native_hit&mesh_hit
        axial=np.abs(expected[common]-observed[common]); radial=axial*norm[common]
        id_hit=ids[yy,xx].ravel()==row['asset_id']
        scene_valid=np.isfinite(scene_values)&(scene_values>0)&(scene_values<100)
        # Reuse the frozen transport identity tolerance; this is not a new gate.
        mesh_visible=mesh_hit&scene_valid&(np.abs(expected.astype(float)-scene_values.astype(float))<=.001)
        visible_common=common&id_hit
        names=('target-geometry.json','camera.json','isolated_depth.transport.npy','depth_left.transport.npy','instance_left.png')
        reports.append(dict(id=row['id'],asset_id=row['asset_id'],occlusion=row['occlusion'],triangles=len(triangles),
            selected_rays=int(xx.size),native_isolated_hits=int(native_hit.sum()),mesh_hits=int(mesh_hit.sum()),common_hits=int(common.sum()),
            missing_mesh_hits=int((native_hit&~mesh_hit).sum()),false_mesh_hits=int((mesh_hit&~native_hit).sum()),
            axial_abs_error_m=stats(axial),radial_abs_error_m=stats(radial),
            sampled_visible_id_pixels=int(id_hit.sum()),full_visible_id_pixels=int((ids==row['asset_id']).sum()),
            visible_id_missing_mesh_hits=int((id_hit&~mesh_hit).sum()),
            visible_id_axial_abs_error_m=stats(np.abs(expected[visible_common]-observed[visible_common])),
            visible_id_radial_abs_error_m=stats(np.abs(expected[visible_common]-observed[visible_common])*norm[visible_common]),
            mesh_depth_visible_pixels=int(mesh_visible.sum()),mesh_visible_without_id=int((mesh_visible&~id_hit).sum()),
            id_without_mesh_depth_visibility=int((id_hit&~mesh_visible).sum()),
            input_sha256={name:sha(folder/name) for name in names}))
    return dict(status='DESCRIPTIVE_DIAGNOSTIC_COMPLETE',benchmark_eligible=False,source_admission='NOT_RUN',
        scope='CONSUMED_INSERTED_ASSET_CANARY_EXPORTED_GEOMETRY_VS_NATIVE',backend='CUDA',
        device=torch.cuda.get_device_name(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        pixel_grid=dict(stride=4,centre_offset=2,selection='FIXED_FULL_IMAGE_GRID_NOT_MODEL_SELECTED'),
        depth_units='AXIAL_METRES',radial_conversion='AXIAL_TIMES_OPTICAL_RAY_NORM',
        native_hit_window_axial_m=[0,100],visibility_comparison_tolerance_m=.001,
        acceptance_thresholds='NONE_DESCRIPTIVE_ONLY',
        limitations='Two-sided exported target rays only; sparse pixels can miss thin features. No background coverage or formal source admission.',
        capture=str(capture),journal=journal,manifest_sha256=sha(capture/'raw-manifest.json'),
        auditor_sha256=sha(Path(__file__)),trace_implementation_sha256=sha(Path(__file__).with_name('cnh_route_scene_depth_audit.py')),
        frames=reports,wall_s=time.monotonic()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    artifact_root=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    output=args.output.resolve()
    if output.exists() or not output.is_relative_to(artifact_root) or output==artifact_root:
        raise ValueError('Fresh output strictly under artifacts.local required')
    result=audit(args.capture)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
