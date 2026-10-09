"""Official ARKitScenes synchronized lowres RGB/LiDAR ray references.

Camera-local Development only. Official loader backprojects axial millimetre
depth with the RGB pincam and reads RGB at the same pixels; no Tango extrinsic
assumption is imported. This does not establish wearer/body clearance.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image
from rgb_body_query_3rscan import optical_z, sensor_labels, task_queries
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_transfer_prepare import independent_labels


def index(archive, suffix):
    return {Path(name).stem: name for name in archive.namelist() if name.endswith(suffix)}


def prepare(source, output, frames=16, budget_s=240):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'selection.json').exists():
        raise FileExistsError('Preserve prior selection; no implicit restart')
    receipt = dict(status='STARTING', cpu_budget_s=budget_s, gpu_s=0,
                   source_sha256=sha(Path(__file__)), independent_xyz_checks=0)
    rows = []
    queries = task_queries()
    download = json.loads((source/'download_receipt.json').read_text())
    if download['status'] != 'COMPLETE':
        raise ValueError('Official acquisition incomplete')
    for row in download['rows']:
        if sha(source/row['name']) != row['sha256']:
            raise ValueError('Acquisition SHA mismatch')
    try:
        with zipfile.ZipFile(source/'lowres_wide.zip') as rgbz, \
             zipfile.ZipFile(source/'lowres_depth.zip') as depthz, \
             zipfile.ZipFile(source/'lowres_wide_intrinsics.zip') as kz:
            rgb_index, depth_index, k_index = index(rgbz, '.png'), index(depthz, '.png'), index(kz, '.pincam')
            common = sorted(set(rgb_index) & set(depth_index) & set(k_index), key=lambda n: float(n.rsplit('_',1)[1]))
            chosen_indices = np.rint(np.linspace(0, len(common)-1, frames)).astype(int)
            if len(set(chosen_indices)) != frames:
                raise ValueError('Insufficient exact synchronized pairs')
            chosen = [common[i] for i in chosen_indices]
            selection = dict(source='ARKitScenes raw Training/47333462', source_acquisition_sha256=sha(source/'download_receipt.json'),
                             method='16 uniform indices of exact RGB/depth/pincam timestamp intersection, no image/depth/outcome selection',
                             common_pairs=len(common), rgb_frames=len(rgb_index), depth_frames=len(depth_index), intrinsics_frames=len(k_index),
                             frames=[dict(frame=i, source_id=n, timestamp_s=float(n.rsplit('_',1)[1])) for i,n in enumerate(chosen)],
                             queries=queries, saved_before_depth_reads=True)
            write(output/'selection.json', selection)
            for frame, name in enumerate(chosen):
                if time.perf_counter()-start >= budget_s:
                    raise TimeoutError('240s CPU preparation wall limit')
                rgb_bytes, depth_bytes, k_bytes = rgbz.read(rgb_index[name]), depthz.read(depth_index[name]), kz.read(k_index[name])
                raw = np.asarray(Image.open(io.BytesIO(depth_bytes)))
                rgb = Image.open(io.BytesIO(rgb_bytes)).convert('RGB')
                w,h,fx,fy,cx,cy = np.fromstring(k_bytes.decode(), sep=' ')
                k = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)
                if (rgb.height,rgb.width) != (int(h),int(w)) or raw.shape != (int(h),int(w)) or raw.dtype != np.uint16:
                    raise ValueError('Only official same-size registered lowres assets supported')
                if not np.isfinite(k).all() or min(fx,fy)<=0:
                    raise ValueError('Invalid public pincam')
                depth = optical_z(raw, 1000.)
                my,mx = np.indices(depth.shape,dtype=np.float32)
                observed = np.ones(depth.shape,bool)
                labels, qs = zip(*(sensor_labels(depth,k,q,observed) for q in queries))
                for label,q in zip(labels,queries):
                    if not np.array_equal(label,independent_labels(depth,k,q,observed)):
                        raise ValueError('Independent XYZ labels disagree')
                    receipt['independent_xyz_checks'] += 1
                rgb_path = output/(name+'.png'); rgb_path.write_bytes(rgb_bytes)
                ref_path = output/(name+'.npz')
                np.savez_compressed(ref_path,labels=np.stack(labels),depth=depth,depth_K=k,color_K=k,map_x=mx,map_y=my,observed=observed)
                rows.append(dict(environment='arkitscenes_47333462',scan='arkitscenes_47333462',split='validation',official_split='Training',
                                 frame=frame,source_id=name,timestamp_s=float(name.rsplit('_',1)[1]),
                                 rgb_path=str(rgb_path),rgb_sha256=sha(rgb_path),reference_path=str(ref_path),reference_sha256=sha(ref_path),
                                 source_rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest(),source_depth_sha256=hashlib.sha256(depth_bytes).hexdigest(),
                                 calibration_sha256=hashlib.sha256(k_bytes).hexdigest(),color_K=k.tolist(),depth_K=k.tolist(),
                                 color_shape=list(raw.shape),depth_shape=list(raw.shape),depth_shift=1000.,queries=list(qs)))
        counts=dict(Counter(q['state'] for r in rows for q in r['queries']))
        units={key:sum(q[key] for r in rows for q in r['queries']) for key in ('positive_pixels','free_ray_pixels','unknown_pixels','domain_pixels')}
        manifest=dict(status='REAL_SENSOR_SPARSE_QUERY_READY',rows=rows,queries=queries,frames=len(rows),
                      groups=[dict(environment='arkitscenes_47333462',scan='arkitscenes_47333462',split='validation')],
                      state_counts_by_split={'train':{},'cal':{},'validation':counts},query_ray_units=units,
                      sensor='iPad Pro Apple LiDAR; official loader RGB pincam axial depth/1000 metres, same registered lowres pixels; not metrology GT',
                      source_alignment_basis='Official tenFpsDataLoader.py generate_point uses intrinsic inverse*[u*d,v*d,d] and rgb_image[v,u]; verified same256x192 RGB/depth/pincam',
                      label_contract='0 first measured surface beyond query;1 surface inside;2 missing/occluded;255 no ray-box intersection',
                      query_contract='camera-local sampled first-return rays; FREE_ON_SAMPLED_RAYS does not certify entire volume',
                      observation_contract='RGB/public K/query only; evaluator depth/labels/masks excluded',
                      official_data_role='official Training capture, frozen cross-camera Development; one capture, no broad-generalization claim',
                      license='Official Apple LICENSE permits non-commercial research; source cached under ignored artifacts, no data redistribution',
                      selection_sha256=sha(output/'selection.json'))
        write(output/'dataset_manifest.json',manifest)
        public_fields=('environment','scan','split','frame','rgb_path','rgb_sha256','color_K','color_shape','depth_K','depth_shape')
        write(output/'observations.json',dict(rows=[{k:r[k] for k in public_fields} for r in rows],queries=queries,dataset_manifest_sha256=sha(output/'dataset_manifest.json')))
        receipt.update(status='COMPLETE',frames=len(rows),query_cells=len(rows)*len(queries),query_states=counts,query_ray_units=units,
                       source_download_bytes=download['total_received_bytes'],source_acquisition_sha256=sha(source/'download_receipt.json'),
                       selection_sha256=sha(output/'selection.json'),manifest_sha256=sha(output/'dataset_manifest.json'),observations_sha256=sha(output/'observations.json'),
                       observations_evaluator_fields_absent=True,exact_timestamp_rgb_depth_intrinsics=True)
    except Exception as exc:
        receipt.update(status='FAILED',error=repr(exc),completed_frames=len(rows)); raise
    finally:
        receipt['cpu_wall_s']=time.perf_counter()-start
        write(output/'preparation_receipt.json',receipt)
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=16); p.add_argument('--budget-s',type=float,default=240)
    a=p.parse_args(); prepare(a.source.resolve(),a.output.resolve(),a.frames,a.budget_s)
