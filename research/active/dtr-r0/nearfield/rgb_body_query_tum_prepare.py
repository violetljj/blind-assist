"""TUM Freiburg3 registered Kinect references on native rays, Development only.

Uniform RGB selection precedes reference reads; nearest timestamp association
is not exact synchronization. Downloaded sensor depth is not metrology or body
clearance truth. Pose is deliberately unused; observations exclude references.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import numpy as np
from PIL import Image
from rgb_body_query_3rscan import optical_z, sensor_labels, task_queries
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_transfer_prepare import independent_labels


def fixed_pairs(rgb_entries, depth_entries, count=16, tolerance=.02):
    """No filtering by depth quality or outcomes; rejected selected RGB stays rejected."""
    rgb_entries=sorted(rgb_entries); depth_entries=sorted(depth_entries)
    indices=np.rint(np.linspace(0,len(rgb_entries)-1,count)).astype(int)
    if len(set(indices)) != count: raise ValueError('Insufficient distinct RGB timestamps')
    rows=[]
    for i in indices:
        t,rgb=rgb_entries[i]
        dt,depth=min(depth_entries,key=lambda item:(abs(item[0]-t),item[0],item[1]))
        rows.append(dict(rgb_index=int(i),rgb_timestamp_s=t,depth_timestamp_s=dt,
                         time_difference_s=abs(dt-t),rgb_member=rgb,depth_member=depth,
                         accepted=abs(dt-t)<=tolerance))
    if len({r['depth_member'] for r in rows if r['accepted']}) != sum(r['accepted'] for r in rows):
        raise ValueError('Selected references must be distinct')
    return rows


def prepare(source,output,frames=16,budget_s=240,sequence='freiburg3_long_office_household',sensor='Asus Xtion structured-light sensor'):
    start=time.perf_counter(); output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise FileExistsError('Preserve preparation attempt')
    acquisition=json.loads((source/'download_receipt.json').read_text())
    if not sequence.startswith('freiburg3_') or not all(c.isalnum() or c=='_' for c in sequence):
        raise ValueError('Only public Freiburg3 sequence identities are supported')
    environment='tum_'+sequence.replace('freiburg3_','fr3_',1)
    receipt=dict(status='STARTING',cpu_budget_s=budget_s,gpu_s=0,source_sha256=sha(Path(__file__)),
                 independent_xyz_checks=0,source_acquisition_sha256=sha(source/'download_receipt.json'))
    rows=[]; queries=task_queries(); k=np.array([[535.4,0,320.1],[0,539.2,247.6],[0,0,1]],np.float64)
    try:
        if acquisition['status']!='COMPLETE': raise ValueError('Source acquisition incomplete')
        archive=source/acquisition['rows'][0]['name']
        if sha(archive)!=acquisition['rows'][0]['sha256']: raise ValueError('Source SHA mismatch')
        rgb_entries=[]; depth_entries=[]; texts={}
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                if time.perf_counter()-start>budget_s: raise TimeoutError('CPU preparation budget')
                path=Path(member.name)
                if not member.isfile(): continue
                if path.suffix=='.png' and path.parent.name in ('rgb','depth'):
                    target=rgb_entries if path.parent.name=='rgb' else depth_entries
                    target.append((float(path.stem),member.name))
                elif path.name in ('rgb.txt','depth.txt'):
                    texts[path.name]=tar.extractfile(member).read().decode()
        chosen=fixed_pairs(rgb_entries,depth_entries,frames)
        selection=dict(source='TUM RGB-D '+sequence,method=f'{frames} uniform RGB indices, nearest timestamp depth <=0.020 s, deterministic earlier-time tie break; no reference/image/outcome filtering',
                       frames=chosen,rgb_frames=len(rgb_entries),depth_frames=len(depth_entries),queries=queries,
                       saved_before_depth_reads=True,time_tolerance_s=.02,exact_synchronization=False)
        write(output/'selection.json',selection)
        for name,payload in texts.items(): (output/name).write_text(payload)
        required={r[field] for r in chosen if r['accepted'] for field in ('rgb_member','depth_member')}; selected={}
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                if time.perf_counter()-start>budget_s: raise TimeoutError('CPU preparation budget')
                if member.name in required: selected[member.name]=tar.extractfile(member).read()
                if len(selected)==len(required): break
        if set(selected)!=required: raise ValueError('Selected asset missing')
        yy,xx=np.indices((480,640),dtype=np.float32); observed=np.ones((480,640),bool)
        for frame,selection_row in enumerate(chosen):
            if not selection_row['accepted']: continue
            if time.perf_counter()-start>budget_s: raise TimeoutError('CPU preparation budget')
            rgb_bytes=selected[selection_row['rgb_member']]; depth_bytes=selected[selection_row['depth_member']]
            rgb=Image.open(io.BytesIO(rgb_bytes)).convert('RGB'); raw=np.asarray(Image.open(io.BytesIO(depth_bytes)))
            if raw.shape!=(480,640) or raw.dtype!=np.uint16 or rgb.size!=(640,480):
                raise ValueError('Official native RGB/depth dimensions/type mismatch')
            depth=optical_z(raw,5000.); labels,states=zip(*(sensor_labels(depth,k,q,observed) for q in queries))
            for label,q in zip(labels,queries):
                if not np.array_equal(label,independent_labels(depth,k,q,observed)): raise ValueError('Independent XYZ mismatch')
                receipt['independent_xyz_checks']+=1
            stem=f'{environment}_{frame:06d}'
            rgb_path=output/(stem+'.png'); rgb_path.write_bytes(rgb_bytes)
            ref_path=output/(stem+'.npz')
            np.savez_compressed(ref_path,labels=np.stack(labels),depth=depth,depth_K=k,color_K=k,map_x=xx,map_y=yy,observed=observed)
            rows.append(dict(environment=environment,scan=environment,split='validation',official_split='public_sequence',
                             frame=frame,rgb_timestamp_s=selection_row['rgb_timestamp_s'],depth_timestamp_s=selection_row['depth_timestamp_s'],
                             time_difference_s=selection_row['time_difference_s'],rgb_path=str(rgb_path),rgb_sha256=sha(rgb_path),
                             reference_path=str(ref_path),reference_sha256=sha(ref_path),color_K=k.tolist(),depth_K=k.tolist(),
                             color_shape=[480,640],depth_shape=[480,640],depth_shift=5000.,queries=list(states),
                             source_rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest(),source_depth_sha256=hashlib.sha256(depth_bytes).hexdigest()))
        counts=dict(Counter(q['state'] for r in rows for q in r['queries']))
        units={key:sum(q[key] for r in rows for q in r['queries']) for key in ('positive_pixels','free_ray_pixels','unknown_pixels','domain_pixels')}
        manifest=dict(status='REAL_SENSOR_SPARSE_QUERY_READY',frames=len(rows),rows=rows,queries=queries,
                      groups=[dict(environment=environment,scan=environment,split='validation')],
                      state_counts_by_split={'train':{},'cal':{},'validation':counts},query_ray_units=units,
                      sensor=sensor+', registered 640x480 Freiburg3 undistorted RGB/depth; optical Z=uint16/5000, invalid0 unknown; source pre-scaled, no extra ds',
                      source_alignment_basis='Official TUM file_formats: depth reprojected to color frame pixel correspondence1:1; fr3 public RGB K, distortion0; nearest timestamps <=20ms, not exact synchronization',
                      label_contract='0 first measured surface after box;1 surface in box;2 missing/occluded;255 no ray-box intersection',
                      observation_contract='RGB/public K/query only; evaluator reference depth/labels excluded',
                      query_contract='camera-local sampled first-return rays; FREE_ON_SAMPLED_RAYS does not certify entire volume',
                      license='TUM RGB-D CC BY4.0, Sturm et al. IROS2012 attribution; local research-only no redistribution',
                      official_data_role='one public handheld capture, frozen cross-camera Development; no wearer/body/event reference',
                      selection_sha256=sha(output/'selection.json'))
        write(output/'dataset_manifest.json',manifest)
        fields=('environment','scan','split','frame','rgb_path','rgb_sha256','color_K','color_shape','depth_K','depth_shape')
        write(output/'observations.json',dict(rows=[{key:r[key] for key in fields} for r in rows],queries=queries,dataset_manifest_sha256=sha(output/'dataset_manifest.json')))
        receipt.update(status='COMPLETE',frames=len(rows),query_cells=len(rows)*len(queries),query_states=counts,query_ray_units=units,
                       source_download_bytes=acquisition['total_received_bytes'],max_time_difference_s=max(r['time_difference_s'] for r in rows),
                       selection_sha256=sha(output/'selection.json'),manifest_sha256=sha(output/'dataset_manifest.json'),observations_sha256=sha(output/'observations.json'),observations_evaluator_fields_absent=True)
    except Exception as exc:
        receipt.update(status='FAILED',error=repr(exc),completed_frames=len(rows)); raise
    finally:
        receipt['cpu_wall_s']=time.perf_counter()-start; write(output/'preparation_receipt.json',receipt); print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=16); p.add_argument('--budget-s',type=float,default=240)
    p.add_argument('--sequence',default='freiburg3_long_office_household'); p.add_argument('--sensor',default='Asus Xtion structured-light sensor')
    a=p.parse_args(); prepare(a.source.resolve(),a.output.resolve(),a.frames,a.budget_s,a.sequence,a.sensor)
