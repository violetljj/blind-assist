"""Fixed additional ARKit captures, range-acquired subsets and 27 native ray boxes.

Same iPad/LiDAR family Development only; third hardware and body/event reference
remain absent. Selection uses full official directory names before any pixels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile
import numpy as np
from PIL import Image
from rgb_body_query_hires_prepare import RangeZip
from rgb_body_query_3rscan import optical_z, sensor_labels
from rgb_body_query_fixed_grid import queries, independent_xyz_labels, PUBLIC
from rgb_body_query_input_diagnostic import sha, write


def run(source, output, network_s=710, cpu_s=270, byte_limit=803000000):
    start=time.monotonic(); cpu_start=time.process_time(); deadline=start+network_s
    source.mkdir(parents=True,exist_ok=True); output.mkdir(parents=True,exist_ok=True)
    if (source/'selection.json').exists(): raise FileExistsError('Preserve prior attempt')
    rows=list(csv.DictReader((source/'raw_train_val_splits.csv').read_text().splitlines()))
    chosen_rows=[];seen_visits=set()
    for row in sorted(rows,key=lambda r:int(r['video_id'])):
        visit=row.get('visit_id')
        if row['fold']!='Training' or row['video_id']=='47333462' or not visit or visit in ('NA','nan','NaN') or visit in seen_visits: continue
        chosen_rows.append(row);seen_visits.add(visit)
        if len(chosen_rows)==2:break
    ids=[r['video_id'] for r in chosen_rows]
    if len(ids)!=2: raise ValueError('Need two fixed official Training captures')
    fixed=queries()
    selection=dict(captures=ids,metadata_sha256=sha(source/'raw_train_val_splits.csv'),
        rule='Numeric official Training video_id order, excluding consumed47333462 and missing visit_id; first capture from each of first2 distinct nonmissing visit groups; no outcome selection',
        official_metadata_rows=chosen_rows,visit_relation='Different official visit_id groups; not independent physical-scene confirmation',
        timestamp_rule='16 uniform indices of full-directory RGB/depth/pincam exact timestamp intersection',
        queries=fixed,saved_before_asset_reads=True)
    write(source/'selection.json',selection)
    records=[]; receipt=dict(status='STARTING',captured_sources=[],frames=0,gpu_s=0,
        source_sha256=sha(__file__),selection_sha256=sha(source/'selection.json'),
        original_TUM_status='NOT_RUN_INCOMPLETE_SOURCE',third_hardware_completed=False)
    def budget():
        if time.monotonic()>=deadline: raise TimeoutError('Remaining network allocation')
        if time.process_time()-cpu_start>cpu_s: raise TimeoutError('CPU processing allocation')
        if sum(r['bytes'] for r in records)>byte_limit: raise TimeoutError('Received byte allocation')
    try:
        for capture in ids:
            budget(); dest=source/capture; dest.mkdir(exist_ok=True)
            sensor=output/capture; sensor.mkdir(exist_ok=True)
            archives={}; streams={}; details={}; indices={}
            try:
                for asset,suffix in [('lowres_wide','.png'),('lowres_depth','.png'),('lowres_wide_intrinsics','.pincam')]:
                    budget(); url=f'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{capture}/{asset}.zip'
                    req=urllib.request.Request(url,method='HEAD'); t=time.monotonic()
                    with urllib.request.urlopen(req,timeout=min(25,max(1,deadline-t))) as response:
                        if response.status!=200: raise ValueError('Official asset HEAD not200')
                        size=int(response.headers['Content-Length']); headers=dict(response.headers)
                    stream=RangeZip(url,size,records,deadline); archive=zipfile.ZipFile(stream)
                    directory=[dict(name=i.filename,crc32=i.CRC,size=i.file_size,compressed_size=i.compress_size,header_offset=i.header_offset) for i in archive.infolist()]
                    write(dest/(asset+'_directory.json'),directory)
                    index={Path(n).stem:n for n in archive.namelist() if n.endswith(suffix)}
                    if len(index)!=sum(n.endswith(suffix) for n in archive.namelist()): raise ValueError('Duplicate member stems')
                    archives[asset]=archive;streams[asset]=stream;indices[asset]=index
                    details[asset]=dict(url=url,official_archive_length_bytes=size,headers=headers,head_wall_s=time.monotonic()-t,
                        directory_sha256=sha(dest/(asset+'_directory.json')),members=len(directory),constructed_subset=True)
                common=sorted(set.intersection(*(set(i) for i in indices.values())),key=lambda n:float(n.rsplit('_',1)[1]))
                chosen=np.rint(np.linspace(0,len(common)-1,16)).astype(int)
                if len(set(chosen))!=16: raise ValueError('Insufficient exact timestamp pairs')
                names=[common[i] for i in chosen]
                capture_selection=dict(capture=capture,common_pairs=len(common),chosen_indices=chosen.tolist(),source_ids=names,
                    saved_before_rgb_depth_calibration_member_reads=True,original_directories=details,queries=fixed)
                write(dest/'selection.json',capture_selection)
                files={}
                for asset,archive in archives.items():
                    with zipfile.ZipFile(dest/(asset+'.zip'),'w',compression=zipfile.ZIP_STORED) as subset:
                        member_rows=[]
                        for name in names:
                            budget(); member=indices[asset][name]; payload=archive.read(member)
                            subset.writestr(member,payload); files[asset,name]=payload
                            member_rows.append(dict(name=member,crc32=archive.getinfo(member).CRC,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
                    details[asset].update(subset_zip_sha256=sha(dest/(asset+'.zip')),selected_members=member_rows)
                acquisition=dict(status='COMPLETE',source_kind='Constructed ZIP subset; NOT whole official archive',rows=[dict(name=a+'.zip',sha256=sha(dest/(a+'.zip'))) for a in archives],original_archives=details,
                    selection_sha256=sha(dest/'selection.json'),zip_crc_checked=True,HTTP206_exact_ContentRange_checked=True)
                write(dest/'download_receipt.json',acquisition)
            finally:
                for a in archives.values(): a.close()
                for s in streams.values(): s.close()
            sensor_rows=[];checks=0
            for frame,name in enumerate(names):
                budget(); rgb_bytes=files['lowres_wide',name]; depth_bytes=files['lowres_depth',name]; kb=files['lowres_wide_intrinsics',name]
                rgb=Image.open(io.BytesIO(rgb_bytes)).convert('RGB'); raw=np.asarray(Image.open(io.BytesIO(depth_bytes)))
                w,h,fx,fy,cx,cy=np.fromstring(kb.decode(),sep=' '); k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float64)
                if (w,h)!=(256,192) or rgb.size!=(256,192) or raw.shape!=(192,256) or raw.dtype!=np.uint16: raise ValueError('Native registered asset size/type mismatch')
                if not np.isfinite(k).all() or min(fx,fy)<=0: raise ValueError('Invalid public K')
                depth=optical_z(raw,1000.); my,mx=np.indices(depth.shape,dtype=np.float32); observed=np.ones(depth.shape,bool)
                labels=[];states=[]
                for q in fixed:
                    label,state=sensor_labels(depth,k,q,observed); independent,domain=independent_xyz_labels(depth,k,q,observed)
                    if not np.array_equal(label,independent): raise ValueError('Independent XYZ mismatch')
                    known=state['positive_pixels']+state['free_ray_pixels'];reach=state['domain_pixels']
                    state.update(observed_reachable_rays=int((domain&observed).sum()),reachable_ray_fraction=reach/label.size,known_reference_fraction=known/reach if reach else None)
                    labels.append(label);states.append(state);checks+=1
                rp=sensor/(name+'.png');rp.write_bytes(rgb_bytes);ref=sensor/(name+'.npz')
                np.savez_compressed(ref,labels=np.stack(labels),depth=depth,depth_K=k,color_K=k,map_x=mx,map_y=my,observed=observed)
                row=dict(environment='arkitscenes_'+capture,scan='arkitscenes_'+capture,split='validation',official_split='Training',frame=frame,source_id=name,timestamp_s=float(name.rsplit('_',1)[1]),
                    rgb_path=str(rp),rgb_sha256=sha(rp),reference_path=str(ref),reference_sha256=sha(ref),source_rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest(),source_depth_sha256=hashlib.sha256(depth_bytes).hexdigest(),
                    calibration_sha256=hashlib.sha256(kb).hexdigest(),color_K=k.tolist(),depth_K=k.tolist(),color_shape=[192,256],depth_shape=[192,256],depth_shift=1000.,queries=states)
                sensor_rows.append(row)
            counts=dict(Counter(q['state'] for r in sensor_rows for q in r['queries']))
            units={key:sum(q[key] for r in sensor_rows for q in r['queries']) for key in ('positive_pixels','free_ray_pixels','unknown_pixels','domain_pixels')}
            manifest=dict(status='REAL_SENSOR_FIXED_FRAGMENT_READY',rows=sensor_rows,queries=fixed,frames=len(sensor_rows),groups=[dict(environment='arkitscenes_'+capture,scan='arkitscenes_'+capture,split='validation')],state_counts_by_split={'validation':counts},query_ray_units=units,
                sensor='iPad Pro Apple LiDAR registered official lowres RGB/depth/pincam256x192; axial mm/1000, invalid0 UNKNOWN; not metrology',
                source_alignment_basis='Official DATA.md/raw README/tenFpsDataLoader: same registered lowres pixels, public pincam, axial depth/1000; exact asset timestamp intersection',
                query_contract='Fixed27 camera-local sampled first-return boxes; full-sampled-free does not certify whole volume/body clearance',observation_contract='Only RGB/public K/query; no evaluator truth',
                official_data_role='Two fixed extra official Training captures, consumed Development; same camera family multi-environment, not third hardware or walking events',
                license='Cached official Apple LICENSE non-commercial research; local ignored storage, no redistribution',source_kind='Constructed fixed subset ZIPs, not entire official packages',selection_sha256=sha(dest/'selection.json'))
            write(sensor/'dataset_manifest.json',manifest)
            obs=dict(rows=[{k:r[k] for k in PUBLIC} for r in sensor_rows],queries=fixed,dataset_manifest_sha256=sha(sensor/'dataset_manifest.json'))
            if any(set(r)!=set(PUBLIC) for r in obs['rows']): raise ValueError('Evaluator field leak')
            write(sensor/'observations.json',obs)
            preparation=dict(status='COMPLETE',frames=len(sensor_rows),query_cells=27*len(sensor_rows),independent_xyz_checks=checks,query_states=counts,query_ray_units=units,
                exact_timestamp_rgb_depth_intrinsics=True,observations_evaluator_fields_absent=True,manifest_sha256=sha(sensor/'dataset_manifest.json'),observations_sha256=sha(sensor/'observations.json'),source_acquisition_sha256=sha(dest/'download_receipt.json'),depthpro_status='NOT_RUN',gpu_s=0)
            write(sensor/'preparation_receipt.json',preparation)
            receipt['captured_sources'].append(dict(capture=capture,sensor_path=str(sensor),**preparation));receipt['frames']+=len(sensor_rows)
        receipt['status']='COMPLETE_ADDITIONAL_SAME_CAMERA_REFERENCE'
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL_ADDITIONAL_REFERENCE',error=repr(exc));raise
    finally:
        receipt.update(wall_s=time.monotonic()-start,cpu_process_s=time.process_time()-cpu_start,downloaded_range_bytes=sum(r['bytes'] for r in records),network_request_wall_s=sum(r['wall_s'] for r in records),network_range_requests=records)
        write(source/'completion_receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='network_range_requests'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--network-s',type=float,default=710)
    a=p.parse_args();run(a.source.resolve(),a.output.resolve(),a.network_s)
