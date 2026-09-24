"""Run the bounded source collector; rejected prefilters remain terminal evidence."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import cnh_route_insert_launch as common
from cnh_route_source_capture import validate_insertions
from cnh_route_source_compare_adapter import validate_spec as validate_source

HERE=Path(__file__).resolve().parent


def validate(spec):
    validate_source(spec)
    validate_insertions(spec)


def finalize(out):
    import numpy as np
    import OpenEXR
    from PIL import Image
    out=Path(out)
    engine=json.loads((out/'engine-receipt.json').read_text())
    if not engine.get('source_unchanged') or not engine.get('actor_release',{}).get('released'):
        raise ValueError('Native source integrity/release failed: '+str(engine.get('error')))
    if engine['status']=='PREFILTER_REJECTED':
        return dict(status='NOT_ADMITTED_PREFILTER',benchmark_eligible=False,frames=0,
            source_gate='FAIL_PREFILTER_OR_UNRESOLVED',selection=json.loads((out/'candidate-selection.json').read_text()),
            seven_pass='NOT_RUN_PREFILTER_REJECTED',formal_pilot='NOT_ADMITTED')
    if engine['status']!='PASS_NATIVE_TRANSPORT':
        raise ValueError('Source transport failed: '+str(engine.get('error')))
    manifest=json.loads((out/'raw-manifest.json').read_text())
    rows=manifest['frames']; h,w=manifest['rig']['height'],manifest['rig']['width']
    layout_ids={r['layout_id'] for r in rows}
    expected={(l,c,i) for l in layout_ids for c in ('centre','boundary','outside','removed') for i in (0,1)}
    if len(layout_ids)!=2 or len(rows)!=16 or {(r['layout_id'],r['clip_id'],r['pose_index']) for r in rows}!=expected:
        raise ValueError('Expected exactly two layouts and four pairs of endpoints')
    reports=[]
    for row in rows:
        folder=(out/row['folder']).resolve()
        if not folder.is_relative_to(out.resolve()):
            raise ValueError('Frame path outside capture')
        depths={}
        for side in ('left','right'):
            with Image.open(folder/(side+'.png')) as rgb:
                rgb.load()
                if rgb.size!=(w,h): raise ValueError('RGB dimensions differ')
            depth=np.load(folder/f'depth_{side}.transport.npy',allow_pickle=False)
            if depth.dtype!=np.dtype('<f4') or depth.shape!=(h,w): raise ValueError('Depth schema differs')
            valid=np.isfinite(depth)&(depth>0)&(depth<100)
            depths[side]=(depth,valid)
            target=folder/f'depth_{side}.exr'
            if target.exists(): raise FileExistsError(target)
            OpenEXR.File({'compression':OpenEXR.ZIP_COMPRESSION,'type':OpenEXR.scanlineimage},
                {'Z':np.where(valid,depth,np.nan).astype(np.float32)}).write(str(target))
            np.save(folder/f'depth_{side}_valid.npy',valid,allow_pickle=False)
        for kind in ('normal','albedo'):
            values=np.load(folder/f'{kind}_left.transport.npy',allow_pickle=False)
            if values.dtype!=np.dtype('<f4') or values.shape!=(h,w,3): raise ValueError('Attribute schema differs')
            valid=depths['left'][1]&np.isfinite(values).all(-1)
            if kind=='normal': valid &= np.abs(np.linalg.norm(values,axis=-1)-1)<.04
            elif np.any((values[valid]<0)|(values[valid]>1.001)): raise ValueError('Albedo outside range')
            if valid.sum()<.9*depths['left'][1].sum(): raise ValueError('Attribute coverage below 90 percent')
            np.save(folder/f'{kind}_left.npy',values.astype(np.float16),allow_pickle=False)
            np.save(folder/f'{kind}_left_valid.npy',valid,allow_pickle=False)
        depth,valid=depths['left']
        ids=np.where(valid,0,65535).astype(np.uint16)
        owners=np.zeros((h,w),dtype=np.uint8); ambiguous=np.zeros((h,w),dtype=bool)
        for identifier in (1,254):
            isolated=np.load(folder/f'isolated_depth_{identifier}.transport.npy',allow_pickle=False)
            candidate=common.compose_instance_ids(depth,isolated,identifier)
            mask=candidate==identifier
            ambiguous |= candidate==65535
            owners[mask]+=1;ids[mask]=identifier
        ambiguous |= owners>1
        ids[ambiguous]=65535
        Image.fromarray(ids).save(folder/'instance_left.png')
        counts={str(i):int((ids==i).sum()) for i in (0,1,254,65535)}
        if row['target_hidden'] and counts['1']: raise ValueError('Removed target remains visible')
        reports.append(dict(row,instance_pixels=counts,overlapping_inserted_id_pixels=int((owners>1).sum()),
            hashes={p.name:common.file_hash(p) for p in folder.iterdir() if p.is_file()}))
    receipt=dict(status='PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY',benchmark_eligible=False,
        source_gate='NOT_ADMITTED_REQUIRES_GEOMETRY_ECHO_LABEL_AND_PROVENANCE_GATES',
        formal_pilot='NOT_ADMITTED',frame_count=16,frames=reports,
        temporal_authority=manifest['temporal_authority'])
    common.write(out/'format-receipt.json',receipt)
    return receipt


def launch(args):
    original_run=common.run_owned
    original_validate=common.validate_spec
    original_finalize=common.finalize
    def source_run(command,env,out,timeout):
        extras=('cnh_route_source_capture.py','cnh_route_source_compare_adapter.py','cnh_route_source_clearance.py',
                'cnh_route_native_clearance.py','cnh_route_scene_probe.py','cnh_route_source_launch.py',
                'cnh_route_street_static_background.py','cnh_route_city_lod0.py')
        launch_path=out/'launch.json';receipt=json.loads(launch_path.read_text())
        for name in extras:
            dest=out/'source'/name;shutil.copy2(HERE/name,dest)
            receipt['source_hashes'][name]=common.file_hash(dest)
        command=[('-ExecCmds=py '+(out/'source/cnh_route_source_capture.py').as_posix())
                 if arg.startswith('-ExecCmds=py ') else arg for arg in command]
        env=dict(env,BA_CNH_SOURCE_SPEC=env['BA_CNH_INSERT_SPEC'],BA_CNH_SOURCE_OUTPUT=env['BA_CNH_INSERT_OUTPUT'])
        receipt.update(command=command,scope='TWO_LAYOUT_SOURCE_ENGINEERING_NOT_BENCHMARK')
        common.write(launch_path,receipt)
        stop=threading.Event();samples=[];errors=[]
        def sample_gpu():
            while not stop.is_set():
                try:
                    result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    if result.returncode: raise RuntimeError(result.stderr.strip())
                    samples.append(dict(monotonic_s=time.monotonic(),gpus_mib=[[int(v) for v in line.split(',')]
                        for line in result.stdout.strip().splitlines()]))
                except Exception as exc:
                    errors.append(str(exc));break
                stop.wait(1.)
        worker=threading.Thread(target=sample_gpu,daemon=True);worker.start()
        try:
            return original_run(command,env,out,timeout)
        finally:
            stop.set();worker.join(timeout=4.)
            common.write(out/'gpu-memory.json',dict(scope='WHOLE_GPU_OBSERVED_NOT_PROCESS_ALLOCATION',
                cadence_s=1.,samples=samples,errors=errors,worker_released=not worker.is_alive(),
                observed_peak_used_mib=max((g[0] for s in samples for g in s['gpus_mib']),default=None)))
    common.validate_spec=validate;common.finalize=finalize;common.run_owned=source_run
    try:
        return common.launch(args)
    finally:
        common.run_owned=original_run
        common.validate_spec=original_validate
        common.finalize=original_finalize


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','engine','plugin','spec','output','result'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timeout',type=float,default=600)
    print(json.dumps(launch(p.parse_args())))
