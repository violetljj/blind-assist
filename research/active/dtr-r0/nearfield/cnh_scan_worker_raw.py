"""Recover frozen worker Street raw128 at original source; no fitting/scoring."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from cnh_qg1_response_decomposition import raw_stage, COMPONENTS
from cnh_route_sensor import SensorParameters
from cnh_street_development_baseline import sample_depth
from cnh_rgb_visible_depth_audit import load_scene_depth
from cnh_street_e2e_materialize import frame_identity

RAW_MANIFEST_SHA='25ced0da709c149cd0d78db3fbe50115aa5a8165fca8ebf68c69e8594b97f758'
MATERIAL_MANIFEST_SHA='f093a5ed14ba9efb1002e642f33e2ed309c09f5fb0d2808cf2248079a8a5540e'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def run(bundle,capture,output):
    started=time.monotonic();binding=read(bundle/'binding.json')
    if not binding.get('protocol_revision'):raise ValueError('Protocol must be committed before synthesis')
    for name,expected in binding['files'].items():
        p=(bundle/name).resolve()
        if not p.is_relative_to(bundle.resolve()) or sha(p)!=expected:raise ValueError('Bundle differs: '+name)
    if sha(Path(__file__))!=binding['files']['code/'+Path(__file__).name]:raise ValueError('Runner differs')
    if sha(capture/'raw-manifest.json')!=RAW_MANIFEST_SHA:raise ValueError('Frozen raw source differs')
    material=bundle/'inputs/worker'
    if sha(material/'manifest.json')!=MATERIAL_MANIFEST_SHA:raise ValueError('Frozen material source differs')
    rows=read(material/'manifest.json')['frames']; raw_rows=read(capture/'raw-manifest.json')['frames']
    if len(rows)!=960 or len(raw_rows)!=960 or any(r.get('data_role')!='Development' for r in rows+raw_rows):
        raise ValueError('Only original 960 Development frames allowed')
    by_id={r['id']:r for r in raw_rows}
    keys=np.asarray([r['frame_key'] for r in rows])
    with np.load(material/'observations.npz',allow_pickle=False) as obs:
        if not np.array_equal(keys,obs['frame_key']):raise ValueError('Observation order differs')
        h3=obs['histogram'].reshape(960,64,16).copy()
    output.mkdir(parents=True,exist_ok=False)
    raw=np.lib.format.open_memmap(output/'worker-raw128.npy',mode='w+',dtype=np.float64,shape=(960,64,128))
    try:
        seeds=[]
        for i,row in enumerate(rows):
            if time.monotonic()-started>1800:raise TimeoutError('Worker raw recovery 1800s budget')
            source=by_id[row['id']]
            if any(source[k]!=row[k] for k in ('layout_id','clip_id','pose_index','nominal_time_s')):raise ValueError('Trajectory identity differs')
            folder=(capture/source['folder']).resolve()
            if not folder.is_relative_to(capture.resolve()):raise ValueError('Source escape')
            frame=dict(original_files={name:dict(path=str(folder/name),sha256=row[field]) for name,field in [('camera.json','camera_sha256'),('depth_left.exr','depth_sha256'),('depth_left_valid.npy','depth_valid_sha256')]})
            key,seed=frame_identity(RAW_MANIFEST_SHA,row,row['camera_sha256'],row['depth_sha256'])
            if key!=row['frame_key'] or seed!=row['seed']:raise ValueError('Original seed/key differs')
            depth,camera=load_scene_depth(frame)
            if not np.allclose(camera['T_camera_tof'],np.eye(4),atol=1e-12):raise ValueError('Sensor alignment differs')
            radial,area=sample_depth(depth,camera,16)
            raw[i]=raw_stage(radial,area,COMPONENTS,int(seed),SensorParameters()).reshape(64,128)
            if not np.array_equal(raw[i].reshape(64,16,8).sum(-1).astype(np.float32),h3[i]):raise ValueError(f'Original H3 parity failed: {i}')
            seeds.append(int(seed))
            if (i+1)%160==0:print(json.dumps(dict(parity_frames=i+1,total=960)),flush=True)
        raw.flush();del raw
        write(output/'worker-frame-keys.json',dict(frame_keys=keys.tolist(),seeds=seeds))
        result=dict(status='COMPLETE_ORIGINAL_STREET_RAW128',frames=960,parity_frames=960,dtype='float64',shape=[960,64,128],
                    raw_sha256=sha(output/'worker-raw128.npy'),frame_keys_sha256=sha(output/'worker-frame-keys.json'),
                    source_manifest_sha256=RAW_MANIFEST_SHA,materialized_manifest_sha256=MATERIAL_MANIFEST_SHA,
                    binding_sha256=sha(bundle/'binding.json'),binding=binding,no_fit=True,no_new_capture=True,wall_s=time.monotonic()-started)
        write(output/'worker-raw-result.json',result);return result
    except BaseException as error:
        write(output/'failure.json',dict(error=repr(error),wall_s=time.monotonic()-started));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.bundle,a.capture,a.output);print(json.dumps({'status':r['status'],'frames':r['frames']}))
