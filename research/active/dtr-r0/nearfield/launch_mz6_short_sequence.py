"""Launch and verify one owned 200-frame MZ6 capture, without model inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'research/active/dtr-r0/unreal')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,data):
    Path(p).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def verify(out):
    import numpy as np
    from PIL import Image
    import torch
    from multizone64_observation import native_events
    from verify_temporal_structure import rays, intersect
    receipt=json.loads((out/'receipt.json').read_text())
    release=json.loads((out/'process-release.json').read_text())
    manifest=json.loads((out/'model/sensor_manifest.json').read_text())
    spec=json.loads((out/'evaluator/spec.json').read_text())
    assert receipt['status']=='PASS' and receipt['source_unchanged'] and release['released']
    frames=manifest['frames'];assert len(frames)==len(spec['cases'])==200
    assert len({f['clip_id'] for f in frames})==8
    assert set(manifest)=={'calibration','frames'} and manifest['calibration']==spec['calibration']
    assert not list((out/'model').rglob('*.npy'))
    assert len(list((out/'model/sample').glob('*.png')))==len(list((out/'evaluator/native').glob('*.npy')))==200
    rows=[];depths=[]
    for i,(frame,case) in enumerate(zip(frames,spec['cases'])):
        assert set(frame)=={'sample_index','clip_id','rgb_path','nominal_time_s'}
        assert frame['sample_index']==i and frame['clip_id']==case['clip_id']
        assert frame['nominal_time_s']==case['nominal_time_s']
        p=case['camera'];assert p['pitch']==p['roll']==p['yaw']==0. and abs(p['z']-case['floor_z_m']-1.7)<1e-9
        rgb=out/'model'/frame['rgb_path'];native=out/f'evaluator/native/{i:04d}.npy'
        with Image.open(rgb) as im:assert im.size==(640,360) and im.mode in ('RGB','RGBA')
        d=np.load(native,allow_pickle=False);assert d.shape==(360,640) and np.isfinite(d).all()
        direction=rays(p);origin=np.array([p[k] for k in ('x','y','z')])
        object_rows=[]
        for obj in case['objects']:
            expected=intersect(origin,direction,obj)
            visible=np.isfinite(expected)&(d>0)&(np.abs(d-expected)<.03)
            object_rows.append(dict(name=obj['name'],projected_pixels=int(np.isfinite(expected).sum()),
                native_visible_pixels=int(visible.sum())))
            if obj['name']=='target':assert visible.any(),f'Target position unsupported by native capture frame{i}'
        depths.append(d)
        rows.append(dict(sample_index=i,clip_id=frame['clip_id'],rgb_sha256=sha(rgb),native_sha256=sha(native),
            native_valid_fraction=float((d>0).mean()),objects=object_rows))
    assert torch.cuda.is_available()
    for begin in range(0,200,16):
        labels=native_events(torch.from_numpy(np.stack(depths[begin:begin+16])).cuda(),crop=True)
        for j in range(min(16,200-begin)):
            events=labels['events'][j].cpu().tolist();valid=bool(labels['observation_valid'][j])
            rows[begin+j].update(events=events,counts=labels['counts'][j].cpu().tolist(),observation_valid=valid,
                support_status='POSITIVE_SUPPORT' if any(events) else ('NO_POSITIVE_SUPPORT_NOT_CLEAR' if valid else 'UNKNOWN'))
    result=dict(status='PASS',frames=200,clips=8,source_unchanged=True,processes_released=True,
        inference_manifest_excludes_pose_objects_depth=True,sampling=spec['sampling'],
        label_source='native_events crop=True positive pixel support only; no positive bits never imply CLEAR',
        event_order=['BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR'],
        backend=dict(operation='native pixel geometry labels',device=torch.cuda.get_device_name()),
        source_sha256=sha(__file__),spec_sha256=sha(out/'evaluator/spec.json'),rows=rows)
    write(out/'source-verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--spec',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--reuse-cache-from',type=Path);a=p.parse_args()
    artifacts=(ROOT/'artifacts.local').resolve();out=a.output.resolve()
    assert out.is_relative_to(artifacts)
    if a.verify_only:return verify(out)
    from run_obstacle_research import engine_root
    from street_process_lifecycle import TaskProcessTree
    import psutil
    assert not any(proc.info['name'] and proc.info['name'].startswith('UnrealEditor') for proc in psutil.process_iter(['name'])),'Host editor occupied'
    spec=a.spec.resolve();assert spec.is_relative_to(artifacts) and spec.is_file() and not out.exists()
    source=Path(__file__).with_name('capture_mz6_short_sequence.py')
    project=ROOT/'artifacts.local/unreal/BlindAssistStreetLab'
    mapfile=project/'Content/StreetLab/WillowSampleV1.umap'
    assert sha(mapfile)==json.loads(spec.read_text())['map_sha256']
    out.mkdir(parents=True);snapshot=out/'source';snapshot.mkdir()
    for f in (source,Path(__file__),Path(__file__).with_name('mz6_short_sequence_spec.py'),
              Path(__file__).with_name('ue_capture_readiness.py'),spec):shutil.copy2(f,snapshot/f.name)
    reuse=None
    if a.reuse_cache_from:
        previous=a.reuse_cache_from.resolve()
        assert previous.parent==out.parent and previous!=out
        prior=json.loads((previous/'launch.json').read_text())
        assert json.loads((previous/'process-release.json').read_text())['released']
        assert prior['spec_sha256']==sha(spec) and prior['map_sha256']==sha(mapfile)
        cache=Path(prior['cache']).resolve();assert cache==previous/'cache' and cache.is_dir()
        reuse=dict(owner_capture=str(previous),launch_sha256=sha(previous/'launch.json'),release_sha256=sha(previous/'process-release.json'))
    else:
        cache=out/'cache';cache.mkdir()
    temp=out/'temp';temp.mkdir()
    plugin=ROOT/'artifacts.local/ue-hlod-b2/package/BlindAssistCapture.uplugin'
    binary=plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll'
    assert sha(binary)=='0fe8fde21ba06f248434e2402eba7577235aa456b5c3d49efe0d25c3c7557939'
    # UE's Zen command conversion uses signed16; never choose an ephemeral >32767 port.
    start=int.from_bytes(hashlib.sha256(str(cache).encode()).digest()[:4],'big')%10000
    for offset in range(64):
        port=20000+(start+offset)%10000
        with socket.socket() as sock:
            if hasattr(socket,'SO_EXCLUSIVEADDRUSE'):
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            try:sock.bind(('127.0.0.1',port))
            except OSError:continue
        break
    else:raise RuntimeError('No available isolated Zen port')
    env=dict(os.environ,BA_NEARFIELD_SPEC=str(spec),BA_NEARFIELD_OUTPUT=str(out),TEMP=str(temp),TMP=str(temp),PYTHONDONTWRITEBYTECODE='1')
    env['UE-LocalDataCachePath']=str(cache)
    command=[str(engine_root()/'Engine/Binaries/Win64/UnrealEditor.exe'),str(project/'BlindAssistStreetLab.uproject'),
        '-ExecCmds=py '+(snapshot/source.name).as_posix(),'-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash','-ddc=NoShared',
        '-ini:Engine:[Zen.AutoLaunch]:DesiredPort='+str(port),'-abslog='+str(out/'editor.log'),
        '-PLUGIN='+str(plugin),'-EnablePlugins=BlindAssistCapture,PythonScriptPlugin',
        '-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=',
        '-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
    write(out/'launch.json',dict(command=command,spec_sha256=sha(spec),map_sha256=sha(mapfile),
        source_hashes={f.name:sha(f) for f in snapshot.iterdir()},cache=str(cache),zen_port=port,
        reused_task_cache=reuse,readiness_plugin_sha256=sha(plugin),readiness_binary_sha256=sha(binary)))
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    proc=subprocess.Popen(command,env=env,startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
    tree=TaskProcessTree(proc,owner=str(out))
    try:
        code=tree.wait(timeout=900)
        assert code==0,'Editor failed; preserve attempt'
    finally:
        release=tree.cleanup(ports=[('127.0.0.1',port)])
        write(out/'process-release.json',release)
        assert release['released'],'Owned resource release incomplete'
    verify(out)


if __name__=='__main__':main()
