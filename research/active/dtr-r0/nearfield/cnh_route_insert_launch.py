"""Bounded inserted-asset transport canary; never formal benchmark admission."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / 'tools'))
from ue_native_capture import REPO, file_hash, owned_output, run_owned, write


def compose_instance_ids(depth, isolated, asset_id, tolerance_m=.001):
    """Resolve inserted visibility against native scene depth; invalid stays UNKNOWN."""
    import numpy as np
    if depth.shape != isolated.shape or depth.ndim != 2:
        raise ValueError('Matching two-dimensional depth arrays required')
    if asset_id not in (1, 254) or tolerance_m != .001:
        raise ValueError('Canary IDs and depth tolerance are fixed')
    valid = np.isfinite(depth) & (depth > 0) & (depth < 100)
    target = np.isfinite(isolated) & (isolated > 0) & (isolated < 100)
    visible = valid & target & (np.abs(depth.astype(float)-isolated.astype(float)) <= tolerance_m)
    ids = np.where(valid, 0, 65535).astype(np.uint16)
    # A target in front of the full-scene surface should have been visible.
    # Preserve this inconsistent observation as UNKNOWN, never background.
    inconsistent = valid & target & (depth.astype(float)-isolated.astype(float) > tolerance_m)
    ids[inconsistent] = 65535
    ids[visible] = asset_id
    return ids


def validate_spec(spec):
    if spec.get('scene_layer') != 'INSERTED_ASSET_ENGINEERING_CANARY' or spec.get('benchmark_eligible') is not False:
        raise ValueError('Only the non-benchmark inserted canary is supported')
    assets = spec.get('assets', [])
    if len(assets) != 2 or {a.get('id') for a in assets} != {1, 254}:
        raise ValueError('Exactly two assets with IDs 1 and 254 required')
    if any(not isinstance(a.get('mesh_asset'), str) or not a['mesh_asset'].startswith('/') for a in assets):
        raise ValueError('Absolute mesh asset paths required')


def finalize(out):
    import numpy as np
    import OpenEXR
    from PIL import Image
    out = Path(out).resolve()
    engine = json.loads((out/'engine-receipt.json').read_text())
    if engine.get('status') != 'PASS_NATIVE_TRANSPORT':
        raise ValueError('Native transport did not pass: '+str(engine.get('error','no engine detail')))
    if engine.get('actor_release', {}).get('released') is not True or engine.get('source_unchanged') is not True:
        raise ValueError('Actor release and source integrity required')
    manifest = json.loads((out/'raw-manifest.json').read_text())
    rows = manifest['frames']
    if len(rows) != 6 or {(r['asset_id'], r['occlusion']) for r in rows} != {(a,c) for a in (1,254) for c in ('visible','occluded','partial')}:
        raise ValueError('Exactly six distinct fixed visibility controls required')
    reports, counts, folders = [], {}, set()
    for row in rows:
        folder = (out/row['folder']).resolve()
        if not folder.is_relative_to(out) or folder == out or folder in folders:
            raise ValueError('Frame folders must be distinct and contained')
        folders.add(folder)
        depth = np.load(folder/'depth_left.transport.npy', allow_pickle=False)
        if depth.ndim != 2 or depth.dtype != np.dtype('<f4'):
            raise ValueError('Float32 axial depth required')
        h,w = depth.shape
        if (h,w) != (manifest['rig']['height'],manifest['rig']['width']):
            raise ValueError('Depth differs from declared rig')
        depths = {}
        for side in ('left','right'):
            with Image.open(folder/(side+'.png')) as image:
                image.load()
                if image.size != (w,h):
                    raise ValueError('RGB dimensions differ')
            d = np.load(folder/('depth_'+side+'.transport.npy'), allow_pickle=False)
            if d.shape != (h,w) or d.dtype != np.dtype('<f4'):
                raise ValueError('Depth transport dimensions/dtype differ')
            valid = np.isfinite(d) & (d > 0) & (d < 100)
            depths[side] = valid
            target = folder/('depth_'+side+'.exr')
            if target.exists():
                raise FileExistsError(target)
            OpenEXR.File({'compression':OpenEXR.ZIP_COMPRESSION,'type':OpenEXR.scanlineimage}, {'Z':np.where(valid,d,np.nan).astype(np.float32)}).write(str(target))
            np.save(folder/('depth_'+side+'_valid.npy'), valid, allow_pickle=False)
        for kind in ('normal','albedo'):
            values = np.load(folder/(kind+'_left.transport.npy'), allow_pickle=False)
            if values.shape != (h,w,3) or values.dtype != np.dtype('<f4'):
                raise ValueError('Attribute dimensions/dtype differ')
            valid = depths['left'] & np.isfinite(values).all(-1)
            if kind == 'normal':
                valid &= np.abs(np.linalg.norm(values,axis=-1)-1) < .04
            elif np.any((values[valid] < 0) | (values[valid] > 1.001)):
                raise ValueError('Albedo outside linear unit interval')
            if valid.sum() < .9*depths['left'].sum():
                raise ValueError('Attribute valid coverage below 90 percent')
            np.save(folder/(kind+'_left.npy'), values.astype(np.float16), allow_pickle=False)
            np.save(folder/(kind+'_left_valid.npy'), valid, allow_pickle=False)
        isolated = np.load(folder/'isolated_depth.transport.npy', allow_pickle=False)
        if isolated.dtype != np.dtype('<f4'):
            raise ValueError('Isolated float32 depth required')
        ids = compose_instance_ids(depth, isolated, row['asset_id'])
        Image.fromarray(ids).save(folder/'instance_left.png')
        count = int((ids == row['asset_id']).sum())
        counts[(row['asset_id'],row['occlusion'])] = count
        reports.append(dict(row, visible_pixels=count, unknown_pixels=int((ids==65535).sum()),
            hashes={p.name:file_hash(p) for p in folder.iterdir() if p.is_file()}))
    for asset in (1,254):
        if not (counts[(asset,'visible')] >= 10 and counts[(asset,'occluded')] == 0 and 0 < counts[(asset,'partial')] < counts[(asset,'visible')]):
            raise ValueError('Visibility controls failed for asset '+str(asset)+': '+str(counts))
    report = dict(status='PASS_INSERTED_ASSET_ID_CANARY', benchmark_eligible=False,
        background_geometry_gate='NOT_ADMITTED', formal_pilot='NOT_ADMITTED', frame_count=6,
        identity_method='ISOLATED_NATIVE_DEPTH_AGREEMENT_1MM', frames=reports)
    write(out/'format-receipt.json',report)
    return report


def release_temp(out, launched):
    """Delete only our verified scratch tree after the owned process is released."""
    temp = out/'temp'
    if not temp.exists():
        return dict(released=True, path=str(temp))
    receipt = out/'process-release.json'
    if launched and (not receipt.is_file() or json.loads(receipt.read_text()).get('released') is not True):
        return dict(released=False, path=str(temp), reason='PROCESS_RELEASE_NOT_CONFIRMED')
    resolved = temp.resolve(strict=True)
    if resolved != out.resolve()/'temp' or not resolved.is_relative_to(out.resolve()):
        return dict(released=False, path=str(temp), reason='TEMP_CONTAINMENT_FAILED')
    for directory, dirs, files in os.walk(temp, followlinks=False):
        for path in [Path(directory), *(Path(directory)/name for name in dirs+files)]:
            metadata = path.lstat()
            if path.is_symlink() or getattr(metadata,'st_file_attributes',0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                return dict(released=False, path=str(temp), reason='REPARSE_POINT_PRESERVED')
    # Workspace Windows deletion policy requires native PowerShell, no -Force.
    env = dict(os.environ, BA_CNH_OWNED_TEMP=str(resolved))
    result = subprocess.run(['pwsh','-NoProfile','-NonInteractive','-Command',
        "Remove-Item -LiteralPath $env:BA_CNH_OWNED_TEMP -Recurse -ErrorAction Stop"],
        env=env, capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    return dict(released=result.returncode == 0 and not temp.exists(), path=str(temp),
                error=result.stderr.strip() if result.returncode else None)


def launch(args):
    import numpy
    import OpenEXR
    import PIL
    from run_city_pcg_capture import artifact_file, cache_service_port
    from run_obstacle_research import engine_root
    journal = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    if not journal or json.loads(Path(journal).read_text(encoding='utf-8-sig')).get('state') != 'running':
        raise RuntimeError('Use governed research-ue execution')
    project = artifact_file(args.project,'Project')
    if project.suffix.lower() != '.uproject':
        raise ValueError('Unreal project descriptor required')
    spec_path = artifact_file(args.spec,'Spec')
    plugin = artifact_file(args.plugin,'Plugin')
    binary = artifact_file(plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll','Plugin DLL')
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig')); validate_spec(spec)
    timeout_limit=3600 if spec.get('scope')=='STREET_DEVELOPMENT_PILOT_NOT_BENCHMARK' else 600
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= timeout_limit:
        raise ValueError(f'Capture timeout must be within {timeout_limit} seconds')
    result = Path(args.result).resolve()
    root = (REPO/'artifacts.local').resolve()
    if not result.is_relative_to(root) or result == root or result.exists():
        raise ValueError('Result must be fresh and under artifact root')
    engine = engine_root(args.engine)
    cache = project.parent/'DerivedDataCache'
    port = cache_service_port(cache)
    import psutil
    occupied = [(p.pid,p.info.get('name')) for p in psutil.process_iter(['name'])
                if (p.info.get('name') or '').lower().startswith('unrealeditor')]
    if occupied:
        raise RuntimeError('Shared UnrealEditor already active; preserved: '+str(occupied))
    out = owned_output(args.output)
    before = file_hash(project)
    start = time.monotonic()
    terminal = dict(status='FAIL', benchmark_eligible=False)
    launched = False
    try:
        source = out/'source'; source.mkdir()
        for name in ('cnh_route_insert_capture.py','cnh_route_derived_assets.py','cnh_route_capture.py','ue_pair_export.py','ue_capture_readiness.py','cnh_route_insert_launch.py'):
            shutil.copy2(HERE/name,source/name)
        shutil.copy2(spec_path,source/'spec.json')
        temp = out/'temp'; temp.mkdir()
        env = dict(os.environ, TEMP=str(temp), TMP=str(temp), BA_CNH_INSERT_SPEC=str(source/'spec.json'),BA_CNH_INSERT_OUTPUT=str(out),PYTHONDONTWRITEBYTECODE='1')
        env['UE-LocalDataCachePath'] = str(cache)
        command = [str(engine/'Engine/Binaries/Win64/UnrealEditor.exe'),str(project),'-ExecCmds=py '+(source/'cnh_route_insert_capture.py').as_posix(),'-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash','-ddc=NoShared','-ini:Engine:[Zen.AutoLaunch]:DesiredPort='+str(port),'-abslog='+str(out/'editor.log'),'-PLUGIN='+str(plugin),'-EnablePlugins=CitySamplePCG,BlindAssistCapture,PythonScriptPlugin,ProceduralMeshComponent','-DisablePlugins=CLionSourceCodeAccess,VisualStudioCodeSourceCodeAccess','-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=','-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
        write(out/'launch.json',dict(command=command,project_sha256=before,plugin_sha256=file_hash(binary),source_hashes={p.name:file_hash(p) for p in source.iterdir()},cache=str(cache),zen_port=port,journal=journal,
            python_dependencies=dict(numpy=numpy.__version__,OpenEXR=OpenEXR.__version__,PIL=PIL.__version__,OpenEXR_path=OpenEXR.__file__)))
        launched = True
        run_owned(command,env,out,args.timeout)
        if file_hash(project) != before:
            raise RuntimeError('Source project modified')
        serial_start = time.monotonic()
        terminal.update(finalize(out))
        terminal['finalize_wall_s'] = time.monotonic()-serial_start
    except BaseException as exc:
        terminal.update(status='FAIL',error=str(exc))
        raise
    finally:
        try:
            cleanup = release_temp(out, launched)
        except Exception as exc:
            cleanup = dict(released=False, error=str(exc), path=str(out/'temp'))
        write(out/'temp-release.json',cleanup)
        terminal['temp_release'] = cleanup
        if not cleanup['released']:
            terminal.update(status='FAIL', cleanup_error='Task temp retained; inspect temp-release.json')
        unchanged = file_hash(project) == before
        write(out/'source-integrity.json',dict(project_unchanged=unchanged, before=before,after=file_hash(project)))
        terminal.update(total_wall_s=time.monotonic()-start,output=str(out),source_project_unchanged=unchanged)
        write(out/'terminal.json',terminal)
        result.parent.mkdir(parents=True,exist_ok=True)
        write(result,terminal)
    return terminal


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('project','engine','plugin','spec','output','result'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=600)
    print(json.dumps(launch(parser.parse_args())))
