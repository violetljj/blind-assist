"""Owned one-editor unsaved derived-material switch probe; no map or capture."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[3]/'tools'))
from ue_native_capture import REPO,file_hash,owned_output,run_owned,write


def launch(args):
    from run_city_pcg_capture import cache_service_port
    import psutil

    project=args.project.resolve()
    engine=args.engine.resolve()
    spec=args.spec.resolve()
    if not project.is_file() or project.suffix.lower()!='.uproject':
        raise ValueError('Existing UE project descriptor required')
    if not (engine/'Engine/Binaries/Win64/UnrealEditor.exe').is_file():
        raise ValueError('UE editor unavailable')
    assets=json.loads(spec.read_text(encoding='utf-8'))['assets']
    if len(assets)!=15 or len({(a['role'],a['key']) for a in assets})!=15:
        raise ValueError('Expected fifteen distinct used source assets')
    protected=[project,*[Path(p) for p in json.loads(spec.read_text(encoding='utf-8'))['protected_files']]]
    before={str(p):file_hash(p) for p in protected}
    occupied=[(p.pid,p.info.get('name')) for p in psutil.process_iter(['name'])
              if (p.info.get('name') or '').lower().startswith('unrealeditor')]
    if occupied:raise RuntimeError('Shared UnrealEditor already active: '+repr(occupied))
    out=owned_output(args.output)
    source=out/'source';source.mkdir()
    shutil.copy2(HERE/'cnh_alley_mfpd_probe.py',source/'cnh_alley_mfpd_probe.py')
    shutil.copy2(HERE/'cnh_route_derived_assets.py',source/'cnh_route_derived_assets.py')
    shutil.copy2(spec,source/'spec.json')
    temp=out/'temp';temp.mkdir()
    cache=project.parent/'DerivedDataCache'
    env=dict(os.environ,TEMP=str(temp),TMP=str(temp),PYTHONDONTWRITEBYTECODE='1',
             BA_CNH_MFPD_SPEC=str(source/'spec.json'),BA_CNH_MFPD_OUTPUT=str(out))
    env['UE-LocalDataCachePath']=str(cache)
    command=[str(engine/'Engine/Binaries/Win64/UnrealEditor.exe'),str(project),
        '-ExecCmds=py '+(source/'cnh_alley_mfpd_probe.py').as_posix(),
        '-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash','-ddc=NoShared',
        '-ini:Engine:[Zen.AutoLaunch]:DesiredPort='+str(cache_service_port(cache)),
        '-abslog='+str(out/'editor.log'),
        '-PLUGIN='+str(args.plugin.resolve()),
        '-EnablePlugins=CitySamplePCG,BlindAssistCapture,PythonScriptPlugin,ProceduralMeshComponent',
        '-DisablePlugins=CLionSourceCodeAccess,VisualStudioCodeSourceCodeAccess',
        '-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=',
        '-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
    write(out/'launch.json',dict(command=command,source_hashes={p.name:file_hash(p) for p in source.iterdir()},
        protected_file_hashes=before,scope='UNSAVED_DERIVED_MATERIAL_NO_MAP_OR_FRAME_CAPTURE'))
    error=None
    try:
        run_owned(command,env,out,args.timeout)
        raw=json.loads((out/'mfpd-materials.json').read_text(encoding='utf-8'))
        if raw.get('status') not in ('PASS_ZERO_CROSS_SPLIT_EDITOR_USED_TEXTURES','INCOMPLETE_ISOLATION'):
            raise RuntimeError('Native material probe failed: '+str(raw.get('error')))
        if len(raw['assets'])!=6:raise ValueError('Native insert count differs')
        after={str(p):file_hash(p) for p in protected}
        if after!=before:raise RuntimeError('Project, map, or source mesh bytes changed')
        terminal=dict(status=raw['status'],
                      map_or_frame_capture='NOT_RUN',protected_file_count=len(protected),
                      protected_file_hashes_unchanged=True,native_receipt_sha256=file_hash(out/'mfpd-materials.json'),
                      process_released=True)
    except Exception:
        error=traceback.format_exc()
        terminal=dict(status='FAIL',error=error,map_or_frame_capture='NOT_RUN')
    finally:
        release=out/'process-release.json'
        if release.is_file() and json.loads(release.read_text()).get('released') is True:
            resolved=temp.resolve(strict=True)
            if resolved!=out.resolve()/'temp' or not resolved.is_relative_to(out.resolve()):
                raise RuntimeError('Task temp path escaped output')
            cleanup=subprocess.run(['pwsh','-NoProfile','-NonInteractive','-Command',
                'Remove-Item -LiteralPath $env:BA_CNH_AUDIT_TEMP -Recurse -ErrorAction Stop'],
                env=dict(os.environ,BA_CNH_AUDIT_TEMP=str(resolved)),capture_output=True,text=True)
            terminal['temp_released']=cleanup.returncode==0 and not temp.exists()
            if not terminal['temp_released']:
                terminal['temp_release_error']=cleanup.stderr.strip()
        else:
            terminal['temp_released']=False
            terminal['temp_retained_reason']='Process release not confirmed'
        write(out/'terminal.json',terminal)
        args.result.parent.mkdir(parents=True,exist_ok=True)
        write(args.result,terminal)
    if error:raise RuntimeError(error)
    if not terminal['temp_released']:raise RuntimeError('Task temp retained: '+str(terminal))
    return terminal


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ('project','engine','plugin','spec','output','result'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=360)
    print(json.dumps(launch(parser.parse_args())))
