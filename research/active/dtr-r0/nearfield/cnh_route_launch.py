"""Task-owned worker launcher for CNH UE capture; no shared asset mutation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def configured_path(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError('Required configured worker environment is absent: ' + name)
    return Path(value).resolve()


def prepare(spec_path, output, timeout):
    from cnh_route_capture import validate_spec
    if not 0 < timeout <= 86400:
        raise ValueError('Timeout must be positive and no greater than the 24-hour render budget')
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    validate_spec(spec)
    artifact_root = configured_path('BLINDASSIST_ARTIFACTS')
    if output == artifact_root or not output.is_relative_to(artifact_root):
        raise ValueError('Output must stay strictly below the configured worker artifact tree')
    if output.exists():
        raise FileExistsError('Never overwrite prior capture: ' + str(output))
    engine = configured_path('UE_ENGINE_ROOT')
    project = configured_path('BLINDASSIST_UE_DEVELOPMENT_PROJECT')
    if project.is_dir():
        candidates = list(project.glob('*.uproject'))
        if len(candidates) != 1:
            raise ValueError('Configured project directory must contain one .uproject')
        project = candidates[0]
    plugin = configured_path('BLINDASSIST_UE_CAPTURE_PLUGIN')
    editor = engine/'Engine/Binaries/Win64/UnrealEditor.exe'
    plugin_binary = plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll'
    for path in (project, plugin, editor, plugin_binary):
        if not path.is_file():
            raise FileNotFoundError(path)
    import psutil
    occupied = [(p.pid, p.info.get('name')) for p in psutil.process_iter(['name'])
                if (p.info.get('name') or '').lower().startswith('unrealeditor')]
    if occupied:
        raise RuntimeError('Shared worker editor already active; leave it untouched: ' + str(occupied))
    output.mkdir(parents=True)
    snapshot = output/'source'; snapshot.mkdir()
    own = Path(__file__).resolve().parent
    names = ('cnh_route_launch.py', 'cnh_route_capture.py', 'ue_capture_readiness.py',
             'ue_pair_export.py', 'street_process_lifecycle.py')
    for name in names:
        shutil.copy2(own/name, snapshot/name)
    shutil.copy2(spec_path, snapshot/'spec.json')
    script = snapshot/'cnh_route_capture.py'
    env = dict(os.environ, BA_CNH_CAPTURE_SPEC=str(snapshot/'spec.json'),
               BA_CNH_CAPTURE_OUTPUT=str(output), PYTHONDONTWRITEBYTECODE='1')
    command = [str(editor), str(project), '-RenderOffscreen', '-unattended', '-nosound', '-nop4', '-NoSplash',
        '-PLUGIN='+str(plugin), '-EnablePlugins=PythonScriptPlugin,BlindAssistCapture,ProceduralMeshComponent',
        '-ExecCmds=py '+script.as_posix(), '-abslog='+str(output/'editor.log'),
        '-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=',
        '-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
    write(output/'launch.json', dict(command=command, timeout_sec=timeout,
        source_hashes={p.name: sha(p) for p in snapshot.iterdir()},
        project_sha256=sha(project), plugin_binary_sha256=sha(plugin_binary),
        frames=len(spec['frames']), python_executable=sys.executable,
        inherited_ddc=os.environ.get('UE-LocalDataCachePath'),
        rendering_backend='UE_OFFSCREEN_D3D_GPU_REQUESTED_ACTUAL_BACKEND_IN_EDITOR_LOG',
        no_shared_asset_save=True))
    return command, env, output, snapshot


def launch(spec_path, output, timeout):
    command, env, output, snapshot = prepare(spec_path, output, timeout)
    sys.path.insert(0, str(snapshot))
    from street_process_lifecycle import TaskProcessTree
    started = time.monotonic()
    terminal = dict(status='RUNNING', expected_frames=json.loads((output/'launch.json').read_text())['frames'])
    try:
        startup = subprocess.STARTUPINFO() if os.name == 'nt' else None
        if startup is not None:
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
        with (output/'process.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, startupinfo=startup,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            tree = TaskProcessTree(process, owner=str(output))
            try:
                code = tree.wait(timeout=timeout)
                if code:
                    raise RuntimeError('Owned UE process failed with exit ' + str(code))
            finally:
                release = tree.cleanup()
                release['wall_elapsed_s'] = time.monotonic()-started
                write(output/'process-release.json', release)
                if not release['released']:
                    raise RuntimeError('Task process-tree release incomplete')
        terminal['editor_wall_s'] = time.monotonic()-started
        if (output/'startup-failure.json').exists():
            raise RuntimeError('UE startup failed; inspect startup-failure.json')
        from cnh_route_capture import finalize
        before = time.monotonic()
        result = finalize(output)
        if result['frame_count'] != terminal['expected_frames']:
            raise RuntimeError('Capture frame count differs from immutable spec')
        terminal.update(status='PASS_CAPTURE_FORMAT_REQUIRES_ENGINEERING_CANARY',
                        frame_count=result['frame_count'], finalize_wall_s=time.monotonic()-before,
                        format_receipt_sha256=sha(output/'format-receipt.json'))
    except BaseException:
        terminal.update(status='FAIL', error=traceback.format_exc())
        raise
    finally:
        terminal['total_wall_s'] = time.monotonic()-started
        write(output/'terminal.json', terminal)
    return terminal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-sec', type=float, default=600)
    args = parser.parse_args()
    print(json.dumps(launch(args.spec, args.output, args.timeout_sec)))
