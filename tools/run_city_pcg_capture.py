"""Capture an independent City Sample PCG map without saving project assets."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket

from run_obstacle_research import engine_root
from city_render_health import inspect_capture
from ue_native_capture import REPO, file_hash, owned_output, run_owned, validate_capture, write


def artifact_file(path, label):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to((REPO / 'artifacts.local').resolve()):
        raise ValueError(label + ' must be a file under artifacts.local')
    return path


def cache_service_port(cache):
    """Avoid UE's global 8558 service, which another project can replace.

    Probe availability without terminating any listener. UE binds after this
    probe, so launch failure still remains possible if another owner wins a race.
    """
    start = int.from_bytes(hashlib.sha256(str(cache).casefold().encode()).digest()[:4], 'big') % 10000
    for offset in range(64):
        port = 20000 + (start + offset) % 10000
        with socket.socket() as probe:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                continue
        return port
    raise RuntimeError('No available isolated Zen port; existing listeners preserved')


def capture(args):
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('Timeout must be positive and finite')
    project = artifact_file(args.project, 'Project')
    if project.suffix.lower() != '.uproject':
        raise ValueError('Project must be a .uproject descriptor')
    spec_path = artifact_file(args.spec, 'Spec')
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    if not isinstance(spec.get('map_asset'), str) or not spec['map_asset'].startswith('/'):
        raise ValueError('Spec must declare an absolute Unreal map_asset')
    map_path = Path(spec['map_file'])
    if not map_path.is_absolute():
        raise ValueError('Spec map_file must be an absolute path')
    map_path = artifact_file(map_path, 'Map')
    if not map_path.is_relative_to(project.parent) or map_path.suffix.lower() != '.umap':
        raise ValueError('Map must be a .umap inside the selected project')
    if not isinstance(spec.get('cases'), list) or not spec['cases']:
        raise ValueError('Spec must contain nonempty cases')
    if spec.get('schema')=='city-crossregion-capture-v1' and len({c['region_id'] for c in spec['cases']})>1:
        raise ValueError('Cross-region capture requires per-region specs: compile --by-region')
    before = {'project_sha256': file_hash(project), 'map_sha256': file_hash(map_path)}
    if before['map_sha256'] != spec['map_sha256']:
        raise ValueError('Map SHA256 differs from spec')
    plugin = artifact_file(args.plugin, 'Plugin')
    if plugin.name != 'BlindAssistCapture.uplugin':
        raise ValueError('Expected BlindAssistCapture.uplugin')
    binary = artifact_file(plugin.parent / 'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll', 'Plugin DLL')
    engine = engine_root(getattr(args, 'engine', None))
    cache = Path(getattr(args, 'ddc_path', None) or project.parent / 'DerivedDataCache').resolve()
    if not cache.is_relative_to((REPO / 'artifacts.local').resolve()) or (cache.exists() and not cache.is_dir()):
        raise ValueError('DDC path must be a directory under artifacts.local')
    zen_port = cache_service_port(cache)
    source = REPO / 'research/active/dtr-r0/nearfield'
    sources = [source / name for name in ('city_pcg_capture.py', 'ue_pair_export.py', 'ue_capture_readiness.py')]
    if spec.get('export_dependencies'):
        sources.append(source / 'city_pcg_dependencies.py')
    if spec.get('source_floor_grid'):
        sources.append(source / 'city_source_floor.py')
    if spec.get('export_hlod_membership'):
        sources.append(source / 'city_hlod_membership.py')
    if spec.get('export_native_inventory'):
        sources.append(source / 'city_native_inspect.py')
    if spec.get('native_targets'):
        sources.append(source / 'city_native_targets.py')
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    out = owned_output(args.output)
    try:
        snapshot = out / 'source'
        snapshot.mkdir()
        for path in sources:
            shutil.copy2(path, snapshot / path.name)
        shutil.copy2(spec_path, snapshot / 'spec.json')
        script = snapshot / 'city_pcg_capture.py'
        temp = out / 'temp'
        temp.mkdir()
        env = dict(os.environ, BA_CITY_SPEC=str(snapshot / 'spec.json'), BA_CITY_OUT=str(out),
                   PYTHONDONTWRITEBYTECODE='1', TEMP=str(temp), TMP=str(temp))
        env['UE-LocalDataCachePath'] = str(cache)
        command = [str(engine / 'Engine/Binaries/Win64/UnrealEditor.exe'), str(project),
                   '-ExecCmds=py ' + script.as_posix(), '-RenderOffscreen', '-unattended',
                   '-nosound', '-nop4', '-NoSplash', '-ddc=NoShared',
                   '-ini:Engine:[Zen.AutoLaunch]:DesiredPort=' + str(zen_port),
                   '-abslog=' + str(out / 'editor.log'), '-PLUGIN=' + str(plugin),
                   '-EnablePlugins=CitySamplePCG,BlindAssistCapture,PythonScriptPlugin',
                   '-DisablePlugins=CLionSourceCodeAccess,VisualStudioCodeSourceCodeAccess',
                   '-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=',
                   '-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
        write(out / 'launch.json', dict(command=command, project=str(project), map_file=str(map_path),
              persistent_cache=dict(path=env['UE-LocalDataCachePath'], zen_port=zen_port,
                                    policy='REUSE_ACROSS_RUNS_KEEP_WHEN_RELEASING_PROCESSES'),
              input_hashes=before, spec_sha256=file_hash(snapshot / 'spec.json'),
              source_hashes={path.name: file_hash(snapshot / path.name) for path in sources},
              launcher_sha256=file_hash(Path(__file__)), plugin_sha256=file_hash(plugin),
              plugin_binary_sha256=file_hash(binary)))
        run_owned(command, env, out, args.timeout)
        receipt = json.loads((out / 'receipt.json').read_text(encoding='utf-8'))
        write(out / 'engine-receipt.json', receipt)
        after = {'project_sha256': file_hash(project), 'map_sha256': file_hash(map_path)}
        write(out / 'source-integrity.json', dict(before=before, after=after, unchanged=before == after))
        if before != after:
            raise RuntimeError('Source project or map changed during capture')
        render_health = inspect_capture(out)
        write(out / 'render-resource-health.json', render_health)
        if not render_health['ready_data_eligible'] and spec.get('scope') != 'SOURCE_ONLY_RECONNAISSANCE_NOT_RESEARCH_COHORT':
            raise RuntimeError('Missing Nanite/VT resources: capture excluded from ready data; see render-resource-health.json')
        validate_capture(out)
    except BaseException as exc:
        after = {key: file_hash(path) if path.is_file() else None for key, path in
                 (('project_sha256', project), ('map_sha256', map_path))}
        write(out / 'source-integrity.json', dict(before=before, after=after, unchanged=before == after))
        write(out / 'completion.json', dict(status='FAIL', error=str(exc)))
        raise
    print(json.dumps(dict(status='PASS', output=str(out), frames=receipt['frame_count'])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'spec', 'plugin', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--ddc-path', type=Path, help='Explicit persistent cache directory under artifacts.local')
    capture(parser.parse_args())


if __name__ == '__main__':
    main()
