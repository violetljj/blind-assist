"""Launch the one frozen 48-clip/576-frame local-stability UE capture.

The protocol seals the spec plus code_hashes (nearfield filenames) and
input_hashes (repository-relative paths). No source, budget, cohort or
output-overwrite override.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

EXPECTED_FRAMES = 576
EXPECTED_CLIPS = 48
FRAMES_PER_CLIP = 12
CAPTURE_BUDGET_SECONDS = 900
EXPECTED_MAP_SHA = 'cf35e5c9df54cd0f781f09ea8105fe8ef6078ed0822d4e594d64216e79a254fb'
MANDATORY_CODE = {
    'local_stability_capture.py', 'launch_local_stability.py', 'local_stability_spec.py',
    'ue_capture_readiness.py',
    'core_transfer_spec.py', 'query_occupancy_data.py', 'query_occupancy_spec.py',
    'local_transfer_spec.py', 'data_coverage_spec.py', 'spatial_bce_spec.py',
    'ba_camera_corridor_spec.py',
    'LOCAL_STABILITY_PROTOCOL_20260923.md',
}
MANDATORY_INPUTS = {
    'tools/run_obstacle_research.py',
    'research/active/dtr-r0/unreal/street_process_lifecycle.py',
    'artifacts.local/evidence/ba-inherit-spatial-20260922-run/raw.pkl',
    'artifacts.local/evidence/ba-inherit-spatial-20260922-run/local.pkl',
    'artifacts.local/evidence/ba-inherit-spatial-20260922-run/selection.json',
    'artifacts.local/evidence/ba-inherit-spatial-20260922-run/model-seal.json',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_capture_output(out, artifact_root):
    """Admit a new directory or the active runtime's exact empty output parent."""
    out, artifact_root = Path(out).resolve(), Path(artifact_root).resolve()
    assert out != artifact_root and out.is_relative_to(artifact_root), 'Canonical capture output required'
    if not out.exists():
        return dict(mode='NEW_DIRECTORY')
    assert out.is_dir() and next(out.iterdir(), None) is None, 'Existing capture output must be completely empty'
    journal_name = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal_name, 'Existing output requires an active governed run journal'
    journal_path = Path(journal_name).resolve()
    assert journal_path.is_relative_to(artifact_root) and journal_path.is_file(), 'Governed journal outside artifact root'
    journal = json.loads(journal_path.read_text(encoding='utf-8-sig'))
    assert journal.get('schema') == 'blindassist-asset-run-journal-v1', 'Unexpected governed journal schema'
    assert journal.get('state') == 'running', 'Governed output owner is not running'
    matching = []
    for row in journal.get('outputs', []):
        declared = Path(row['path'])
        declared = (declared if declared.is_absolute() else artifact_root/declared).resolve()
        if declared.parent == out and declared.name == 'launcher-terminal.json':
            matching.append(row['alias'])
    assert matching, 'Empty directory is not this governed run output parent'
    command = journal.get('command', [])
    assert command.count('--output') == 1 and command.index('--output') + 1 < len(command), 'Missing governed output command'
    assert Path(command[command.index('--output') + 1]).resolve() == out, 'Governed command owns a different output'
    return dict(mode='GOVERNED_EMPTY_DIRECTORY', journal=str(journal_path),
                run_id=journal.get('id'), output_aliases=matching)


def verify_protocol(protocol, spec, repo):
    data = json.loads(protocol.read_text(encoding='utf-8-sig'))
    assert data['spec_sha256'] == sha(spec), 'Frozen spec mismatch'
    assert data['frames'] == EXPECTED_FRAMES and data['clips'] == EXPECTED_CLIPS
    assert data['capture_timeout_s'] == CAPTURE_BUDGET_SECONDS
    codes, inputs = data['code_hashes'], data['input_hashes']
    assert isinstance(codes, dict) and isinstance(inputs, dict)
    assert MANDATORY_CODE <= set(codes), 'Capture/launcher/readiness code must be frozen'
    assert MANDATORY_INPUTS <= set(inputs), 'Engine-location/process-lifecycle code must be frozen'
    nearfield = Path(__file__).resolve().parent
    artifact_root = (repo/'artifacts.local').resolve()
    resolved = {}
    for base, files in ((nearfield, codes), (repo, inputs)):
        for name, digest in files.items():
            assert not Path(name).is_absolute(), 'Protocol paths must be relative'
            path = (base/name).resolve()
            assert (path.is_relative_to(base.resolve()) or
                    (base == repo and path.is_relative_to(artifact_root))), 'Frozen path outside permitted roots'
            assert path.is_file() and sha(path) == digest, 'Frozen input mismatch: '+str(path)
            assert path not in resolved or resolved[path] == digest, 'Conflicting frozen path'
            resolved[path] = digest
    return resolved


def validate_cases(data):
    from local_stability_spec import check_spec
    check_spec(data)
    cases = data['cases']
    assert data['expected_map_sha256'] == EXPECTED_MAP_SHA
    assert len(cases) == EXPECTED_FRAMES
    assert len({c['name'] for c in cases}) == EXPECTED_FRAMES
    assert len({c['clip_id'] for c in cases}) == EXPECTED_CLIPS
    for offset in range(0, EXPECTED_FRAMES, FRAMES_PER_CLIP):
        clip = cases[offset:offset+FRAMES_PER_CLIP]
        assert len({c['clip_id'] for c in clip}) == 1
        assert [c['frame_in_clip'] for c in clip] == list(range(FRAMES_PER_CLIP))
        assert all(clip[i]['time_s'] < clip[i+1]['time_s'] for i in range(FRAMES_PER_CLIP-1))
        assert all(c['objects'] == clip[0]['objects'] for c in clip)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for option in ('spec', 'protocol', 'output'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--plugin', type=Path, default=os.environ.get('BLINDASSIST_UE_CAPTURE_PLUGIN',
        str(Path(__file__).resolve().parents[4]/'artifacts.local/ue-hlod-b2/package/BlindAssistCapture.uplugin')))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    sys.path[:0] = [str(repo/'tools'), str(repo/'research/active/dtr-r0/unreal')]
    from run_obstacle_research import engine_root
    from street_process_lifecycle import TaskProcessTree
    root = (repo/'artifacts.local').resolve()
    spec, protocol, out = (p.resolve() for p in (args.spec, args.protocol, args.output))
    assert all(p.is_relative_to(root) and p.is_file() for p in (spec, protocol))
    validate_capture_output(out, root)
    capture_script = Path(__file__).with_name('local_stability_capture.py').resolve()
    helper = capture_script.with_name('ue_capture_readiness.py')
    frozen = verify_protocol(protocol, spec, repo)
    data = json.loads(spec.read_text(encoding='utf-8-sig'))
    validate_cases(data)
    project = repo/'artifacts.local/unreal/BlindAssistStreetLab'
    engine = engine_root()
    map_file = project/'Content/StreetLab/WillowSampleV1.umap'
    assert sha(map_file) == EXPECTED_MAP_SHA
    plugin = args.plugin.resolve()
    assert plugin.name == 'BlindAssistCapture.uplugin' and plugin.is_file()
    plugin_binary = plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll'
    assert plugin_binary.is_file()
    assets = {}
    for case in data['cases']:
        for obj in case['objects']:
            assert obj['kind'] == 'cube', 'Composite truth requires exact cube meshes'
            mesh = 'Cube'
            paths = [engine/f'Engine/Content/BasicShapes/{mesh}.uasset',
                     project/('Content/'+obj['material'].removeprefix('/Game/')+'.uasset')]
            for path in paths:
                assets[str(path)] = sha(path)
    source = dict(spec_sha256=sha(spec), protocol_sha256=sha(protocol),
        frozen_files={str(p): h for p, h in frozen.items()}, capture_script_sha256=sha(capture_script),
        launcher_sha256=sha(__file__), map_sha256=sha(map_file),
        uproject_sha256=sha(project/'BlindAssistStreetLab.uproject'), readiness_helper_sha256=sha(helper),
        plugin_path=str(plugin), plugin_sha256=sha(plugin), plugin_binary_sha256=sha(plugin_binary),
        assets=assets, engine=str(engine), project=str(project), sampling=data['sampling'],
        frame_count=EXPECTED_FRAMES, clip_count=EXPECTED_CLIPS, budget_seconds=CAPTURE_BUDGET_SECONDS)
    source['output_directory_admission'] = validate_capture_output(out, root)
    out.mkdir(parents=True, exist_ok=source['output_directory_admission']['mode'] == 'GOVERNED_EMPTY_DIRECTORY')
    assert next(out.iterdir(), None) is None, 'Capture output changed before initialization'
    write(out/'launch-receipt.json', source)
    env = dict(os.environ, BA_LOCAL_STABILITY_SPEC=str(spec), BA_LOCAL_STABILITY_OUTPUT=str(out),
        BA_LOCAL_STABILITY_SCRIPT=str(capture_script), BA_LOCAL_STABILITY_PROTOCOL_SHA=source['protocol_sha256'],
        BA_LOCAL_STABILITY_SPEC_SHA=source['spec_sha256'], BA_LOCAL_STABILITY_SCRIPT_SHA=source['capture_script_sha256'])
    env['UE-LocalDataCachePath'] = str(project/'DerivedDataCache')
    startup = subprocess.STARTUPINFO() if os.name == 'nt' else None
    if startup is not None:
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    tree = None
    terminal = dict(status='FAIL', budget_seconds=CAPTURE_BUDGET_SECONDS)
    try:
        verify_protocol(protocol, spec, repo)
        assert sha(protocol) == source['protocol_sha256']
        proc = subprocess.Popen([str(engine/'Engine/Binaries/Win64/UnrealEditor.exe'), str(project/'BlindAssistStreetLab.uproject'),
            '-ExecCmds=py '+capture_script.as_posix(), '-RenderOffscreen', '-unattended', '-nosound', '-nop4', '-NoSplash', '-ddc=NoShared',
            '-abslog='+str(out/'editor.log'), '-PLUGIN='+str(plugin)], env=env, startupinfo=startup,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        tree = TaskProcessTree(proc, owner=str(out))
        code = tree.wait(timeout=CAPTURE_BUDGET_SECONDS)
        receipt = json.loads((out/'receipt.json').read_text())
        assert code == 0 and receipt['status'] == 'PASS' and receipt['frame_count'] == EXPECTED_FRAMES
        assert receipt['source_unchanged'] and receipt['task_actors_released']
        assert receipt['readiness_helper_sha256'] == source['readiness_helper_sha256'] == sha(helper)
        assert len(receipt['view_readiness']) == EXPECTED_FRAMES and all(r['status'] == 'READY' for r in receipt['view_readiness'])
        assert sha(plugin) == source['plugin_sha256'] and sha(plugin_binary) == source['plugin_binary_sha256']
        assert receipt['protocol_sha256'] == source['protocol_sha256'] and receipt['spec_sha256'] == source['spec_sha256']
        assert receipt['script_sha256'] == source['capture_script_sha256']
        assert sha(map_file) == source['map_sha256']
        assert sha(project/'BlindAssistStreetLab.uproject') == source['uproject_sha256']
        assert sha(protocol) == source['protocol_sha256']
        verify_protocol(protocol, spec, repo)
        for path, digest in assets.items():
            assert sha(path) == digest
        terminal.update(status='PASS', returncode=code, frame_count=EXPECTED_FRAMES)
        print(json.dumps(terminal), flush=True)
    except BaseException:
        terminal['error'] = traceback.format_exc()
        raise
    finally:
        release = tree.cleanup() if tree is not None else dict(released=True, process_started=False)
        write(out/'process-release.json', release)
        terminal['processes_released'] = release['released']
        if not release['released']:
            terminal['status'] = 'FAIL'
        write(out/'launcher-terminal.json', terminal)
        if not release['released']:
            raise RuntimeError('Owned process release incomplete')


if __name__ == '__main__':
    main()
