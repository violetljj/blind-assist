"""Run existing UE captures with native lossless EXR and bounded NPY conversion.

All generated material stays under artifacts.local. The shared .uproject and saved map are never edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
UE_SOURCE = REPO / 'research/active/dtr-r0/unreal'
CAPTURES = {'whisker': 'capture_whisker.py', 'grounding': 'grounding_capture.py',
            'factorial': 'factorial_capture.py'}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owned_output(path):
    path = path.resolve()
    root = (REPO / 'artifacts.local').resolve()
    if not path.is_relative_to(root) or path == root or path.exists():
        raise ValueError('Use a new output directory strictly under artifacts.local')
    path.mkdir(parents=True)
    return path


def run_owned(command, env, out, timeout, on_poll=None):
    sys.path.insert(0, str(UE_SOURCE))
    from street_process_lifecycle import TaskProcessTree
    startup = subprocess.STARTUPINFO() if os.name == 'nt' else None
    if startup:
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    started = time.monotonic()
    with (out / 'process.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, startupinfo=startup,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        tree = TaskProcessTree(proc, owner=str(out))
        try:
            code = tree.wait(timeout=timeout, on_poll=on_poll)
            if code:
                raise RuntimeError(f'Owned process exited {code}; inspect {out / "process.log"}')
        finally:
            release = tree.cleanup()
            release['wall_elapsed_s'] = time.monotonic()-started
            write(out / 'process-release.json', release)
            if not release['released']:
                raise RuntimeError('Owned process release incomplete')


def capture(args):
    from run_obstacle_research import engine_root
    root = (REPO / 'artifacts.local').resolve()
    spec = args.spec.resolve()
    if not spec.is_relative_to(root) or not spec.is_file():
        raise ValueError('Spec must exist under artifacts.local')
    script = REPO / 'research/active/dtr-r0/nearfield' / CAPTURES[args.capture]
    engine = engine_root(args.engine)
    project = root / 'unreal/BlindAssistStreetLab'
    out = owned_output(args.output)
    # Freeze the actual scripts used by this process against concurrent edits.
    snapshot = out/'source'
    snapshot.mkdir()
    for source in (script, script.with_name('ue_depth_export.py'), script.with_name('ue_exr_transport.py')):
        shutil.copy2(source, snapshot/source.name)
    script = snapshot/script.name
    shutil.copy2(spec, snapshot/'spec.json')
    spec = snapshot/'spec.json'
    env = dict(os.environ, BA_NEARFIELD_SPEC=str(spec), BA_NEARFIELD_OUTPUT=str(out),
               BA_UE_DEPTH_EXPORT=args.depth_export)
    env['UE-LocalDataCachePath'] = str(project / 'DerivedDataCache')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [str(engine / 'Engine/Binaries/Win64/UnrealEditor.exe'),
               str(project / 'BlindAssistStreetLab.uproject'),
               '-ExecCmds=py ' + script.as_posix(), '-RenderOffscreen', '-unattended',
               '-nosound', '-nop4', '-NoSplash', '-ddc=NoShared', '-abslog=' + str(out/'editor.log')]
    write(out / 'launch.json', dict(depth_export=args.depth_export, script=str(script),
          script_sha256=file_hash(script), exporter_sha256=file_hash(script.with_name('ue_depth_export.py')),
          transport_sha256=file_hash(script.with_name('ue_exr_transport.py')),
          spec_sha256=file_hash(spec), command=command))
    transport = None
    try:
        if args.depth_export in ('exr', 'exr_probe'):
            sys.path.insert(0, str(script.parent))
            from ue_exr_transport import Transport
            transport = Transport(out, len(json.loads(spec.read_text(encoding='utf-8-sig'))['cases']),
                                  probe=args.depth_export == 'exr_probe')
        run_owned(command, env, out, args.timeout, transport.pump if transport else None)
        receipt = json.loads((out/'receipt.json').read_text(encoding='utf-8'))
        write(out/'engine-receipt.json', receipt)
        if receipt['status'] != 'PASS' or not receipt['source_unchanged']:
            raise RuntimeError('Capture failed; inspect receipt.json')
        if transport:
            conversion = transport.finish()
            write(out/'transport.json', conversion)
    except BaseException as exc:
        if (out/'receipt.json').is_file():
            failed = json.loads((out/'receipt.json').read_text(encoding='utf-8'))
            if not (out/'engine-receipt.json').exists():
                write(out/'engine-receipt.json', failed)
            failed.update(status='FAIL', transport_error=str(exc))
            write(out/'receipt.json', failed)
        write(out/'completion.json', dict(status='FAIL', error=str(exc)))
        raise
    finally:
        if transport:
            transport.close()
    write(out/'completion.json', dict(status='PASS', depth_export=args.depth_export,
          frames=receipt['frame_count'], transport_complete=True))
    print(json.dumps(dict(status='PASS', frames=receipt['frame_count'],
                         script_wall_s=receipt['wall_elapsed_s'], output=str(out))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('capture')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--engine', type=Path)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--capture', choices=CAPTURES, required=True)
    p.add_argument('--depth-export', choices=('legacy','single_access','exr','exr_probe'), default='exr')
    capture(parser.parse_args())


if __name__ == '__main__':
    main()
