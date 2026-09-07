"""Run UE capture with native NPY or bounded lossless EXR transport.

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


def verify_rgb(out, count, probe=False):
    from PIL import Image
    rows = []
    for index in range(count):
        path = out / f'model/sample/{index:04d}.png'
        with Image.open(path) as image:
            image.load()
            if image.size != (640, 360):
                raise ValueError('Unexpected RGB dimensions: ' + str(path))
            pixels = image.convert('RGBA').tobytes()
        row = dict(sample_index=index, pixel_sha256=hashlib.sha256(pixels).hexdigest())
        if probe:
            with Image.open(path.with_name(path.stem + '.reference.png')) as reference:
                if reference.size != (640, 360) or reference.convert('RGBA').tobytes() != pixels:
                    raise ValueError('Native RGB differs from same-target reference: ' + str(path))
            row['same_target_pixels_equal'] = True
        rows.append(row)
    write(out / 'rgb-validation.json', dict(status='PASS', frames=count, rows=rows))


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
    cadence = getattr(args, 'cadence', 'auto')
    if cadence == 'auto':
        cadence = 'burst' if args.capture in ('grounding', 'factorial') else 'tick'
    settling_policy = getattr(args, 'settling_policy', 'auto')
    if settling_policy == 'reuse' and (args.capture == 'whisker' or cadence != 'burst'):
        raise ValueError('Settling reuse requires grounding/factorial burst poses')
    plugin = getattr(args, 'plugin', None)
    pair_export = getattr(args, 'pair_export', 'auto')
    if pair_export == 'auto' and (not plugin or args.capture == 'whisker' or args.depth_export not in ('auto', 'native') or getattr(args, 'rgb_export', 'auto') != 'auto'):
        pair_export = 'off'
    if pair_export != 'off' and args.capture == 'whisker':
        raise ValueError('GPU pair export currently supports settled grounding/factorial captures')
    if pair_export != 'off' and args.depth_export not in ('auto', 'native'):
        raise ValueError('GPU pair export requires auto/native depth; EXR and depth probes use the separate path')
    rgb_export = getattr(args, 'rgb_export', 'auto')
    if rgb_export == 'auto' and not plugin:
        rgb_export = 'legacy'
    if args.depth_export == 'auto':
        args.depth_export = 'native' if plugin else 'exr'
    if args.depth_export in ('native', 'native_probe') or rgb_export != 'legacy' or pair_export != 'off':
        if not plugin or not plugin.is_file():
            raise ValueError('Native export requires --plugin from build_ue_capture_plugin.py')
        if not (plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll').is_file():
            raise ValueError('Native plugin DLL is missing')
    root = (REPO / 'artifacts.local').resolve()
    spec = args.spec.resolve()
    if not spec.is_relative_to(root) or not spec.is_file():
        raise ValueError('Spec must exist under artifacts.local')
    if settling_policy == 'auto':
        sampling = json.loads(spec.read_text(encoding='utf-8-sig')).get('sampling')
        settling_policy = 'reuse' if args.capture in ('grounding', 'factorial') and cadence == 'burst' and sampling == 'THREE_STATIC_SETTLED_POSES_SIMULATED_5HZ_NOT_MOTION_TEST' else 'full'
    script = REPO / 'research/active/dtr-r0/nearfield' / CAPTURES[args.capture]
    engine = engine_root(args.engine)
    project = root / 'unreal/BlindAssistStreetLab'
    out = owned_output(args.output)
    # Freeze the actual scripts used by this process against concurrent edits.
    snapshot = out/'source'
    snapshot.mkdir()
    for source in (script, script.with_name('ue_depth_export.py'), script.with_name('ue_exr_transport.py'), script.with_name('ue_rgb_export.py'), script.with_name('ue_settling.py'), script.with_name('ue_pair_export.py')):
        shutil.copy2(source, snapshot/source.name)
    script = snapshot/script.name
    shutil.copy2(spec, snapshot/'spec.json')
    spec = snapshot/'spec.json'
    env = dict(os.environ, BA_NEARFIELD_SPEC=str(spec), BA_NEARFIELD_OUTPUT=str(out),
               BA_UE_DEPTH_EXPORT=args.depth_export,
               BA_UE_CADENCE=cadence, BA_UE_RGB_EXPORT=rgb_export, BA_UE_SETTLING=settling_policy, BA_UE_PAIR_EXPORT=pair_export)
    env['UE-LocalDataCachePath'] = str(project / 'DerivedDataCache')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [str(engine / 'Engine/Binaries/Win64/UnrealEditor.exe'),
               str(project / 'BlindAssistStreetLab.uproject'),
               '-ExecCmds=py ' + script.as_posix(), '-RenderOffscreen', '-unattended',
               '-nosound', '-nop4', '-NoSplash', '-ddc=NoShared', '-abslog=' + str(out/'editor.log')]
    if args.depth_export in ('native', 'native_probe') or rgb_export != 'legacy' or pair_export != 'off':
        command += ['-PLUGIN=' + str(plugin.resolve()), '-EnablePlugins=BlindAssistCapture']
    write(out / 'launch.json', dict(depth_export=args.depth_export, rgb_export=rgb_export, pair_export=pair_export, settling_policy=settling_policy, cadence=env['BA_UE_CADENCE'], script=str(script),
          script_sha256=file_hash(script), exporter_sha256=file_hash(script.with_name('ue_depth_export.py')),
          rgb_exporter_sha256=file_hash(script.with_name('ue_rgb_export.py')),
          pair_exporter_sha256=file_hash(script.with_name('ue_pair_export.py')),
          settling_sha256=file_hash(script.with_name('ue_settling.py')),
          transport_sha256=file_hash(script.with_name('ue_exr_transport.py')),
          spec_sha256=file_hash(spec), command=command,
          plugin_binary_sha256=file_hash(plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll') if plugin else None))
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
        effective_rgb = receipt.get('rgb_exports', {}).get('mode', 'legacy')
        effective_pair = receipt.get('pair_exports', {}).get('mode', 'off')
        if effective_pair != 'off':
            import numpy as np
            verify_rgb(out, receipt['frame_count'], probe=effective_pair == 'native_probe')
            for i in range(receipt['frame_count']):
                path = out / f'evaluator/native/{i:04d}.npy'
                depth = np.load(path, allow_pickle=False)
                if depth.shape != (360, 640) or depth.dtype != np.dtype('<f4') or not np.isfinite(depth).all() or not ((depth >= 0) & (depth < 100)).all():
                    raise ValueError('Invalid GPU depth output: ' + str(path))
                if effective_pair == 'native_probe' and path.read_bytes() != path.with_name(path.stem + '.reference.npy').read_bytes():
                    raise ValueError('GPU depth differs from same-target reference: ' + str(path))
        elif effective_rgb != 'legacy':
            verify_rgb(out, receipt['frame_count'], probe=effective_rgb == 'native_probe')
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
          rgb_export=effective_rgb, pair_export=effective_pair, settling_policy=settling_policy,
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
    p.add_argument('--depth-export', choices=('auto','legacy','single_access','exr','exr_probe','native','native_probe'), default='auto',
                   help='Auto selects native when --plugin is supplied, otherwise EXR')
    p.add_argument('--pair-export', choices=('auto','off','native_async','native_probe'), default='auto')
    p.add_argument('--settling-policy', choices=('auto','full','reuse'), default='auto')
    p.add_argument('--rgb-export', choices=('auto','legacy','native_async','native_probe'), default='auto',
                   help='Auto uses background PNG encoding when the loaded plugin supports it')
    p.add_argument('--cadence', choices=('auto','tick','burst'), default='auto',
                   help='Auto batches grounding/factorial poses; whisker motion retains tick cadence')
    p.add_argument('--plugin', type=Path, default=os.environ.get('BA_UE_CAPTURE_PLUGIN'),
                   help='Built native plugin descriptor; also accepts BA_UE_CAPTURE_PLUGIN')
    capture(parser.parse_args())


if __name__ == '__main__':
    main()
