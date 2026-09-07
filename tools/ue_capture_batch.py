"""One UE process, independent capture blocks, overlapping validation and hash-safe resume.

Manifest: {"jobs": [{"id": "block-001", "capture": "grounding", "spec": "absolute spec path"}]}.
Each spec is a complete independent dataset. No implicit renumbering or splitting
of scientific clips. Outputs become usable when listed PASS in completed.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import ue_native_capture as capture


def atomic_write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    capture.write(tmp, value)
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def request_identity(args):
    manifest = read(args.manifest)
    jobs, seen = manifest['jobs'], set()
    if not jobs:
        raise ValueError('Manifest has no jobs')
    for job in jobs:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', job['id']) or job['id'] in seen:
            raise ValueError('Invalid or duplicate block id')
        seen.add(job['id'])
        if job['capture'] not in ('grounding', 'factorial'):
            raise ValueError('Batch supports settled grounding/factorial capture only')
        spec = Path(job['spec']).resolve()
        if not spec.is_relative_to((capture.REPO/'artifacts.local').resolve()):
            raise ValueError('Block spec must be under artifacts.local')
        job['spec'] = str(spec)
        job['spec_sha256'] = capture.file_hash(spec)
    from run_obstacle_research import engine_root
    engine = engine_root(args.engine)
    project = (capture.REPO/'artifacts.local/unreal/BlindAssistStreetLab').resolve()
    source = capture.REPO/'research/active/dtr-r0/nearfield'
    scripts = ['grounding_capture.py', 'factorial_capture.py', 'ue_capture_readiness.py',
               'ue_capture_session.py', 'ue_pair_export.py', 'ue_rgb_export.py',
               'ue_settling.py', 'ue_depth_export.py', 'ue_exr_transport.py']
    identity = dict(schema='ue-capture-batch-v1', jobs=jobs,
        settings=dict(startup_policy=args.startup_policy, settling_policy=args.settling_policy,
                      pair_export=args.pair_export, lean_init=not args.standard_init),
        engine=str(engine), engine_version=capture.file_hash(engine/'Engine/Build/Build.version'),
        project=str(project), project_sha256=capture.file_hash(project/'BlindAssistStreetLab.uproject'),
        config_hashes={p.relative_to(project).as_posix(): capture.file_hash(p) for p in sorted((project/'Config').rglob('*.ini'))},
        map_sha256=capture.file_hash(project/'Content/StreetLab/WillowSampleV1.umap'),
        plugin=str(args.plugin.resolve()), plugin_sha256=capture.file_hash(args.plugin),
        plugin_binary_sha256=capture.file_hash(args.plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll'),
        source_hashes={name: capture.file_hash(source/name) for name in scripts},
        host_hashes={name: capture.file_hash(Path(__file__).with_name(name)) for name in ('ue_capture_batch.py', 'ue_native_capture.py')})
    identity['fingerprint'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity


def reusable(record):
    """Never trust a completed flag without checking the retained payload hashes."""
    if record.get('status') != 'PASS':
        return False
    out = Path(record['output'])
    try:
        completion = read(out/'completion.json')
        hashes = out/'payload-hashes.json'
        payloads = read(hashes)
        return (bool(payloads) and completion['status'] == 'PASS'
                and completion['payload_hashes_sha256'] == capture.file_hash(hashes)
                and all((out/name).resolve().is_relative_to(out.resolve())
                        and capture.file_hash(out/name) == value for name, value in payloads.items()))
    except (OSError, KeyError, ValueError):
        return False


def batch(args):
    identity = request_identity(args)
    out = args.output.resolve()
    root = (capture.REPO/'artifacts.local').resolve()
    if not out.is_relative_to(root) or out == root:
        raise ValueError('Batch output must be strictly under artifacts.local')
    if out.exists():
        if not args.resume:
            raise FileExistsError('Use --resume for an existing batch')
        if read(out/'identity.json') != identity:
            raise ValueError('Inputs/environment changed; use a new batch output')
    else:
        out.mkdir(parents=True)
        atomic_write(out/'identity.json', identity)
    # Exclusive host owner. A crashed owner leaves an explicit lock: inspect it
    # and its process before removing; never steal a live batch implicitly.
    lock = out/'owner.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps(dict(pid=os.getpid(), created_unix_s=time.time())).encode())
    os.close(fd)
    try:
        return run_batch(args, identity, out)
    finally:
        lock.unlink()


def run_batch(args, identity, out):
    ledger = read(out/'completed.json') if (out/'completed.json').exists() else {}
    remaining = [job for job in identity['jobs'] if not (
        Path(ledger.get(job['id'], {}).get('output', '.')).resolve().is_relative_to((out/'jobs'/job['id']).resolve())
        and reusable(ledger.get(job['id'], {})))]
    if not remaining:
        print(json.dumps(dict(status='PASS', cached=len(ledger), launched=False)))
        return
    session = out/'sessions'/f'{time.time_ns()}'
    session.mkdir(parents=True)
    for job in remaining:
        previous = ledger.get(job['id'])
        if previous:
            atomic_write(session/(job['id'] + '-previous.json'), previous)
        ledger[job['id']] = dict(status='PENDING', previous_output=previous.get('output') if previous else None)
    atomic_write(out/'completed.json', ledger)
    prepared = []
    futures = {}
    error = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='capture-validation') as pool:
        def validate(item):
            begin = time.time()
            try:
                receipt = capture.validate_capture(item['out'])
                return dict(status='PASS', output=str(item['out']), frames=receipt['frame_count'],
                            validation_started_unix_s=begin, completed_unix_s=time.time())
            except Exception as exc:
                record = dict(status='FAIL', output=str(item['out']), error=str(exc), completed_unix_s=time.time())
                atomic_write(item['out']/'completion.json', record)
                return record

        def pump():
            for item in prepared:
                name = item['id']
                if name not in futures and (item['out']/'receipt.json').is_file():
                    # Writer may still be closing on Windows; retry next poll.
                    try:
                        read(item['out']/'receipt.json')
                    except (OSError, ValueError):
                        continue
                    futures[name] = pool.submit(validate, item)
                if name in futures and futures[name].done():
                    result = futures[name].result()
                    if ledger.get(name) != result:
                        ledger[name] = result
                        atomic_write(out/'completed.json', ledger)
                        atomic_write(session/(name + '-validation.json'), result)
                        if result['status'] != 'PASS':
                            (session/'stop.request').touch()

        try:
            for job in remaining:
                block = out/'jobs'/job['id']/session.name
                options = argparse.Namespace(capture=job['capture'], spec=Path(job['spec']), output=block,
                    engine=args.engine, plugin=args.plugin, timeout=args.timeout,
                    depth_export='native', rgb_export='auto', pair_export=args.pair_export,
                    cadence='burst', settling_policy=args.settling_policy,
                    startup_policy=args.startup_policy, lean_init=not args.standard_init, _prepare_only=True)
                item = capture.capture(options)
                item['id'] = job['id']
                if capture.file_hash(item['out']/'source/spec.json') != job['spec_sha256']:
                    raise ValueError('Spec changed during preparation')
                for name, expected in identity['source_hashes'].items():
                    snapshot = item['out']/'source'/name
                    if snapshot.exists() and capture.file_hash(snapshot) != expected:
                        raise ValueError('Capture source changed during preparation')
                prepared.append(item)
            bootstrap = session/'ue_capture_session.py'
            shutil.copy2(capture.REPO/'research/active/dtr-r0/nearfield/ue_capture_session.py', bootstrap)
            if capture.file_hash(bootstrap) != identity['source_hashes']['ue_capture_session.py']:
                raise ValueError('Session source changed during preparation')
            atomic_write(session/'request.json', dict(fingerprint=identity['fingerprint'], jobs=[dict(
                id=p['id'], script=str(p['script']), output=str(p['out']),
                env={k: v for k, v in p['env'].items() if k.startswith('BA_UE_') or k.startswith('BA_NEARFIELD_')}) for p in prepared]))
            command = [('-ExecCmds=py ' + bootstrap.as_posix()) if arg.startswith('-ExecCmds=') else
                       ('-abslog=' + str(session/'editor.log')) if arg.startswith('-abslog=') else arg
                       for arg in prepared[0]['command']]
            env = dict(prepared[0]['env'], BA_UE_SESSION=str(session))
            atomic_write(session/'launch.json', dict(command=command, fingerprint=identity['fingerprint']))
            if request_identity(args) != identity:
                raise ValueError('Inputs/environment changed during preparation')
            capture.run_owned(command, env, session, args.timeout, pump)
            if read(session/'receipt.json')['status'] != 'PASS':
                raise RuntimeError('Engine session failed; inspect session receipt')
        except BaseException as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        finally:
            pump()
            for future in futures.values():
                future.result()
            pump()
            for item in prepared:
                if item['id'] not in futures:
                    record = dict(status='FAIL', output=str(item['out']), error=error or 'No engine receipt')
                    ledger[item['id']] = record
                    atomic_write(item['out']/'completion.json', record)
                    atomic_write(session/(item['id'] + '-validation.json'), record)
            atomic_write(out/'completed.json', ledger)
            passed = all(ledger.get(job['id'], {}).get('status') == 'PASS' for job in identity['jobs'])
            atomic_write(session/'completion.json', dict(status='PASS' if passed and not error else 'FAIL', error=error))
    if error or not passed:
        raise RuntimeError(error or 'One or more blocks failed validation')
    print(json.dumps(dict(status='PASS', blocks=len(ledger), acquired=len(remaining), session=str(session))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plugin', type=Path, default=os.environ.get('BLINDASSIST_UE_CAPTURE_PLUGIN') or os.environ.get('BA_UE_CAPTURE_PLUGIN'), required=False)
    p.add_argument('--engine', type=Path)
    p.add_argument('--timeout', type=float, default=1800)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--startup-policy', choices=('ready', 'fixed'), default='ready')
    p.add_argument('--settling-policy', choices=('auto', 'full', 'reuse'), default='auto')
    p.add_argument('--pair-export', choices=('native_async', 'native_probe'), default='native_async')
    p.add_argument('--standard-init', action='store_true')
    args = p.parse_args()
    if not args.plugin:
        p.error('--plugin or BLINDASSIST_UE_CAPTURE_PLUGIN is required')
    batch(args)


if __name__ == '__main__':
    main()
