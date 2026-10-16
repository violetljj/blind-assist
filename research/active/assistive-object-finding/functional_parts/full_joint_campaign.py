"""Bounded, sequential worker campaign; no model libraries in the controller.

Consumes an explicit sealed request. CPU preparation may overlap the separately
owned UMD controls; GPU work waits for their complete terminal. Every child has
an output, command receipt, timeout, and terminal. Failures are retained and are
never automatically retried. The parent can resume completed steps by identity.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ARMS = ('plain', 'plain_quality', 'detail', 'detail_quality', 'candidate_quality', 'joint')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stop_owned_tree(child, launched_unix, command):
    """The Windows venv PID may be a launcher; stop only its verified subtree."""
    import psutil
    try:
        process = psutil.Process(child.pid)
    except psutil.NoSuchProcess:
        return []
    if process.create_time() < launched_unix - 5 or str(command[2]).replace('\\', '/').lower() not in ' '.join(process.cmdline()).replace('\\', '/').lower():
        raise RuntimeError('Child identity changed; retain evidence instead of stopping an uncertain PID')
    owned = process.children(recursive=True)
    owned.reverse()
    owned.append(process)
    identities = [(p, p.create_time()) for p in owned]
    for p, created in identities:
        try:
            if p.create_time() == created:
                p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, retained = psutil.wait_procs(owned, timeout=15)
    if retained:
        raise RuntimeError('Owned process subtree did not exit; inspect actual identities before recovery')
    return [{'pid': p.pid, 'creation_unix': created} for p, created in identities]


def relocate(value, mappings):
    normalized = str(value).replace('\\', '/')
    for old, new in sorted(mappings, key=lambda pair: len(pair[0]), reverse=True):
        prefix = old.replace('\\', '/').rstrip('/')
        if normalized.lower() == prefix.lower() or normalized.lower().startswith(prefix.lower() + '/'):
            return new.rstrip('/\\') + normalized[len(prefix):]
    return str(value)


def prepare_inputs(legacy, root):
    mappings = read(legacy / 'migration/path_mapping.json')['prefixes']
    receipts = {}
    for kind, relative in (('iit', 'metadata/iit_training_manifest.jsonl'), ('umd', 'data/umd/shared_label_records.jsonl')):
        original = legacy / relative
        rows = [json.loads(line) for line in original.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        derived, scientific = [], []
        checked = 0
        for row in rows:
            item = dict(row)
            for key in ('image', 'mask'):
                item[key] = relocate(row[key], mappings)
                # The original manifest also retains excluded duplicate-val
                # and test metadata. Filter before filesystem access, exactly
                # as the inherited training harness does; do not fetch them.
                eligible = kind == 'umd' or row['split'] in ('train', 'val')
                if eligible and not Path(item[key]).is_file():
                    raise FileNotFoundError(item[key])
            if {k: v for k, v in item.items() if k not in ('image', 'mask')} != {k: v for k, v in row.items() if k not in ('image', 'mask')}:
                raise AssertionError('Scientific record changed during host-path derivation')
            if kind == 'iit' and row['split'] in ('train', 'val'):
                for field, expected in (('image', 'image_sha256'), ('mask', 'mask_sha256')):
                    if sha(item[field]) != row[expected]:
                        raise ValueError(f'Frozen {field} bytes differ: {row["source_id"]}')
                    checked += 1
            scientific.append({k: v for k, v in row.items() if k not in ('image', 'mask')})
            derived.append(item)
        if kind == 'iit':
            counts = {s: sum(row['split'] == s for row in rows) for s in ('train', 'val')}
            if counts != {'train': 4417, 'val': 1757}:
                raise ValueError(counts)
        elif len(rows) != 9939:
            raise ValueError('Complete UMD author-manual records required')
        target = root / 'inputs' / f'{kind}_records.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in derived).encode('utf-8')
        if target.exists() and target.read_bytes() != payload:
            raise ValueError('Derived manifest already exists with different identity')
        target.write_bytes(payload)
        receipts[kind] = {'path': str(target), 'source_path': str(original), 'source_sha256': sha(original),
            'derived_sha256': sha(target), 'rows': len(rows), 'IIT_RGB_mask_files_verified': checked,
            'filesystem_access_scope': 'IIT train/cleanval only, or UMD author-manual observations; excluded/test metadata retained without access',
            'nonpath_record_identity_sha256': hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest(),
            'only_image_and_mask_path_strings_relocated': True}
    config = read(legacy / 'runs/dino/config.json')
    pretrained = Path(relocate(config['dino_weights'], mappings))
    if sha(pretrained) != config['pretrained_sha256']:
        raise ValueError('Inherited DINO pretraining bytes changed')
    receipts['pretrained'] = {'path': str(pretrained), 'sha256': sha(pretrained)}
    write(root / 'inputs/receipt.json', dict(status='COMPLETE', utc=utc(), **receipts))
    return receipts


def run(request_path):
    request_path = Path(request_path).resolve()
    request = read(request_path)
    root, legacy = Path(request['root']), Path(request['legacy_root'])
    control = root / 'campaign'
    control.mkdir(parents=True, exist_ok=True)
    seal = sha(request_path)
    prior = read(control / 'progress.json') if (control / 'progress.json').exists() else None
    if prior and prior['request_sha256'] != seal:
        raise ValueError('Different prior campaign request; inspect before resume')
    lock = control / 'owner.lock'
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'request_sha256': seal, 'started_utc': utc(),
             'stage': 'prepare', 'completed_steps': [], 'training_process_wall_seconds': 0., 'child_pid': None}
    if prior:
        state['training_process_wall_seconds'] = prior.get('training_process_wall_seconds', 0.)
    child = None
    python = request['python']
    code = {name: Path(path) for name, path in request['sources'].items()}

    def progress():
        state['updated_utc'] = utc()
        write(control / 'progress.json', state)

    def await_gpu():
        # Do not compete with another chat's GPU work. Display memory alone is
        # expected; no external process is stopped or reconfigured here.
        deadline = time.monotonic() + 3600
        while True:
            probe = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                   text=True, capture_output=True, check=True)
            free = int(probe.stdout.strip().splitlines()[0])
            compute = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                     text=True, capture_output=True, check=True)
            pids = [int(line.strip()) for line in compute.stdout.splitlines() if line.strip().isdigit()]
            if free >= 4900 and not pids:
                return
            state.update(stage='WAIT_GPU_AVAILABLE', gpu_free_mib=free, other_gpu_pids=pids)
            progress()
            if time.monotonic() >= deadline or (control / 'STOP_REQUESTED').exists():
                raise RuntimeError('GPU capacity wait ended; other task resources left untouched')
            time.sleep(30)

    def step(name, command, output, expected, *, seconds=3600, cpu=False, training=False):
        nonlocal child
        for path, digest in request['source_sha256'].items():
            if sha(path) != digest:
                raise ValueError(f'Active source changed: {path}')
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        command_sha = hashlib.sha256(json.dumps(command).encode()).hexdigest()
        receipt = output / 'campaign_step.json'
        terminal = output / 'terminal.json'
        if receipt.exists():
            old = read(receipt)
            if old['command_sha256'] != command_sha:
                raise ValueError(f'Existing command differs: {name}')
            if old['status'] == 'SUCCESS' and terminal.exists() and read(terminal).get('status') == expected:
                state['completed_steps'].append(name)
                return read(terminal)
            raise RuntimeError(f'Prior failed/uncertain step {name}; inspect owner/checkpoint, do not retry automatically')
        environment = os.environ.copy()
        environment.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        if cpu:
            environment['CUDA_VISIBLE_DEVICES'] = ''
        else:
            environment.pop('CUDA_VISIBLE_DEVICES', None)
            await_gpu()
        state.update(stage=name, child_pid=None)
        progress()
        started, launched_unix = time.monotonic(), time.time()
        result = dict(status='RUNNING', command=command, command_sha256=command_sha, started_utc=utc(),
                      wall_seconds=0., timeout_seconds=seconds, training=training)
        write(receipt, result)
        with (output / 'stdout.log').open('ab', buffering=0) as stdout, (output / 'stderr.log').open('ab', buffering=0) as stderr:
            child = subprocess.Popen(command, cwd=code['train'].parent, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
            result['pid'] = child.pid
            state['child_pid'] = child.pid
            write(receipt, result)
            while child.poll() is None:
                elapsed = time.monotonic() - started
                state['elapsed_step_seconds'] = elapsed
                progress()
                if (control / 'STOP_REQUESTED').exists() or elapsed > seconds:
                    reason = 'OPERATOR_STOP' if (control / 'STOP_REQUESTED').exists() else 'STEP_BUDGET'
                    result['stopped_owned_subtree'] = stop_owned_tree(child, launched_unix, command)
                    child.wait(timeout=10)
                    result.update(status='STOPPED', stop_reason=reason)
                    break
                time.sleep(5)
            elapsed = time.monotonic() - started
            result.update(exit_code=child.returncode, wall_seconds=elapsed, ended_utc=utc())
            child = None
        if training:
            state['training_process_wall_seconds'] += elapsed
        if result['status'] != 'STOPPED':
            result['status'] = 'SUCCESS' if result['exit_code'] == 0 and terminal.exists() and read(terminal).get('status') == expected else 'FAILED'
        write(receipt, result)
        state['child_pid'] = None
        progress()
        if result['status'] != 'SUCCESS':
            raise RuntimeError(f'{name} {result["status"]}; inspect logs/owner, no automatic retry')
        state['completed_steps'].append(name)
        return read(terminal)

    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'request_sha256': seal, 'started_utc': utc()}, stream)
    try:
        inputs = prepare_inputs(legacy, root)
        records, weights = inputs['iit']['path'], inputs['pretrained']['path']
        geometry = root / 'geometry'
        step('geometry', [python, '-B', str(code['train']), 'precompute', '--records', records,
             '--output', str(geometry), '--size', '448'], geometry, 'COMPLETE', seconds=1800, cpu=True)
        state['stage'] = 'WAIT_UMD_CONTROLS'
        progress()
        controls_terminal = Path(request['controls_terminal'])
        deadline = time.monotonic() + request.get('controls_wait_seconds', 10800)
        while not controls_terminal.exists():
            if (control / 'STOP_REQUESTED').exists() or time.monotonic() >= deadline:
                raise RuntimeError('Control wait stopped/expired; no GPU training started')
            time.sleep(20)
        if read(controls_terminal).get('status') != 'COMPLETE':
            raise RuntimeError('UMD controls did not complete; no GPU training started')
        control_summary = read(controls_terminal.parent / 'summary.json')
        if control_summary['images'] != 9939 or control_summary['positive_queries'] != 19370:
            raise ValueError('UMD controls complete denominator differs')
        headroom = {'status': 'COMPLETE', 'scope': 'posthoc descriptive headroom, never a UMD-selected model output',
                    'trained_selector_rule': 'fixed IIT validation point/quality rule; no UMD parameter tuning', 'partitions': {}}
        for partition in ('overall', 'area/small'):
            values = control_summary['partitions'][partition]['models']
            headroom['partitions'][partition] = {model: values[model]['candidate_pool_headroom_descriptive']
                                                 for model in ('segformer', 'dino')}
        write(root / 'controls/HEADROOM_BEFORE_TRAINING.json', headroom)
        profiles = {}
        base = [python, '-B', str(code['train'])]
        common = ['--records', records, '--cache', str(geometry), '--dino-weights', weights,
            '--size', '448', '--batch', str(request['batch']), '--effective-batch', '16',
            '--workers', str(request['workers']), '--cpu-threads', str(request['cpu_threads']),
            '--seed', '20261016', '--init-seed', '20261016', '--lr', '6e-5',
            '--val-every', str(request.get('val_every', 4)),
            '--expected-source-sha256', request['source_sha256'][str(code['train'])],
            '--expected-model-sha256', request['source_sha256'][str(code['models'])]]
        for arm in ('joint', *ARMS[:-1]):
            output = root / 'profiles' / arm
            profiles[arm] = step('profile_' + arm, [*base, 'profile', *common, '--arm', arm,
                '--output', str(output), '--profile-updates', str(request.get('profile_updates', 8)),
                '--profile-val-images', '16'], output, 'PROFILE_COMPLETE', seconds=900)
        # Fixed before training and before seeing any trained validation outcome.
        updates_per_epoch = math.ceil(4417 / 16)
        cap = request['training_cap_seconds']
        def projected(epochs):
            return sum(p['median_seconds_excluding_first'] * updates_per_epoch * epochs +
                p['validation_speed']['estimated_full_1757_seconds_linear'] * math.ceil(epochs / request.get('val_every', 4)) +
                8 * epochs + 45 for p in profiles.values())
        eligible = [epochs for epochs in (80, 60, 40) if projected(epochs) <= cap * .9]
        if not eligible:
            raise RuntimeError('Even common40epoch matrix exceeds sealed24h projection; report capacity, do not silently reduce to a tiny pilot')
        epochs = max(eligible)
        common_budget = min(p['median_seconds_excluding_first'] * updates_per_epoch * epochs for p in profiles.values()) * .75
        plan = dict(status='COMPLETE', arms=list(ARMS), epochs=epochs, schedule_epochs=epochs,
            train_rows=4417, cleanval_rows=1757, seed=20261016, batch=request['batch'], effective_batch=16,
            training_process_cap_seconds=cap, projected_training_process_seconds=projected(epochs),
            common_update_wall_budget_seconds=common_budget, budget_checkpoint_selection='last eligible scheduled validation boundaries; actual undershoot reported',
            validation_every_epochs=request.get('val_every', 4),
            profile_scope='measured8updates plus16spaced-native-val speed check with startup separated; estimates not full-run timing', profiles=profiles)
        write(root / 'profiles/summary.json', plan)
        for arm in ARMS:
            remaining = cap - state['training_process_wall_seconds']
            if remaining <= 0:
                raise RuntimeError('Campaign training budget exhausted')
            output = root / 'runs' / arm
            step('train_' + arm, [*base, 'train', *common, '--arm', arm, '--output', str(output),
                 '--epochs', str(epochs), '--schedule-epochs', str(epochs),
                 '--compute-budget-seconds', str(common_budget)], output, 'COMPLETE', seconds=remaining, training=True)
        # No CPU scoring competes with training; all frozen checkpoint evaluation
        # follows the six-arm training block and never changes model selection.
        evaluation_started = time.monotonic()
        for selection, checkpoint_name in (('point', 'best.pt'), ('budget', 'best_budget.pt')):
            for arm in ARMS:
                checkpoint = root / 'runs' / arm / checkpoint_name
                for dataset in ('iit', 'umd'):
                    remaining = request.get('evaluation_cap_seconds', 21600) - (time.monotonic() - evaluation_started)
                    if remaining <= 0:
                        raise RuntimeError('Evaluation budget exhausted; retain all partial results')
                    prefix = root / 'evaluation' / selection / arm / dataset
                    exported = prefix / 'export'
                    step('_'.join(('export', selection, arm, dataset)), [python, '-B', str(code['evaluate']), 'export',
                        '--checkpoint', str(checkpoint), '--selection', selection,
                        '--records', inputs[dataset]['path'], '--dataset', dataset, '--data-root', str(legacy),
                        '--output', str(exported), '--batch', str(request['batch']), '--cpu-threads', '1'],
                        exported, 'COMPLETE', seconds=min(3600, remaining))
                    scored = prefix / 'score'
                    step('_'.join(('score', selection, arm, dataset)), [python, '-B', str(code['evaluate']), 'score',
                        '--predictions', str(exported / 'predictions.jsonl'), '--data-root', str(legacy),
                        '--output', str(scored), '--bootstrap', '1000', '--seed', '20261015'],
                        scored, 'COMPLETE', seconds=min(1800, remaining), cpu=True)
        final = root / 'final'
        step('final_report', [python, '-B', str(code['report']), '--root', str(root), '--output', str(final)],
             final, 'COMPLETE', seconds=600, cpu=True)
        state.update(status='SUCCESS', stage='COMPLETE', ended_utc=utc())
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc(), ended_utc=utc())
        raise
    finally:
        if child is not None and child.poll() is None:
            state['retained_child_pid'] = child.pid
        progress()
        write(control / 'terminal.json', state)
        if child is None or child.poll() is not None:
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    run(parser.parse_args().request)


if __name__ == '__main__':
    main()
