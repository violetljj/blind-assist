"""Durable sequential local training queue; each subprocess owns one output tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def alive(pid):
    try:
        if os.name == 'nt':
            # os.kill(pid, 0) on Windows can terminate a process; use a read-only query.
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return ctypes.get_last_error() == 5  # Access denied is not proof of death.
            try:
                code = wintypes.DWORD()
                return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
            finally:
                kernel.CloseHandle(handle)
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def check_success_artifact(job):
    spec = job.get('success_artifact')
    if not spec:
        return
    path = Path(spec['path'] if isinstance(spec, dict) else spec)
    if not path.is_file():
        raise RuntimeError(f'Missing success artifact for {job["id"]}: {path}')
    expected = spec.get('expected_status') if isinstance(spec, dict) else job.get('expected_status')
    if expected is not None:
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('status') != expected:
            raise RuntimeError(f'Artifact status for {job["id"]}: {value.get("status")} != {expected}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    a = p.parse_args()
    request_path = a.request.resolve()
    raw = request_path.read_bytes()
    request = json.loads(raw)
    root = Path(request['output']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    seal = hashlib.sha256(raw).hexdigest()
    lock = root / 'queue.lock.json'
    if lock.exists():
        previous = json.loads(lock.read_text(encoding='utf-8'))
        if alive(previous.get('pid')):
            raise RuntimeError(f'Queue already active: pid={previous.get("pid")}')
        if previous.get('request_sha256') != seal:
            raise RuntimeError('Stale lock belongs to different request; inspect ownership')
        possible_children = [previous.get('child_pid')]
        for state_path in (root / 'progress.json', root / 'terminal.json'):
            if state_path.exists():
                prior_state = json.loads(state_path.read_text(encoding='utf-8'))
                possible_children.extend([prior_state.get('retained_child_pid'), prior_state.get('child_pid')])
        for job in request['jobs']:
            owner_path = Path(job['output']) / 'queue_owner.json'
            if owner_path.exists():
                possible_children.append(json.loads(owner_path.read_text(encoding='utf-8')).get('pid'))
        if any(pid and alive(pid) for pid in possible_children):
            raise RuntimeError('Queue owner died but a retained child may still be writing; inspect before resume')
        lock.unlink()
    with lock.open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'request_sha256': seal, 'started': time.time()}, f)
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'request_sha256': seal,
             'started': time.time(), 'jobs': [], 'current': None}
    child = None
    try:
        for job in request['jobs']:
            out = Path(job['output']).resolve()
            if not out.is_relative_to(root.parent):
                raise ValueError(f'Job output must stay in run tree: {out}')
            out.mkdir(parents=True, exist_ok=True)
            terminal = out / 'queue_terminal.json'
            command = list(map(str, job['command']))
            command_sha = hashlib.sha256(json.dumps(command).encode()).hexdigest()
            execution_command = command
            if terminal.exists():
                old = json.loads(terminal.read_text(encoding='utf-8'))
                if old.get('command_sha256') != command_sha:
                    raise RuntimeError(f'Changed command for existing job {job["id"]}')
                if old.get('status') == 'SUCCESS':
                    check_success_artifact(job)
                    state['jobs'].append(old)
                    continue
                if not job.get('resume', False):
                    raise RuntimeError(f'Failed/interrupted job requires explicit resume: {job["id"]}')
                history = out / 'queue_attempts'
                history.mkdir(exist_ok=True)
                save(history / (str(old.get('started', time.time())).replace('.', '_') + '.json'), old)
                if job.get('resume_command'):
                    execution_command = list(map(str, job['resume_command']))
            for path_str, expected in job.get('input_sha256', {}).items():
                value = hashlib.sha256(Path(path_str).read_bytes()).hexdigest()
                if value != expected:
                    raise RuntimeError(f'Input changed: {path_str}')
            state['current'] = job['id']
            state['last_activity'] = time.time()
            save(root / 'progress.json', state)
            started = time.time()
            result = {'id': job['id'], 'command': command, 'command_sha256': command_sha,
                      'execution_command': execution_command,
                      'started': started, 'status': 'RUNNING', 'output': str(out)}
            env = os.environ.copy()
            env.update({'PYTHONUNBUFFERED': '1'})
            env.update({str(k): str(v) for k, v in request.get('environment', {}).items()})
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            with (out / 'queue_stdout.log').open('ab', buffering=0) as stdout, \
                    (out / 'queue_stderr.log').open('ab', buffering=0) as stderr:
                child = subprocess.Popen(execution_command, cwd=request.get('cwd'), env=env,
                                         stdout=stdout, stderr=stderr, creationflags=flags)
                result['pid'] = child.pid
                save(out / 'queue_owner.json', result)
                save(lock, {'pid': os.getpid(), 'child_pid': child.pid, 'request_sha256': seal, 'started': state['started']})
                print(json.dumps({'started': job['id'], 'pid': child.pid}), flush=True)
                while child.poll() is None:
                    state['child_pid'] = child.pid
                    state['last_activity'] = time.time()
                    state['elapsed_job_s'] = time.time() - started
                    save(root / 'progress.json', state)
                    if (root / 'STOP_REQUESTED').exists():
                        if os.name == 'nt':
                            child.send_signal(signal.CTRL_BREAK_EVENT)
                        else:
                            child.send_signal(signal.SIGINT)
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            child.terminate()
                        raise RuntimeError('User/operator stop request received')
                    time.sleep(5)
                result.update({'exit_code': child.returncode,
                               'ended': time.time(), 'wall_s': time.time() - started,
                               'status': 'SUCCESS' if child.returncode == 0 else 'FAILED'})
                child = None
            if result['status'] == 'SUCCESS':
                try:
                    check_success_artifact(job)
                except Exception as error:
                    result.update(status='FAILED', artifact_error=str(error))
            save(terminal, result)
            state['jobs'].append(result)
            print(json.dumps({'finished': job['id'], 'status': result['status']}), flush=True)
            if result['status'] != 'SUCCESS':
                raise RuntimeError(f'Job failed: {job["id"]}; inspect job logs, no automatic restart')
        state.update({'status': 'SUCCESS', 'current': None})
    except BaseException:
        state.update({'status': 'FAILED', 'error': traceback.format_exc()})
        raise
    finally:
        state['ended'] = time.time()
        if child is not None:
            state['retained_child_pid'] = child.pid if child.poll() is None else None
            state['retained_reason'] = 'Queue interrupted; inspect child before any resume'
        save(root / 'progress.json', state)
        save(root / 'terminal.json', state)
        # Do not release writer ownership while a retained subprocess can still write.
        if child is None or child.poll() is not None:
            if lock.exists():
                lock.unlink()


if __name__ == '__main__':
    main()
