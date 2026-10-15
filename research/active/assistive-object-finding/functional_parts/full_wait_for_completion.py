"""Read-only readiness gate for a post-training queue; no model or GT access."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import traceback


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--terminal', type=Path, required=True)
    p.add_argument('--expected-status', required=True)
    p.add_argument('--required-file', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-seconds', type=float, default=36*3600)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        while time.monotonic() - start < a.max_seconds:
            current = json.loads(a.terminal.read_text(encoding='utf-8')) if a.terminal.exists() else None
            status = current.get('status') if current else 'PENDING'
            state = {'status': 'WAITING', 'target': str(a.terminal), 'target_status': status,
                     'elapsed_s': time.monotonic() - start, 'GPU_used': False, 'GT_read': False}
            save(a.output/'progress.json', state)
            if status == a.expected_status:
                if a.required_file and (not a.required_file.is_file() or not a.required_file.stat().st_size):
                    raise RuntimeError(f'Terminal success missing required output: {a.required_file}')
                save(a.output/'terminal.json', dict(state, status='COMPLETE'))
                return
            if current and status not in ('RUNNING', 'WAITING', 'PENDING'):
                raise RuntimeError(f'Upstream terminal did not succeed: {status}; inspect original failure, do not rerun automatically')
            time.sleep(30)
        raise TimeoutError('Readiness deadline reached; original jobs left intact for inspection')
    except BaseException:
        save(a.output/'terminal.json', {'status':'FAILED', 'error':traceback.format_exc(),
                                       'target':str(a.terminal), 'GT_read':False, 'GPU_used':False})
        raise


if __name__ == '__main__':
    main()
