"""Run location and bounded stage accounting for pass/query Explore."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-pass-boundary-dev-20261009'
SOURCE = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    pending.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def charged_seconds():
    return sum(read(path)['seconds'] for path in (OUT/'stage_receipts').glob('*.json'))


class Stage:
    """Sequential command-stage wall, including failures and internal CPU/I/O."""
    def __init__(self, name):
        self.name, self.start = name, time.monotonic()
        self.previous = charged_seconds()
        self.cap = read(OUT/'PLAN.json')['gpu_wall_cap_seconds']
        self.check()

    def check(self):
        if self.previous+time.monotonic()-self.start >= self.cap:
            raise TimeoutError('Pass/query experiment stage wall cap reached')

    def finish(self, status, **details):
        receipt = dict(stage=self.name, status=status, seconds=time.monotonic()-self.start,
            previous_seconds=self.previous, **details)
        save(OUT/'stage_receipts'/f'{self.name}_{time.time_ns()}.json', receipt)
        return receipt
