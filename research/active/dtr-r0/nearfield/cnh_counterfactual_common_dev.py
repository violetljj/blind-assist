"""Shared immutable run location and command-stage wall accounting for Explore."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-counterfactual-dev-20261009'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 23), b''):
            digest.update(block)
    return digest.hexdigest()


def charged_seconds():
    return sum(read(p)['seconds'] for p in (OUT/'stage_receipts').glob('*.json'))


class Stage:
    """One owning process. Successful and failed enclosing stages are both charged.

    GPU stages are run sequentially by the primary controller, so no overlapping
    reservation is required. CPU/statistics/I/O inside these stages are included;
    this is neither kernel timing nor paid-allocation timing.
    """
    def __init__(self, name):
        self.name = name
        self.start = time.monotonic()
        self.previous = charged_seconds()
        self.cap = read(OUT/'PLAN.json')['gpu_wall_cap_seconds']
        self.check()

    def check(self):
        if self.previous + time.monotonic()-self.start >= self.cap:
            raise TimeoutError('Shared experimental command-stage wall budget reached')

    def finish(self, status, **metadata):
        receipt = dict(stage=self.name, status=status, seconds=time.monotonic()-self.start,
                       previous_seconds=self.previous, **metadata)
        save(OUT/'stage_receipts'/f'{self.name}_{time.time_ns()}.json', receipt)
        return receipt
