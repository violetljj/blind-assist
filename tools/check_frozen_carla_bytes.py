"""Check selected CARLA raw-byte locks against their original protocol fields.

No simulation, hash normalization, historical source substitution, or repair.
The manifest selects locks; it never supplies replacement expected hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys


MANIFEST = 'tools/frozen_carla_byte_locks.json'


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    path.relative_to(root.resolve(strict=True))
    return path


def check(root: Path) -> int:
    rows = json.loads((root / MANIFEST).read_text(encoding='utf-8'))['locks']
    if not rows:
        raise ValueError('Frozen lock manifest is empty')
    seen = set()
    failures = []
    for row in rows:
        name = row['path']
        if name in seen:
            raise ValueError(f'Duplicate frozen target: {name}')
        seen.add(name)
        try:
            expected = json.loads(inside(root, row['lock_source']).read_text(encoding='utf-8'))
            for key in row['lock_pointer']:
                expected = expected[key]
            if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
                raise ValueError('Lock pointer must resolve to a SHA-256 string')
            actual = hashlib.sha256(inside(root, name).read_bytes()).hexdigest()
            if actual != expected.lower():
                failures.append(f'{name}: frozen hash mismatch ({actual})')
        except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
            failures.append(f'{name}: {error}')
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f'CARLA frozen raw-byte locks: {len(rows)} checked, {len(failures)} failures')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(check(Path(__file__).resolve().parents[1]))
