"""Check CARLA protocol bytes and frozen component source against original locks.

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


def component_locks(value: object, source: str):
    """Read every entry of the two existing component-lock formats."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'frozen_component_sha256':
                for name, expected in child.items():
                    yield str(Path(source).parent / name), expected
            elif key == 'implementation_locks':
                for row in child:
                    yield row['path'], row['sha256']
            else:
                yield from component_locks(child, source)
    elif isinstance(value, list):
        for child in value:
            yield from component_locks(child, source)


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    path.relative_to(root.resolve(strict=True))
    return path


def check(root: Path) -> int:
    manifest = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    rows = manifest['locks']
    if not rows:
        raise ValueError('Frozen lock manifest is empty')
    seen = set()
    failures = []
    checked = set()

    def verify(name: str, expected: str) -> None:
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
            raise ValueError('Lock must resolve to a SHA-256 string')
        identity = (Path(name).as_posix(), expected.lower())
        if identity in checked:
            return
        checked.add(identity)
        actual = hashlib.sha256(inside(root, name).read_bytes()).hexdigest()
        if actual != expected.lower():
            failures.append(f'{name}: frozen hash mismatch ({actual})')

    for row in rows:
        name = row['path']
        if name in seen:
            raise ValueError(f'Duplicate frozen target: {name}')
        seen.add(name)
        try:
            expected = json.loads(inside(root, row['lock_source']).read_text(encoding='utf-8'))
            for key in row['lock_pointer']:
                expected = expected[key]
            verify(name, expected)
        except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
            failures.append(f'{name}: {error}')
    references = 0
    for source in manifest.get('component_protocols', []):
        try:
            protocol = json.loads(inside(root, source).read_text(encoding='utf-8'))
            locks = list(component_locks(protocol, source))
            if not locks:
                raise ValueError('Component protocol contains no locks')
            references += len(locks)
            for name, expected in locks:
                verify(name, expected)
        except (OSError, KeyError, IndexError, TypeError, ValueError, AttributeError) as error:
            failures.append(f'{source}: {error}')
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f'CARLA frozen raw-byte locks: {len(rows)} protocol references, '
          f'{references} component references, {len(checked)} unique locks, '
          f'{len(failures)} failures')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(check(Path(__file__).resolve().parents[1]))
