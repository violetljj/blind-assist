"""Verify clean CARLA research checkouts retain committed raw bytes on every OS.

This is a checkout regression check, not certification of historical frozen locks.
Never normalizes files or rewrites protocol hashes. Run after a clean checkout;
intentional uncommitted research edits correctly fail this check.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys


def check(root: Path) -> int:
    records = subprocess.check_output(
        ['git', 'ls-tree', '-rz', 'HEAD', '--', 'research/active/dtr-r0/carla/'], cwd=root
    ).split(b'\0')
    failures = []
    checked = 0
    with subprocess.Popen(
        ['git', 'cat-file', '--batch'], cwd=root,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    ) as blobs:
        assert blobs.stdin is not None and blobs.stdout is not None
        for record in records:
            if not record:
                continue
            metadata, raw_path = record.split(b'\t', 1)
            mode, kind, oid = metadata.split()
            if kind != b'blob' or mode == b'120000':
                continue
            path = raw_path.decode('utf-8')
            blobs.stdin.write(oid + b'\n')
            blobs.stdin.flush()
            header = blobs.stdout.readline().split()
            expected = blobs.stdout.read(int(header[2]))
            if blobs.stdout.read(1) != b'\n':
                raise RuntimeError('Invalid git cat-file framing')
            candidate = root / path
            if not candidate.is_file():
                failures.append(f'{path}: missing')
                continue
            actual = candidate.read_bytes()
            checked += 1
            if hashlib.sha256(actual).digest() != hashlib.sha256(expected).digest():
                reason = ('line endings only' if actual.replace(b'\r\n', b'\n') ==
                          expected.replace(b'\r\n', b'\n') else 'content differs')
                failures.append(f'{path}: {reason}')
        blobs.stdin.close()
        blobs.wait()
        if blobs.returncode:
            raise RuntimeError(f'git cat-file failed: {blobs.returncode}')
    for failure in failures[:20]:
        print(failure, file=sys.stderr)
    print(f'CARLA checkout byte identity: {checked} files, {len(failures)} mismatches')
    return int(bool(failures) or checked == 0)


if __name__ == '__main__':
    raise SystemExit(check(Path(__file__).resolve().parents[1]))
