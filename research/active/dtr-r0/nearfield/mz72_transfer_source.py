"""Build an MZ72 transfer hardlink view, retaining raw native audits at their owner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ordinary(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
        raise ValueError(f'Reparse source/view is not admitted: {path}')
    return info


def build(source, output, receipt):
    start = time.perf_counter()
    source, output, receipt = map(Path, (source, output, receipt))
    ordinary(source)
    source = source.resolve(strict=True)
    output, receipt = output.resolve(), receipt.resolve()
    if source.name != 'training-source' or source.parent.name != 'dataset-v1':
        raise ValueError('Expected a completed dataset-v1/training-source')
    shard = source.parent.parent
    if output != shard / 'transfer-source' or receipt != shard / 'transfer-view.json':
        raise ValueError('View and receipt must be the exact sibling shard paths')
    if output.exists() or receipt.exists():
        raise FileExistsError('Preserve the prior view/receipt; do not overwrite')
    ordinary(shard)
    ordinary(source.parent)
    rows, retained = [], []
    for directory, dirs, files in os.walk(source, followlinks=False):
        ordinary(Path(directory))
        for name in dirs:
            ordinary(Path(directory) / name)
        for name in sorted(files):
            path = Path(directory) / name
            info = ordinary(path)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f'Non-file source: {path}')
            relative = path.relative_to(source).as_posix()
            if relative == 'retained-native.json':
                raise ValueError('Source already contains transfer-only metadata')
            row = dict(path=relative, owner_path=str(path), bytes=info.st_size,
                       sha256=sha(path))
            (retained if relative.startswith('evaluator/native/') else rows).append(row)
    if not rows:
        raise ValueError('Empty training source')
    output.mkdir()
    for row in rows:
        src, dst = source / row['path'], output / row['path']
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        if not os.path.samefile(src, dst) or sha(dst) != row['sha256']:
            raise ValueError(f'Hardlink byte verification failed: {row["path"]}')
    native_record = dict(schema='mz72-native-retained-at-owner-v1',
                         original_source=str(source), excluded_prefix='evaluator/native/',
                         native_transferred=False, files=retained)
    marker = output / 'retained-native.json'
    marker.write_text(json.dumps(native_record, indent=2) + '\n', encoding='utf-8')
    result = dict(status='PASS', schema='mz72-transfer-view-v1', source=str(source),
                  output=str(output), excluded_prefix='evaluator/native/',
                  linked_files=len(rows), linked_bytes=sum(r['bytes'] for r in rows),
                  retained_native_files=len(retained),
                  retained_native_bytes=sum(r['bytes'] for r in retained),
                  source_files=rows, retained_native=retained,
                  marker_sha256=sha(marker), source_deletions=0,
                  all_other_members_byte_identical=True, hardlink_view=True,
                  seconds=time.perf_counter()-start)
    receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.output, args.receipt)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('source_files', 'retained_native')}))
