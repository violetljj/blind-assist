"""Losslessly package finalized UE source evidence with per-file hash readback.

Keeps every original file. ZIP packages are storage containers, not converted or
reduced modalities. Forecasts distinguish archive payload from retained originals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    coordinate = probability*(len(ordered)-1)
    lower = int(coordinate); upper = min(lower+1, len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(coordinate-lower)


def pack_files(root, paths, destination):
    """Write a fresh archive, verify CRC and SHA256 of every unpacked stream."""
    started = time.monotonic()
    files = []
    for path in paths:
        path = path.resolve()
        if not path.is_relative_to(root) or path == root or not path.is_file():
            raise ValueError('Packaging path outside capture or not a file: '+str(path))
        files.append(dict(path=path.relative_to(root).as_posix(), size=path.stat().st_size, sha256=sha(path)))
    if not files or len({row['path'] for row in files}) != len(files):
        raise ValueError('Archive members must be nonempty and unique')
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for row in files:
            archive.write(root/row['path'], row['path'])
    written = time.monotonic()
    with zipfile.ZipFile(destination, 'r') as archive:
        if archive.namelist() != [row['path'] for row in files]:
            raise ValueError('Archive membership mismatch')
        for row in files:
            digest = hashlib.sha256(); count = 0
            # Reading to EOF also checks the member CRC in Python's zipfile.
            with archive.open(row['path']) as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk); count += len(chunk)
            if count != row['size'] or digest.hexdigest() != row['sha256']:
                raise ValueError('Lossless archive readback failed: '+row['path'])
            # Preserve consistency if an unexpected writer changed an input mid-pack.
            if sha(root/row['path']) != row['sha256']:
                raise ValueError('Input changed during packaging: '+row['path'])
    verified = time.monotonic()
    return dict(archive_path=destination.relative_to(root).as_posix(), archive_sha256=sha(destination),
                archive_bytes=destination.stat().st_size, source_bytes=sum(row['size'] for row in files),
                source_files=files, encode_and_source_hash_s=written-started,
                verification_s=verified-written, wall_s=verified-started,
                crc_and_per_file_sha256_readback='PASS', originals_retained=True)


def pack(output):
    root = Path(output).resolve()
    format_receipt = json.loads((root/'format-receipt.json').read_text())
    if format_receipt['status'] != 'PASS_FORMAT_ONLY_GEOMETRIC_CANARY_REQUIRED':
        raise ValueError('A successful finalized capture is required; geometry acceptance remains separate')
    manifest = json.loads((root/'raw-manifest.json').read_text())
    rows = manifest['frames']
    if not rows or len(rows) != format_receipt['frame_count']:
        raise ValueError('Finalized frame count mismatch')
    target = root/'packed-raw'
    if target.exists():
        raise FileExistsError('Never overwrite a prior packaging attempt: '+str(target))
    target.mkdir()
    started = time.monotonic()
    receipt = dict(status='RUNNING', source_manifest_sha256=sha(root/'raw-manifest.json'),
        source_format_receipt_sha256=sha(root/'format-receipt.json'), code_sha256=sha(__file__),
        backend='TASK_NOT_GPU_SUITABLE_LOSSLESS_COMPRESSION_HASH_AND_IO',
        compression='ZIP_DEFLATE_LEVEL3_ALLOW_ZIP64', originals_retained=True, frames=[])
    try:
        for i, row in enumerate(rows):
            folder = (root/row['folder']).resolve()
            if not folder.is_relative_to(root/'frames') or not folder.is_dir():
                raise ValueError('Frame folder is not a captured frame directory')
            paths = sorted(p for p in folder.rglob('*') if p.is_file())
            archive = pack_files(root, paths, target/f'frame-{i:06d}.zip')
            receipt['frames'].append(dict(frame_id=row['id'], **archive))
            write(target/'progress.json', dict(status='RUNNING', packed_frames=i+1, expected=len(rows),
                  wall_s=time.monotonic()-started))
        # Geometry, frozen source and terminal/log evidence are also retained in a
        # separate common archive, without copying frame data for a second time.
        common = sorted(p for p in root.rglob('*') if p.is_file()
                        and not p.is_relative_to(root/'frames') and not p.is_relative_to(target))
        receipt['common'] = pack_files(root, common, target/'common-evidence.zip')
        packed = [row['archive_bytes'] for row in receipt['frames']]
        source = [row['source_bytes'] for row in receipt['frames']]
        timing = [row['wall_s'] for row in receipt['frames']]
        common_bytes = receipt['common']['archive_bytes']
        factor = 61440*1.3
        archive_forecast = sum(packed)/len(packed)*factor+common_bytes
        retained_forecast = (sum(packed)+sum(source))/len(packed)*factor+common_bytes+receipt['common']['source_bytes']
        receipt.update(status='PASS_LOSSLESS_ARCHIVES_VERIFIED_ORIGINALS_RETAINED',
            wall_s=time.monotonic()-started, frame_count=len(rows),
            statistics=dict(source_bytes_p50=percentile(source,.5), source_bytes_p95=percentile(source,.95),
                packed_bytes_p50=percentile(packed,.5), packed_bytes_p95=percentile(packed,.95),
                packed_bytes_mean=sum(packed)/len(packed), pack_verify_s_p50=percentile(timing,.5),
                pack_verify_s_p95=percentile(timing,.95), original_source_bytes=sum(source)+receipt['common']['source_bytes'],
                verified_archive_bytes=sum(packed)+common_bytes),
            budget_forecast=dict(time_samples=61440, reserve_multiplier=1.3, raw_evidence_budget_gib=250,
                archive_payload_gib=archive_forecast/2**30,
                originals_plus_archives_gib=retained_forecast/2**30,
                archive_payload_within_250gib=archive_forecast<=250*2**30,
                originals_plus_archives_within_250gib=retained_forecast<=250*2**30,
                compression_and_verify_hours=(sum(timing)/len(timing)*factor)/3600,
                assumptions='Mean of these captured frames; common archive counted once; full layout diversity/mesh growth, transfer replicas, later caches and peak batching require pilot measurement. No originals deleted.'))
    except BaseException:
        receipt.update(status='FAIL', error=traceback.format_exc(), wall_s=time.monotonic()-started)
        raise
    finally:
        write(target/'manifest.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = pack(args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ('frames', 'common')}, indent=2))
