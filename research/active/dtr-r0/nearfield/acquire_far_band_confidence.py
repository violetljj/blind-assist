"""Exact raw ARKit confidence members for the frozen far-band diagnostic roster."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from rgb_near_readout_scale_acquire import Allocation, RangeZip
from sync_rgb_tof_pilot_dev import load, save, sha


def capture_members(capture, selected, fold, destination, deadline):
    destination.mkdir(parents=True, exist_ok=True)
    allocation = Allocation(destination, max(.1, deadline-time.perf_counter()), 0,
                            download_limit=100_000_000)
    allocation.session.trust_env = False
    url = ('https://docs-assets.developer.apple.com/ml-research/datasets/'
           f'arkitscenes/v1/raw/{fold}/{capture}/confidence.zip')
    rows = []
    archive = stream = None
    started = time.perf_counter()
    try:
        size = allocation.request(url)
        stream = RangeZip(url, size, allocation)
        archive = zipfile.ZipFile(stream)
        indexed = {}
        for member in archive.infolist():
            if member.filename.endswith('.png'):
                stem = Path(member.filename).stem
                if stem in indexed:
                    raise ValueError('Duplicate confidence source stem')
                indexed[stem] = member
        directory = [dict(name=m.filename, crc32=m.CRC, size=m.file_size,
                          compressed_size=m.compress_size,
                          header_offset=m.header_offset)
                     for m in archive.infolist()]
        save(destination/'directory.json', directory)
        # Header order lets RangeZip reuse its 256KiB cache across adjacent frames.
        ordered = sorted(selected, key=lambda s: indexed[s].header_offset
                         if s in indexed else float('inf'))
        for source_id in ordered:
            allocation.remaining()
            if source_id not in indexed:
                rows.append(dict(source_id=source_id, capture=capture,
                                 official_fold=fold, status='MISSING_EXACT_TIMESTAMP'))
                continue
            member = indexed[source_id]
            payload = archive.read(member)  # ZIP CRC is verified by zipfile.
            array = np.asarray(Image.open(io.BytesIO(payload)))
            if array.shape != (192, 256) or array.dtype != np.uint8:
                raise ValueError(f'Confidence shape/type mismatch: {source_id}')
            values, counts = np.unique(array, return_counts=True)
            if not set(map(int, values)) <= {0, 1, 2}:
                raise ValueError(f'Confidence values outside 0/1/2: {source_id}')
            path = destination/(source_id+'.png')
            if path.exists():
                raise FileExistsError('Preserve existing confidence payload')
            path.write_bytes(payload)
            rows.append(dict(source_id=source_id, capture=capture,
                             official_fold=fold, status='AVAILABLE',
                             path=str(path.resolve()), sha256=sha(path),
                             member=member.filename, member_crc32=member.CRC,
                             shape=list(array.shape), dtype=str(array.dtype),
                             value_counts={str(int(v)): int(n) for v, n in zip(values, counts)}))
            save(destination/'progress.json', dict(rows=rows))
    except Exception as exc:
        completed = {r['source_id'] for r in rows}
        rows.extend(dict(source_id=s, capture=capture, official_fold=fold,
                         status='UNAVAILABLE_ACQUISITION', error=repr(exc))
                    for s in selected if s not in completed)
    finally:
        if archive is not None:
            archive.close()
        if stream is not None:
            stream.close()
        allocation.session.close()
    receipt = dict(capture=capture, official_fold=fold, url=url,
                   rows=rows, received_bytes=allocation.received,
                   request_count=len(allocation.records),
                   worker_wall_s=time.perf_counter()-started,
                   resources_closed=True)
    save(destination/'receipt.json', receipt)
    return receipt


def run(args):
    started = time.perf_counter()
    deadline = started+args.wall_s-5
    root = args.root.resolve()
    output = root/'confidence'
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise FileExistsError('No implicit confidence rerun')
    selection = load(root/'selection.json')
    groups = defaultdict(set)
    for row in selection['frames']:
        groups[str(row['capture'])].add(row['source_id'])
    metadata = {}
    with args.metadata.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if row['video_id'] in groups:
                if row['video_id'] in metadata:
                    raise ValueError('Duplicate official metadata capture')
                metadata[row['video_id']] = row
    assert set(groups) == set(metadata), 'Every selected capture needs an official fold'
    assert all(r['fold'] in ('Training', 'Validation') for r in metadata.values())
    save(output/'frozen_inputs.json', dict(selection_sha256=sha(root/'selection.json'),
         metadata_path=str(args.metadata.resolve()), metadata_sha256=sha(args.metadata),
         source_sha256=sha(__file__), selected_frames=sum(map(len, groups.values())),
         selected_captures=len(groups), official_rows=metadata,
         confidence_contract='raw AppleDepth uint8 0/1/2; same exact lowres source stem and 192x256 grid',
         cpu_command_wall_limit_s=args.wall_s, GPU_s=0))
    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(capture_members, cap, sorted(ids), metadata[cap]['fold'],
                               output/cap, deadline) for cap, ids in sorted(groups.items())]
        for future in concurrent.futures.as_completed(pending):
            receipts.append(future.result())
            print(json.dumps({k: receipts[-1][k] for k in
                              ('capture', 'received_bytes', 'worker_wall_s')}), flush=True)
    rows = sorted([r for receipt in receipts for r in receipt['rows']],
                  key=lambda r: r['source_id'])
    result = dict(status='COMPLETE' if all(r['status']=='AVAILABLE' for r in rows)
                  else 'COMPLETE_WITH_UNAVAILABLE', rows=rows,
                  selected_frames=sum(map(len, groups.values())),
                  available_frames=sum(r['status']=='AVAILABLE' for r in rows),
                  selected_captures=len(groups),
                  received_bytes=sum(r['received_bytes'] for r in receipts),
                  request_count=sum(r['request_count'] for r in receipts),
                  command_wall_s=time.perf_counter()-started, GPU_s=0,
                  selection_sha256=sha(root/'selection.json'),
                  metadata_sha256=sha(args.metadata), source_sha256=sha(__file__),
                  resources_closed=True, protected_data_access=False,
                  exact_timestamp_only=True)
    save(output/'manifest.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--wall-s', type=float, default=330)
    run(parser.parse_args())
