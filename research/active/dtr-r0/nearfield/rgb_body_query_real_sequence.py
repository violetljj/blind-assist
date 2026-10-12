"""Acquire one fixed consumed SANPO-Real train clip, without models or truth claims.

The anchor comes from the existing Development proposal, before any model result.
Native RGB, two estimated depth maps, and panoptic masks stay separate. Pose is
referenced with its old hash; READY is a tracking flag, not verified pose accuracy.
"""
from __future__ import annotations

import base64
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
from pathlib import Path
import struct
import threading
import time
from urllib.parse import quote
import zlib

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO = Path(__file__).resolve().parents[4]
WORK = REPO / 'artifacts.local/work/rgb-body-query-dev-20261009/sequence'
OLD_WORK = REPO / 'artifacts.local/work/sanpo-coverage-20260927'
OLD_DATA = REPO / 'artifacts.local/datasets/sanpo-real-coverage-20260927'
SESSION = '-PqSDmiEe2pXjmYHgxh4YEBsj0T5LU10'
PREFIX = 'sanpo_dataset/v0/sanpo-real/'
API = 'https://storage.googleapis.com/storage/v1/b/gresearch/o'
KINDS = ('video_frames', 'depth_maps', 'zed_depth_maps', 'segmentation_masks')
LIMIT_BYTES = 268435456
BUDGET_S = 200.0


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), 'utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(resume=False):
    start = time.perf_counter()
    prior = read(WORK / 'acquisition_receipt.json') if resume else None
    elapsed_before = prior['elapsed_s'] if prior else 0.0
    deadline = start + max(0.0, BUDGET_S - elapsed_before)
    cancel = threading.Event()
    network_lock = threading.Lock()
    network_bytes = prior['download_payload_bytes'] if prior else 0
    files = list(prior['verified_files']) if prior else []
    failures, references = [], []
    plan = read(WORK / 'planned_files.json') if resume else None

    def check_time():
        if cancel.is_set() or time.perf_counter() >= deadline:
            cancel.set()
            raise TimeoutError('200 s metadata/download elapsed limit reached')

    def timeout():
        check_time()
        return (min(10, max(.1, deadline-time.perf_counter())),
                min(10, max(.1, deadline-time.perf_counter())))

    def listing(kind):
        result = []
        params = {'prefix': f'{PREFIX}{SESSION}/{camera}/left/{kind}/',
                  'maxResults': 1000,
                  'fields': 'items(name,size,md5Hash,crc32c,generation),nextPageToken'}
        with requests.Session() as client:
            while True:
                with client.get(API, params=params, timeout=timeout()) as response:
                    response.raise_for_status()
                    page = response.json()
                check_time()
                result.extend(page.get('items', []))
                if not page.get('nextPageToken'):
                    return result
                params['pageToken'] = page['nextPageToken']

    def acquire(item):
        nonlocal network_bytes
        check_time()
        old = OLD_DATA / item['relative']
        reused = old.exists()
        if reused:
            path = old
            payload = path.read_bytes()
        else:
            path = WORK / 'files' / item['relative']
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + '.partial')
            if path.exists() or partial.exists():
                raise FileExistsError(f'Preserve existing payload; do not restart it: {path}')
            url = f'https://storage.googleapis.com/gresearch/{quote(item["name"], safe="/")}'
            with requests.Session() as client:
                client.mount('https://', HTTPAdapter(max_retries=Retry(
                    total=2, connect=2, read=0, other=2, backoff_factor=1)))
                with client.get(url, params={'generation': item['generation']},
                                stream=True, timeout=timeout()) as response:
                    response.raise_for_status()
                    with partial.open('xb') as handle:
                        for chunk in response.iter_content(256 * 1024):
                            check_time()
                            with network_lock:
                                if network_bytes + len(chunk) > LIMIT_BYTES:
                                    cancel.set()
                                    raise ValueError('Download hard byte limit reached')
                                network_bytes += len(chunk)
                            handle.write(chunk)
            payload = partial.read_bytes()
        check_time()
        md5 = base64.b64encode(hashlib.md5(payload).digest()).decode()
        if len(payload) != int(item['size']) or md5 != item['md5Hash']:
            raise ValueError(f'Official size/MD5 mismatch: {item["name"]}')
        if item['kind'] in ('depth_maps', 'zed_depth_maps'):
            # gzip.decompress verifies each gzip member's CRC32 and ISIZE.
            raw = gzip.decompress(payload)
            height, width = map(int, struct.unpack('<ee', raw[:4]))
            if len(raw) != 4 + height * width * 2:
                raise ValueError('Invalid native float16 depth dimensions')
            shape = [height, width]
            format_integrity = 'GZIP_CRC32_ISIZE_VALIDATED'
        else:
            if payload[:8] != b'\x89PNG\r\n\x1a\n':
                raise ValueError('Invalid PNG signature')
            offset, shape, ended = 8, None, False
            while offset < len(payload):
                length = struct.unpack('>I', payload[offset:offset+4])[0]
                chunk = payload[offset+4:offset+8+length]
                crc = struct.unpack('>I', payload[offset+8+length:offset+12+length])[0]
                if zlib.crc32(chunk) != crc:
                    raise ValueError('PNG chunk CRC mismatch')
                if chunk[:4] == b'IHDR':
                    width, height = struct.unpack('>II', chunk[4:12])
                    shape = [height, width]
                if chunk[:4] == b'IEND':
                    ended = True
                offset += length + 12
            if not shape or not ended or offset != len(payload):
                raise ValueError('Incomplete native PNG')
            format_integrity = 'PNG_CHUNK_CRC_VALIDATED'
        if not reused:
            partial.replace(path)
        return {**item, 'path': str(path), 'physical_path': str(path.resolve()),
                'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                'reused': reused, 'shape': shape, 'format_integrity': format_integrity,
                'official_crc32c': item.get('crc32c'),
                'crc32c_verification': 'RETAINED_METADATA_NOT_RECOMPUTED'}

    if not WORK.resolve().is_relative_to((REPO / 'artifacts.local/work').resolve()):
        raise ValueError('Output must resolve inside canonical artifact work tree')
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        if resume:
            check_time()
            if prior['status'] == 'COMPLETE':
                raise ValueError('Prior acquisition already complete')
            for record in files:
                check_time()
                if sha(Path(record['path'])) != record['sha256']:
                    raise ValueError(f"Retained verified source changed: {record['path']}")
            completed = {record['name'] for record in files}
            candidate = [item for item in plan['files'] if item['name'] not in completed]
            required_bytes = sum(int(item['size']) for item in candidate)
            if network_bytes + required_bytes > LIMIT_BYTES:
                raise ValueError('Missing objects exceed remaining cumulative byte limit')
            frame_ids, camera = plan['frames'], plan['camera']
            print(json.dumps({'stage': 'RESUME_EXACT_MISSING', 'files': len(candidate),
                              'new_bytes': required_bytes,
                              'remaining_elapsed_s': max(0.0, deadline-time.perf_counter())}), flush=True)
        else:
            proposal = read(OLD_WORK / 'download_proposal.json')
            anchor = next(r for r in proposal['frames'] if r['session'] == SESSION)
            camera, anchor_frame = anchor['camera'], anchor['frame']
            old_receipt = read(OLD_WORK / 'acquisition_receipt.json')
            hashes = {r['path'].replace('\\', '/'): r for r in old_receipt['files']}
            for relative in (f'{SESSION}/description.json',
                             f'{SESSION}/{camera}/fixed_camera_poses.csv'):
                path = OLD_DATA / relative
                digest = sha(path)
                if digest != hashes[relative]['sha256']:
                    raise ValueError(f'Old source reference changed: {relative}')
                references.append({'path': str(path), 'physical_path': str(path.resolve()),
                                   'sha256': digest, 'old_receipt': hashes[relative]})
            description = read(OLD_DATA / SESSION / 'description.json')
            details = dict(zip(description['session_camera_location'],
                               description['session_camera_details']))[camera]
            fps = float(details['fps'])
            with (OLD_DATA / SESSION / camera / 'fixed_camera_poses.csv').open() as handle:
                poses = list(csv.DictReader(handle))
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(listing, KINDS))
            objects = {item['name']: item for result in results for item in result}
            write(WORK / 'object_metadata.json', list(objects.values()))
            # Annotation provenance is small metadata, not an inferred label quality flag.
            with requests.Session() as client:
                with client.get(f'https://storage.googleapis.com/gresearch/{PREFIX}{SESSION}/{camera}/left/frame_segmentation_annotation_type.json', timeout=timeout()) as response:
                    response.raise_for_status()
                    annotations = response.json()
            write(WORK / 'annotation_types.json', annotations)
            max_frame = min(len(poses), len(annotations)) - 1
            original_frames = list(range(max(0, min(anchor_frame-22, max_frame-44)),
                                         max(0, min(anchor_frame-22, max_frame-44))+45))
            for n_frames in range(45, 0, -1):
                left = max(0, min(anchor_frame-n_frames//2, max_frame-n_frames+1))
                frame_ids = list(range(left, left+n_frames))
                candidate, missing = [], []
                for frame in frame_ids:
                    for kind in KINDS:
                        extension = '.png' if kind in ('video_frames', 'segmentation_masks') else '.float16.gz'
                        relative = f'{SESSION}/{camera}/left/{kind}/{frame:06d}{extension}'
                        name = PREFIX + relative
                        if name not in objects:
                            missing.append(name)
                            continue
                        candidate.append({**objects[name], 'relative': relative, 'kind': kind,
                                          'frame': frame, 'reused_available': (OLD_DATA/relative).exists()})
                if missing:
                    raise ValueError(f'Missing exact requested objects: {missing}')
                required_bytes = sum(int(i['size']) for i in candidate if not i['reused_available'])
                if required_bytes <= LIMIT_BYTES:
                    break
            else:
                raise ValueError('Even one complete frame exceeds byte cap')
            plan = {'session': SESSION, 'camera': camera, 'anchor': anchor_frame,
                    'selection': 'first prior human anchor; visually selected Development session; no model output selection',
                    'requested_frames': original_frames, 'frames': frame_ids,
                    'fps': fps, 'nominal_duration_s': len(frame_ids)/fps,
                    'first_last_sample_span_s': (len(frame_ids)-1)/fps,
                    'length_reduced_for_bytes': len(frame_ids) < 45,
                    'planned_download_bytes': required_bytes, 'byte_limit': LIMIT_BYTES,
                    'total_object_bytes_including_reuse': sum(int(i['size']) for i in candidate),
                    'budget_elapsed_limit_s': BUDGET_S, 'metadata_elapsed_s': time.perf_counter()-start,
                    'references': references, 'files': candidate,
                    'frames_metadata': [{'frame': f, 'nominal_time_s': f/fps,
                                         'annotation_type': annotations.get(str(f), 'UNKNOWN'),
                                         'tracking_state': poses[f]['tracking_state']} for f in frame_ids],
                    'label_status': 'estimated visible geometry only; no real body-contact/event truth',
                    'pose_status': 'READY is not pose correctness; official residual pose issue unresolved',
                    'license': 'SANPO dataset CC BY 4.0'}
            write(WORK / 'planned_files.json', plan)
            print(json.dumps({'stage': 'PLANNED', 'frames': len(frame_ids),
                              'download_bytes': required_bytes}), flush=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(acquire, item): item for item in candidate}
            for future in as_completed(futures):
                try:
                    files.append(future.result())
                except Exception as exc:
                    failures.append({'object': futures[future]['name'], 'error': repr(exc)})
                    if isinstance(exc, TimeoutError):
                        cancel.set()
                write(WORK / 'progress.json', {'verified_files': len(files),
                      'failures': failures, 'network_payload_bytes': network_bytes,
                      'elapsed_s': elapsed_before + time.perf_counter()-start})
    except Exception as exc:
        cancel.set()
        failures.append({'stage': 'planning_or_download', 'error': repr(exc)})
    finally:
        cancel.set()
        counts = Counter(f['frame'] for f in files)
        complete = sorted(f for f, count in counts.items() if count == len(KINDS))
        receipt = {'status': 'COMPLETE' if plan and not failures and len(files)==len(plan['files']) else 'PARTIAL',
                   'session': SESSION, 'verified_files': files, 'failures': failures,
                   'download_payload_bytes': network_bytes,
                   'downloaded_verified_bytes': sum(f['bytes'] for f in files if not f['reused']),
                   'reused_verified_bytes': sum(f['bytes'] for f in files if f['reused']),
                   'complete_frames': complete,
                   'elapsed_s': elapsed_before + time.perf_counter()-start,
                   'attempt_elapsed_s': time.perf_counter()-start,
                   'prior_failures_preserved': prior['failures'] if prior else [],
                   'initial_executed_source_sha256': sha(WORK/'executed_source.py') if (WORK/'executed_source.py').exists() else sha(Path(__file__)),
                   'script_sha256': sha(Path(__file__)),
                   'annotation_counts': dict(Counter(f['annotation_type'] for f in plan['frames_metadata'])) if plan else {},
                   'tracking_counts': dict(Counter(f['tracking_state'] for f in plan['frames_metadata'])) if plan else {},
                   'resource_release': 'HTTP contexts closed; worker executor joined; partial payloads retained',
                   'evidence_role': 'real RGB Development acquisition; no model/event/metric-depth accuracy claim'}
        write(WORK / 'acquisition_receipt.json', receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('verified_files', 'complete_frames')}), flush=True)
    return 0 if receipt['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Repair only missing prior planned objects within cumulative limits')
    raise SystemExit(run(parser.parse_args().resume))
