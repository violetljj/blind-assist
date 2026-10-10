"""Fixed official Training visits for additional Development calibration.

Ranges are checked and charged even on failure; ZIPs are constructed subsets.
Observations contain public RGB/K only. Native LiDAR truth stays evaluator-only.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile
import numpy as np
from PIL import Image
from rgb_body_query_3rscan import optical_z, sensor_labels
from rgb_body_query_fixed_grid import queries, independent_xyz_labels, PUBLIC
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_scene_diagnostic import write_csv

CONSUMED = {'47333462', '40777060', '40777065'}
MISSING = {'', 'NA', 'nan', 'NaN', None}
ASSETS = (('lowres_wide', '.png'), ('lowres_depth', '.png'),
          ('lowres_wide_intrinsics', '.pincam'))
FREE = 'FREE_ON_SAMPLED_RAYS'


class Allocation:
    """Network request wall and received payload bytes, separate from CPU wall."""
    def __init__(self, output, limits):
        self.output, self.limits = output, limits
        self.started = time.perf_counter()
        self.prior_wall_s = sum(json.loads(p.read_text('utf-8-sig'))['wall_s']
            for p in output.glob('preflight-failure-*/terminal_receipt.json'))
        self.net_s = 0.; self.received = 0; self.records = []
        self.inspect_s = 0.; self.processing_s = 0.

    def wall_remaining(self):
        remaining = self.limits['network_allocation_wall_s'] - self.prior_wall_s - (time.perf_counter()-self.started)
        if remaining <= 0: raise TimeoutError('Full source allocation wall exhausted')
        return remaining

    def request(self, url, method='GET', bounds=None):
        wall_remaining = self.wall_remaining()
        if self.net_s >= self.limits['network_allocation_wall_s']:
            raise TimeoutError('Cumulative network allocation exhausted')
        remaining = self.limits['received_bytes'] - self.received
        if bounds and bounds[1]-bounds[0]+1 > remaining:
            raise TimeoutError('Requested range exceeds remaining received-byte budget')
        record = dict(url=url, method=method, requested_range=bounds, bytes=0, status='STARTING')
        started = time.perf_counter(); payload = bytearray()
        try:
            headers = {'Range': f'bytes={bounds[0]}-{bounds[1]}'} if bounds else {}
            request = urllib.request.Request(url, headers=headers, method=method)
            timeout = min(30., wall_remaining, self.limits['network_allocation_wall_s']-self.net_s)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                record.update(http_status=response.status, headers=dict(response.headers))
                if method == 'HEAD':
                    if response.status != 200: raise ValueError('HEAD status must be200')
                    result = int(response.headers['Content-Length'])
                else:
                    if bounds:
                        a, b, size = bounds
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {a}-{b}/{size}':
                            raise ValueError('HTTP206 exact Content-Range mismatch')
                    expected = bounds[1]-bounds[0]+1 if bounds else remaining
                    while True:
                        self.wall_remaining()
                        if self.net_s + time.perf_counter()-started >= self.limits['network_allocation_wall_s']:
                            raise TimeoutError('Cumulative network allocation exhausted during read')
                        left = self.limits['received_bytes']-self.received
                        if left <= 0: raise TimeoutError('Received-byte allocation exhausted during read')
                        part = response.read(min(65536, left, expected+1-len(payload)))
                        self.received += len(part); record['bytes'] += len(part); payload.extend(part)
                        if not part: break
                        if len(payload) > expected: raise ValueError('Server sent more than requested')
                    if bounds and len(payload) != expected: raise ValueError('Range payload length mismatch')
                    result = bytes(payload)
                record['status'] = 'COMPLETE'
            return result
        except Exception as exc:
            record.update(status='FAILED', error=repr(exc)); raise
        finally:
            record['wall_s'] = time.perf_counter()-started; self.net_s += record['wall_s']
            record['received_sha256'] = hashlib.sha256(payload).hexdigest()
            if record['status'] == 'FAILED' and payload:
                path = self.output/f'failed-request-{len(self.records):04d}.partial'
                path.write_bytes(payload); record['partial_path'] = str(path)
            self.records.append(record)
            write(self.output/'network_progress.json', dict(network_request_wall_s=self.net_s,
                received_bytes=self.received, requests=self.records))


class BoundedRangeZip(io.RawIOBase):
    def __init__(self, url, size, allocation):
        self.url, self.size, self.allocation, self.pos = url, size, allocation, 0
        self.cache_start = -1; self.cache = b''
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, n, whence=0):
        self.pos = n if whence == 0 else self.pos+n if whence == 1 else self.size+n
        if self.pos < 0: raise ValueError('Negative ZIP seek')
        return self.pos
    def read(self, n=-1):
        n = max(0, self.size-self.pos) if n < 0 else min(n, max(0, self.size-self.pos))
        if n == 0: return b''
        result = bytearray()
        while len(result) < n:
            off = self.pos-self.cache_start
            if not 0 <= off < len(self.cache):
                a = self.pos
                remaining = self.allocation.limits['received_bytes']-self.allocation.received
                length = min(self.size-a, max(n-len(result), 65536), remaining)
                if length < n-len(result): raise TimeoutError('Range cannot fit received-byte budget')
                self.cache = self.allocation.request(self.url, bounds=(a, a+length-1, self.size))
                self.cache_start = a; off = 0
            take = min(n-len(result), len(self.cache)-off)
            result.extend(self.cache[off:off+take]); self.pos += take
        return bytes(result)


def choose(metadata):
    excluded_visits = {r['visit_id'] for r in metadata if r['video_id'] in CONSUMED and r['visit_id'] not in MISSING}
    selected = []; seen = set()
    for row in sorted(metadata, key=lambda r:int(r['video_id'])):
        visit = row['visit_id']
        if row['fold'] != 'Training' or row['video_id'] in CONSUMED or visit in MISSING or visit in excluded_visits or visit in seen:
            continue
        selected.append(row); seen.add(visit)
        if len(selected) == 3: break
    if len(selected) != 3: raise ValueError('Need exactly three new valid Training visits')
    return selected, sorted(excluded_visits)


def run(repo, root):
    source, output = root/'source', root/'reference'
    source.mkdir(parents=True, exist_ok=True); output.mkdir(exist_ok=True)
    if (source/'selection.json').exists(): raise FileExistsError('Preserve prior selection and failure')
    plan = json.loads((root/'plan.json').read_text('utf-8-sig')); limits = plan['budgets']
    allocation = Allocation(source, limits); started = time.perf_counter()
    receipt = dict(status='STARTING', source_sha256=sha(__file__), failures=[], captures=[],
        gpu_s=0, training_calls=0, inference_calls=0, network_budget_unit='Full source allocation wall including failed attempts; request wall also recorded',
        cpu_budget_unit='Stage wall excluding measured HTTP request wall; process CPU also recorded')
    shutil.copyfile(__file__, source/'executed_arkit_cal_reference.py')
    rows = []; records = []; checks = Counter()
    process_start = time.process_time(); active_processing_start = None
    try:
        inspect_start = time.perf_counter(); inspect_net_start = allocation.net_s
        prior = repo/'artifacts.local/work/rgb-body-query-query-level-dev-20261009/third-camera-source/arkit-additional'
        license_root = repo/'artifacts.local/work/rgb-body-query-transfer-dev-20261009/data-audit'
        license_receipt = json.loads((license_root/'source_receipt.json').read_text('utf-8-sig'))
        lr = next(r for r in license_receipt['rows'] if r['name'] == 'arkit_license')
        license_path = license_root/'arkit_license.txt'
        if lr['url'] != 'https://raw.githubusercontent.com/apple/ARKitScenes/main/LICENSE' or lr['status'] != 'COMPLETE' or sha(license_path) != lr['sha256']:
            raise ValueError('Inherited official Apple LICENSE identity mismatch')
        shutil.copyfile(license_path, source/'LICENSE.txt')
        write(source/'license_receipt.json', dict(**lr, inherited_path=str(license_path),
            source_receipt_sha256=sha(license_root/'source_receipt.json'), scope='Local licensed research; no redistribution; retained full terms'))
        metadata_path = prior/'raw_train_val_splits.csv'
        mr = json.loads((prior/'metadata_receipt.json').read_text('utf-8-sig'))
        api_path = prior/'metadata_api.json'
        # Prior API JSON was pretty-serialized; its receipt hashes raw response
        # bytes. Verify the decoded CSV through the retained official Git blob.
        if mr['status'] != 'COMPLETE' or mr['url'] != 'https://api.github.com/repos/apple-aiml-research/ARKitScenes/contents/raw/raw_train_val_splits.csv?ref=main':
            raise ValueError('Inherited official metadata API source mismatch')
        api = json.loads(api_path.read_text('utf-8-sig'))
        decoded = base64.b64decode(api['content'])
        if decoded != metadata_path.read_bytes() or hashlib.sha1(b'blob '+str(len(decoded)).encode()+b'\0'+decoded).hexdigest() != mr['git_blob_sha']:
            raise ValueError('Official CSV decoded Git blob identity mismatch')
        write(source/'metadata_identity_check.json', dict(decoded_csv_bytes_equal=True,
            git_blob_sha=mr['git_blob_sha'], api_serialized_sha256=sha(api_path),
            original_response_sha256=mr['sha256'], csv_sha256=sha(metadata_path),
            note='Raw HTTP response hash differs from pretty-serialized API JSON; decoded Git blob identity checked'))
        metadata = list(csv.DictReader(metadata_path.read_text('utf-8-sig').splitlines()))
        selected, excluded = choose(metadata); fixed = queries()
        for name in ('raw_train_val_splits.csv', 'metadata_api.json', 'DATA.md', 'metadata_receipt.json', 'data_doc_receipt.json'):
            shutil.copyfile(prior/name, source/name)
        selection = dict(captures=[r['video_id'] for r in selected], metadata_rows=selected,
            excluded_captures=sorted(CONSUMED), excluded_nonmissing_visit_ids=excluded,
            metadata_sha256=sha(metadata_path), inherited_metadata_receipt_sha256=sha(prior/'metadata_receipt.json'),
            rule=plan['selection'], queries=fixed, saved_before_asset_directory_or_pixel_reads=True,
            role='additional_cal only, official Training', source_kind='Constructed subsets; not entire official ZIPs')
        write(source/'selection.json', selection)
        print(json.dumps(dict(fixed_captures=selected)), flush=True)
        for selected_row in selected:
            allocation.wall_remaining()
            capture, visit = selected_row['video_id'], selected_row['visit_id']
            dest = source/capture; dest.mkdir(); sensor = output/capture; sensor.mkdir()
            archives = {}; streams = {}; indices = {}; details = {}; files = {}
            try:
                for asset, suffix in ASSETS:
                    url = f'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/Training/{capture}/{asset}.zip'
                    size = allocation.request(url, method='HEAD')
                    stream = BoundedRangeZip(url, size, allocation); streams[asset] = stream
                    archive = zipfile.ZipFile(stream); archives[asset] = archive
                    directory = [dict(name=i.filename, crc32=i.CRC, size=i.file_size,
                        compressed_size=i.compress_size, header_offset=i.header_offset) for i in archive.infolist()]
                    write(dest/f'{asset}_directory.json', directory)
                    index = {Path(n).stem:n for n in archive.namelist() if n.endswith(suffix)}
                    if len(index) != sum(n.endswith(suffix) for n in archive.namelist()): raise ValueError('Duplicate member stems')
                    indices[asset] = index
                    details[asset] = dict(url=url, official_archive_length_bytes=size,
                        directory_sha256=sha(dest/f'{asset}_directory.json'), directory_members=len(directory), constructed_subset=True)
                common = sorted(set.intersection(*(set(index) for index in indices.values())), key=lambda n:float(n.rsplit('_', 1)[1]))
                chosen = np.rint(np.linspace(0, len(common)-1, 16)).astype(int)
                if len(set(chosen)) != 16: raise ValueError('Insufficient exact timestamp intersection')
                names = [common[i] for i in chosen]
                write(dest/'selection.json', dict(capture=capture, visit_id=visit, common_pairs=len(common),
                    chosen_indices=chosen.tolist(), source_ids=names, queries=fixed, original_directories=details,
                    saved_before_selected_member_reads=True))
                allocation.inspect_s = time.perf_counter()-inspect_start-(allocation.net_s-inspect_net_start)-allocation.processing_s
                if allocation.inspect_s >= limits['source_inspection_s']: raise TimeoutError('Source inspection CPU wall allocation')
                for asset, _ in ASSETS:
                    member_rows = []
                    with zipfile.ZipFile(dest/f'{asset}.zip', 'w', compression=zipfile.ZIP_STORED) as subset:
                        for name in names:
                            allocation.wall_remaining()
                            member = indices[asset][name]; payload = archives[asset].read(member)
                            crc = zipfile.crc32(payload) & 0xffffffff
                            if crc != archives[asset].getinfo(member).CRC: raise ValueError('Member CRC mismatch')
                            files[asset, name] = payload; subset.writestr(member, payload)
                            member_rows.append(dict(name=member, crc32=crc, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
                    details[asset].update(subset_zip_sha256=sha(dest/f'{asset}.zip'), selected_members=member_rows)
                write(dest/'download_receipt.json', dict(status='COMPLETE', source_kind='Constructed subset ZIPs, NOT official entire archives',
                    capture=capture, visit_id=visit, original_archives=details, selection_sha256=sha(dest/'selection.json'),
                    HTTP206_exact_ContentRange_checked=True, all_selected_member_CRC_checked=True))
            finally:
                for archive in archives.values(): archive.close()
                for stream in streams.values(): stream.close()
            processing_start = time.perf_counter(); active_processing_start = processing_start; processing_before = allocation.processing_s
            capture_rows = []
            for frame, name in enumerate(names):
                allocation.wall_remaining()
                if processing_before + time.perf_counter()-processing_start >= limits['reference_processing_s']:
                    raise TimeoutError('Reference processing CPU wall allocation')
                rgb_bytes, depth_bytes, kb = (files[a, name] for a, _ in ASSETS)
                with Image.open(io.BytesIO(rgb_bytes)) as img: rgb_size = img.size
                with Image.open(io.BytesIO(depth_bytes)) as img: raw = np.asarray(img).copy()
                w,h,fx,fy,cx,cy = np.fromstring(kb.decode(), sep=' ')
                k = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)
                if (w,h) != (256,192) or rgb_size != (256,192) or raw.shape != (192,256) or raw.dtype != np.uint16:
                    raise ValueError('Native256x192 registered RGB/depth/pincam identity mismatch')
                if not np.isfinite(k).all() or min(fx,fy) <= 0: raise ValueError('Invalid public K')
                depth = optical_z(raw, 1000.); my,mx = np.indices(depth.shape, dtype=np.float32)
                observed = np.ones(depth.shape, bool); labels = []; states = []
                for index, query in enumerate(fixed):
                    label,state = sensor_labels(depth,k,query,observed)
                    independent,domain = independent_xyz_labels(depth,k,query,observed)
                    if not np.array_equal(label,independent): raise ValueError('Independent XYZ labels mismatch')
                    counts = {key:int((label == value).sum()) for key,value in [('positive_pixels',1),('free_ray_pixels',0),('unknown_pixels',2)]}
                    strict = 'POSITIVE' if counts['positive_pixels'] >= 16 else FREE if counts['free_ray_pixels'] >= 16 and not counts['positive_pixels'] and not counts['unknown_pixels'] else 'UNKNOWN'
                    if strict != state['state'] or sum(counts.values()) != state['domain_pixels'] or any(counts[key] != state[key] for key in counts):
                        raise ValueError('Strict state/count contract mismatch')
                    state.update(observed_reachable_rays=int((domain&observed).sum()))
                    labels.append(label); states.append(state); checks['independent_xyz_queries'] += 1
                    records.append(dict(environment='arkitscenes_'+capture, scan='arkitscenes_'+capture, visit_id=visit,
                        split='additional_cal', frame=frame, query=query['name'], query_index=index,
                        distance_band=f'{query["low"][2]:g}-{query["high"][2]:g}m', reference_state=strict,
                        negative_eligible=strict == FREE, domain_pixels=state['domain_pixels'], **counts,
                        observed_reachable_rays=state['observed_reachable_rays']))
                rp = sensor/f'{name}.png'; rp.write_bytes(rgb_bytes); ref = sensor/f'{name}.npz'
                np.savez_compressed(ref, labels=np.stack(labels), depth=depth, depth_K=k,
                    color_K=k, map_x=mx, map_y=my, observed=observed)
                row = dict(environment='arkitscenes_'+capture, scan='arkitscenes_'+capture, visit_id=visit,
                    split='additional_cal', official_split='Training', frame=frame, source_id=name,
                    timestamp_s=float(name.rsplit('_',1)[1]), rgb_path=str(rp), rgb_sha256=sha(rp),
                    reference_path=str(ref), reference_sha256=sha(ref), source_rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest(),
                    source_depth_sha256=hashlib.sha256(depth_bytes).hexdigest(), calibration_sha256=hashlib.sha256(kb).hexdigest(),
                    color_K=k.tolist(), depth_K=k.tolist(), color_shape=[192,256], depth_shape=[192,256], depth_shift=1000., queries=states)
                rows.append(row); capture_rows.append(row); checks['native_identity_frames'] += 1
            allocation.processing_s += time.perf_counter()-processing_start
            active_processing_start = None
            manifest = dict(status='COMPLETE_ADDITIONAL_CAL_REFERENCE', rows=capture_rows, queries=fixed,
                frames=16, groups=[dict(environment='arkitscenes_'+capture, scan='arkitscenes_'+capture, visit_id=visit, split='additional_cal')],
                role='Official Training additional_cal Development; no old eval reassignment',
                query_contract='Fixed27 camera-local strict sampled first-return queries; missing UNKNOWN; not body/volume clearance',
                source_kind='Constructed ZIP subsets; not whole official archives', source_selection_sha256=sha(dest/'selection.json'))
            write(sensor/'dataset_manifest.json', manifest)
            write(sensor/'observations.json', dict(rows=[{key:r[key] for key in PUBLIC} for r in capture_rows],
                queries=fixed, dataset_manifest_sha256=sha(sensor/'dataset_manifest.json')))
            receipt['captures'].append(dict(capture=capture, visit_id=visit, frames=16,
                state_counts=dict(Counter(q['state'] for r in capture_rows for q in r['queries'])),
                dataset_manifest_sha256=sha(sensor/'dataset_manifest.json')))
            print(json.dumps(receipt['captures'][-1]), flush=True)
        groups = [dict(environment='arkitscenes_'+r['video_id'],scan='arkitscenes_'+r['video_id'],visit_id=r['visit_id'],split='additional_cal') for r in selected]
        write(output/'dataset_manifest.json', dict(status='COMPLETE_ADDITIONAL_CAL_REFERENCE', rows=rows,
            queries=fixed, frames=len(rows), groups=groups, role=selection['role'], selection_sha256=sha(source/'selection.json')))
        observations = [{key:r[key] for key in PUBLIC} for r in rows]
        if any(set(r) != set(PUBLIC) for r in observations): raise ValueError('Evaluator field leak')
        write(output/'observations.json', dict(rows=observations,queries=fixed,dataset_manifest_sha256=sha(output/'dataset_manifest.json')))
        write_csv(output/'query_reference_counts.csv', records)
        write_csv(output/'negative_query_index.csv', [r for r in records if r['negative_eligible']])
        summaries = []
        for capture in [None] + selection['captures']:
            for band in [None, '0.3-0.8m', '0.8-1.5m', '1.5-3m']:
                subset = [r for r in records if (capture is None or r['scan'] == 'arkitscenes_'+capture) and (band is None or r['distance_band'] == band)]
                free = [r for r in subset if r['negative_eligible']]
                summaries.append(dict(capture=capture or 'all', distance_band=band or 'all', queries=len(subset),
                    positive_queries=sum(r['reference_state']=='POSITIVE' for r in subset), unknown_queries=sum(r['reference_state']=='UNKNOWN' for r in subset),
                    strict_free_queries=len(free), strict_free_visits=len({r['visit_id'] for r in free}), strict_free_frames=len({(r['scan'],r['frame']) for r in free})))
        write(output/'coverage.json', dict(rows=summaries, limits=plan['limits']))
        write_csv(output/'coverage.csv', summaries)
        near = next(r for r in summaries if r['capture']=='all' and r['distance_band']=='0.3-0.8m')
        receipt.update(status='COMPLETE', frames=len(rows), query_cells=len(records), checks=dict(checks),
            near_coverage=near, infer_fixed48=near['strict_free_visits'] >= 2,
            observations_evaluator_fields_absent=True, dataset_manifest_sha256=sha(output/'dataset_manifest.json'),
            observations_sha256=sha(output/'observations.json'), coverage_sha256=sha(output/'coverage.json'))
        if time.perf_counter()-started-allocation.net_s-allocation.processing_s >= limits['source_inspection_s']:
            raise TimeoutError('Source inspection CPU wall allocation')
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL', error=repr(exc)); raise
    finally:
        if active_processing_start is not None:
            allocation.processing_s += time.perf_counter()-active_processing_start
        allocation.inspect_s = max(0., time.perf_counter()-started-allocation.net_s-allocation.processing_s)
        receipt.update(wall_s=time.perf_counter()-started, cpu_process_s=time.process_time()-process_start,
            source_allocation_wall_s=time.perf_counter()-allocation.started+allocation.prior_wall_s,
            prior_failed_allocation_wall_s=allocation.prior_wall_s,
            network_request_wall_s=allocation.net_s, received_bytes=allocation.received,
            source_inspection_cpu_wall_s=allocation.inspect_s, reference_processing_cpu_wall_s=allocation.processing_s,
            network_requests=len(allocation.records), resource_status='HTTP streams and ZIPs closed; no retained process')
        write(source/'terminal_receipt.json', receipt); print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--repo',type=Path,required=True); parser.add_argument('--run-root',type=Path,required=True)
    args = parser.parse_args(); run(args.repo.resolve(),args.run_root.resolve())
