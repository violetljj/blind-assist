"""Fetch one legacy Nymeria sample's audit files using public ZIP byte ranges.

Whole-group SHA1 is not verified for selective downloads. ZipFile validates each
selected entry's CRC; receipts preserve member size, CRC, and local SHA256.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import requests


class RangeFile(io.RawIOBase):
    def __init__(self, session, url, size, budget):
        self.session, self.url, self.size, self.budget = session, url, size, budget
        self.pos = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos if whence == 1 else self.size) + offset
        if self.pos < 0:
            raise ValueError('negative ZIP offset')
        return self.pos

    def read(self, length=-1):
        length = self.size - self.pos if length < 0 else min(length, self.size - self.pos)
        if length <= 0:
            return b''
        if self.budget['bytes'] + length > self.budget['max_bytes']:
            raise RuntimeError('download byte budget exhausted')
        if time.monotonic() - self.budget['start'] > self.budget['max_seconds']:
            raise RuntimeError('download wall budget exhausted')
        end = self.pos + length - 1
        with self.session.get(self.url, headers={'Range': f'bytes={self.pos}-{end}',
                                               'Accept-Encoding': 'identity'},
                              timeout=(15, 45), stream=True) as response:
            response.raise_for_status()
            expected = f'bytes {self.pos}-{end}/{self.size}'
            if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                raise RuntimeError('server did not honor exact ZIP range; no full-download fallback')
            payload = bytearray()
            for block in response.iter_content(256 * 1024):
                self.budget['bytes'] += len(block)
                payload.extend(block)
                if self.budget['bytes'] > self.budget['max_bytes'] or len(payload) > length:
                    raise RuntimeError('download byte limit exceeded')
                if time.monotonic() - self.budget['start'] > self.budget['max_seconds']:
                    raise RuntimeError('download wall limit exceeded')
            if len(payload) != length:
                raise RuntimeError('short ZIP range')
            self.pos += length
            return bytes(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-bytes', type=int, default=1536 * 1024**2)
    parser.add_argument('--max-seconds', type=int, default=1800)
    parser.add_argument('--resume-receipt', type=Path)
    parser.add_argument('--receipt-name', default='download_receipt.json')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    required = ('metadata_json', 'body_motion', 'recording_head')
    eligible = [(sum(int(groups[g]['file_size_bytes']) for g in required), name)
                for name, groups in manifest['sequences'].items()
                if all(g in groups for g in required) and
                all(metadata.get(name, {}).get(k) for k in ('timesync', 'body_motion', 'head_slam'))]
    _, name = min(eligible)
    sample = args.output / name
    sample.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'RUNNING', 'sequence': name,
               'selection': 'minimum combined official group bytes among complete clock/body/head entries',
               'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               'files': [], 'archives': [], 'whole_zip_sha1_verified': False}
    budget = {'start': time.monotonic(), 'bytes': 0, 'max_bytes': args.max_bytes,
              'max_seconds': args.max_seconds}
    prior_seconds = 0.
    if args.resume_receipt:
        prior = json.loads(args.resume_receipt.read_text(encoding='utf-8'))
        if prior['sequence'] != name or prior['manifest_sha256'] != receipt['manifest_sha256']:
            raise ValueError('Resume source/sequence mismatch')
        receipt['files'] = prior['files']
        receipt['prior_attempt'] = str(args.resume_receipt)
        receipt['prior_error'] = prior.get('error')
        prior_seconds = prior['seconds']
        budget['bytes'] = prior['downloaded_bytes']
        budget['max_seconds'] -= prior_seconds
    receipt['missing_optional_files'] = []
    session = requests.Session()
    targets = {
        'body_motion': ('body/xdata.npz',),
        'recording_head': ('recording_head/data/motion.vrs',
                           'recording_head/mps/slam/closed_loop_trajectory.csv',
                           'recording_head/mps/slam/open_loop_trajectory.csv',
                           'recording_head/mps/slam/online_calibration.jsonl',
                           'recording_head/mps/slam/summary.json'),
    }
    try:
        # Metadata is a small direct file; verify its advertised whole-file SHA1.
        group = manifest['sequences'][name]['metadata_json']
        with session.get(group['download_url'], timeout=(15, 45)) as response:
            response.raise_for_status()
            payload = response.content
        budget['bytes'] += len(payload)
        if len(payload) != group['file_size_bytes'] or hashlib.sha1(payload).hexdigest() != group['sha1sum']:
            raise RuntimeError('metadata checksum/size mismatch')
        (sample / 'metadata.json').write_bytes(payload)
        receipt['files'].append({'path': 'metadata.json', 'bytes': len(payload),
                                 'sha256': hashlib.sha256(payload).hexdigest(), 'sha1_verified': True})
        print(f'SELECTED {name}', flush=True)
        for group_name, files in targets.items():
            group = manifest['sequences'][name][group_name]
            source = RangeFile(session, group['download_url'], int(group['file_size_bytes']), budget)
            with zipfile.ZipFile(source) as archive:
                members = archive.infolist()
                receipt['archives'].append({'group': group_name, 'advertised_bytes': group['file_size_bytes'],
                                             'advertised_sha1': group['sha1sum'],
                                             'members': [{'name': x.filename, 'bytes': x.file_size,
                                                          'compressed_bytes': x.compress_size} for x in members]})
                for target in files:
                    matches = [x for x in members if x.filename.replace('\\', '/').endswith(target)]
                    if not matches and target in (
                        'recording_head/mps/slam/open_loop_trajectory.csv',
                        'recording_head/mps/slam/online_calibration.jsonl',
                        'recording_head/mps/slam/summary.json'):
                        receipt['missing_optional_files'].append(target)
                        print(f'MISSING {target}', flush=True)
                        continue
                    if len(matches) != 1:
                        raise RuntimeError(f'required ZIP member absent/ambiguous: {target}')
                    member = matches[0]
                    destination = sample / target
                    old = next((x for x in receipt['files'] if x['path'] == target), None)
                    if old and destination.is_file() and destination.stat().st_size == old['bytes']:
                        with destination.open('rb') as reader:
                            current = hashlib.file_digest(reader, 'sha256').hexdigest()
                        if current != old['sha256']:
                            raise RuntimeError(f'Existing file failed resume checksum: {target}')
                        print(f'REUSE {target}', flush=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    print(f'FETCH {target} compressed={member.compress_size}', flush=True)
                    with archive.open(member) as reader, destination.open('wb') as writer:
                        while block := reader.read(4 * 1024**2):
                            digest.update(block)
                            writer.write(block)
                    receipt['files'].append({'path': target, 'bytes': member.file_size,
                                             'compressed_bytes': member.compress_size,
                                             'crc32': f'{member.CRC:08x}', 'crc32_verified': True,
                                             'sha256': digest.hexdigest()})
                    print(f'DONE {target}', flush=True)
        receipt['status'] = 'COMPLETE_AVAILABLE_FIELDS' if receipt['missing_optional_files'] else 'COMPLETE'
    except Exception as error:
        receipt['status'] = 'INCOMPLETE'
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        session.close()
        receipt['downloaded_bytes'] = budget['bytes']
        receipt['seconds'] = prior_seconds + time.monotonic() - budget['start']
        receipt_path = args.output / args.receipt_name
        with receipt_path.open('x', encoding='utf-8') as writer:
            writer.write(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
