"""Lossless capture packages and extraction-free directory/ZIP readers.

Pack/verify require only Python's standard library. Array/image readers import
NumPy/Pillow on demand. PNG bytes are stored unchanged; NPY bytes are deflated
without dtype conversion, so invalid values and evaluator truth are preserved.

    python data_lightweight.py pack --source CAPTURE --output CAPTURE.source.zip
    python data_lightweight.py verify --archive CAPTURE.source.zip --source CAPTURE

    with CompactSource(directory_or_zip) as source:
        native = source.load_array('scene/capture/evaluator/native/0000.npy')
        image = source.load_image('scene/capture/model/sample/0000.png')
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
import zipfile

FORMAT = 'blindassist-lossless-source-v1'
MANIFEST = '__data_lightweight__/manifest.json'
BLOCK = 1024 * 1024
REPARSE = 1024
ALREADY_COMPRESSED = {'.png', '.jpg', '.jpeg', '.webp', '.npz', '.zip', '.gz'}


def relative_path(value):
    value = str(value).replace('\\', '/')
    parts = value.split('/')
    if not value or any(p in {'', '.', '..'} or ':' in p for p in parts):
        raise ValueError(f'Expected a clean relative source path: {value!r}')
    return PurePosixPath(*parts).as_posix()


def _plain(path):
    info = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & REPARSE:
        raise ValueError(f'Reparse/symlink source is not admitted: {path}')
    return info


def _snapshot(info):
    return (info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino)


def source_files(root):
    """Inventory an exact source tree; fail rather than follow or omit aliases."""
    if not stat.S_ISDIR(_plain(root).st_mode):
        raise ValueError(f'Source is not a directory: {root}')
    pending, files = [root], {}
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                path = Path(entry.path)
                info = _plain(path)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    rel = relative_path(path.relative_to(root).as_posix())
                    if rel.casefold().startswith('__data_lightweight__/'):
                        raise ValueError('Source collides with package metadata namespace')
                    files[rel] = _snapshot(info)
                else:
                    raise ValueError(f'Non-regular source entry: {path}')
    if len(files) != len({name.casefold() for name in files}):
        raise ValueError('Source paths collide under Windows case folding')
    return dict(sorted(files.items()))


def digest_file(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(BLOCK):
            checksum.update(chunk)
    return checksum.hexdigest()


def _family(rel):
    path = PurePosixPath(rel)
    if path.suffix.lower() == '.npy':
        if path.parent.name == 'native':
            return 'native_depth'
        if path.parent.name == 'world_support':
            return 'world_support'
        return 'other_npy'
    if path.suffix.lower() == '.png':
        return 'png'
    return 'metadata_other'


def pack_source(source, output, level=6):
    """Create one new package; never modify or remove the source directory."""
    source = Path(source).absolute()
    _plain(source)
    source = source.resolve()
    output = Path(output).absolute()
    if output.resolve().is_relative_to(source):
        raise ValueError('Place the package outside its source directory')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + '.partial')
    files = source_files(source)
    if not files:
        raise ValueError('Empty source directory')
    rows, family = [], defaultdict(lambda: dict(files=0, source_bytes=0, compressed_bytes=0))
    started = time.perf_counter()
    owned_partial = False
    try:
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=level, allowZip64=True) as archive:
            owned_partial = True
            for rel, snapshot in files.items():
                path = source / rel
                if _snapshot(_plain(path)) != snapshot:
                    raise ValueError(f'Source changed before packing: {rel}')
                method = zipfile.ZIP_STORED if path.suffix.lower() in ALREADY_COMPRESSED else zipfile.ZIP_DEFLATED
                # A string uses the archive's compression level; a PNG ZipInfo
                # explicitly selects STORE while retaining the exact input bytes.
                member = rel
                if method == zipfile.ZIP_STORED:
                    member = zipfile.ZipInfo(rel)
                    member.compress_type = zipfile.ZIP_STORED
                checksum = hashlib.sha256()
                with path.open('rb') as incoming, archive.open(member, 'w', force_zip64=True) as outgoing:
                    while chunk := incoming.read(BLOCK):
                        checksum.update(chunk)
                        outgoing.write(chunk)
                if _snapshot(_plain(path)) != snapshot:
                    raise ValueError(f'Source changed during packing: {rel}')
                info = archive.getinfo(rel)
                kind = _family(rel)
                row = dict(path=rel, bytes=snapshot[0], mtime_ns=snapshot[1],
                           sha256=checksum.hexdigest(), family=kind,
                           compression='stored' if method == zipfile.ZIP_STORED else 'deflate',
                           compressed_bytes=info.compress_size)
                rows.append(row)
                family[kind]['files'] += 1
                family[kind]['source_bytes'] += row['bytes']
                family[kind]['compressed_bytes'] += row['compressed_bytes']
            if source_files(source) != files:
                raise ValueError('Source tree changed while packing')
            manifest = dict(format=FORMAT, original_root=str(source), entries=rows,
                            preservation='Exact original file bytes; no quantization, relabeling, or authority change',
                            compression=dict(algorithm='zip-deflate', level=level, png='stored'),
                            source_bytes=sum(r['bytes'] for r in rows))
            archive.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode('utf-8'))
        # rename refuses an existing destination on the managed Windows host.
        if output.exists():
            raise FileExistsError(output)
        partial.rename(output)
        owned_partial = False
    finally:
        if owned_partial:
            partial.unlink(missing_ok=True)
    encode_seconds = time.perf_counter() - started
    packed_bytes = output.stat().st_size
    return dict(status='PASS', format=FORMAT, source=str(source), archive=str(output),
                files=len(rows), source_bytes=manifest['source_bytes'], archive_bytes=packed_bytes,
                saved_bytes=manifest['source_bytes']-packed_bytes,
                saved_fraction=1-packed_bytes/manifest['source_bytes'],
                encode_seconds=encode_seconds, archive_sha256=digest_file(output),
                families=dict(family), source_unchanged=True, source_deletions=0,
                backend='CPU stdlib zipfile/deflate; TASK_NOT_GPU_SUITABLE',
                verification='Run verify for full decompression/hash verification')


class CompactSource:
    """Use identical relative paths with a plain directory or a compact package.

    Calls read one requested member and never unpack the full source. `open`
    returns a binary file-like context manager usable by numpy.load/Image.open.
    `load_array` deliberately does not offer mmap for compressed members.
    """

    def __init__(self, path):
        self.path = Path(path).absolute()
        _plain(self.path)
        self.path = self.path.resolve()
        self.archive = None
        self.manifest = None
        self.entries = None
        if self.path.is_dir():
            return
        try:
            self.archive = zipfile.ZipFile(self.path, 'r')
            infos = self.archive.infolist()
            names = [relative_path(i.filename) for i in infos]
            if len(names) != len(set(names)) or len(names) != len({n.casefold() for n in names}):
                raise ValueError('Duplicate logical package paths')
            if MANIFEST not in names:
                raise ValueError('Package manifest is missing')
            if self.archive.getinfo(MANIFEST).file_size > 32 * 1024 * 1024:
                raise ValueError('Package manifest exceeds 32 MiB')
            self.manifest = json.loads(self.archive.read(MANIFEST))
            if self.manifest.get('format') != FORMAT:
                raise ValueError('Unsupported compact-source format')
            rows = self.manifest['entries']
            keys = [relative_path(row['path']) for row in rows]
            if len(keys) != len(set(keys)) or len(keys) != len({k.casefold() for k in keys}):
                raise ValueError('Duplicate manifest paths')
            if MANIFEST in keys or set(keys) != set(names)-{MANIFEST}:
                raise ValueError('Package/manifest file set mismatch')
            self.entries = dict(zip(keys, rows))
            for key, row in self.entries.items():
                info = self.archive.getinfo(key)
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('Package must contain ordinary file members')
                if row['bytes'] != info.file_size or len(row['sha256']) != 64:
                    raise ValueError(f'Invalid package entry: {key}')
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.archive is not None:
            self.archive.close()

    def paths(self):
        return list(self.entries) if self.entries is not None else list(source_files(self.path))

    @contextmanager
    def open(self, relative):
        relative = relative_path(relative)
        if self.archive is not None:
            if relative not in self.entries:
                raise FileNotFoundError(relative)
            with self.archive.open(relative) as stream:
                yield stream
        else:
            path = self.path
            for part in PurePosixPath(relative).parts:
                path = path / part
                _plain(path)
            with path.open('rb') as stream:
                yield stream

    def read_bytes(self, relative):
        with self.open(relative) as stream:
            return stream.read()

    def read_json(self, relative):
        return json.loads(self.read_bytes(relative).decode('utf-8-sig'))

    def load_array(self, relative):
        import numpy as np
        if PurePosixPath(relative_path(relative)).suffix.lower() != '.npy':
            raise ValueError('load_array reads one .npy member, not a lazy .npz collection')
        with self.open(relative) as stream:
            return np.load(stream, allow_pickle=False)

    def load_image(self, relative):
        from PIL import Image
        with self.open(relative) as stream, Image.open(stream) as image:
            image.load()
            return image.copy()


def verify_source(archive_path, source=None):
    """Stream every entry, check SHA-256, optionally compare current source bytes."""
    started = time.perf_counter()
    source_path = Path(source).absolute() if source is not None else None
    if source_path is not None:
        _plain(source_path)
        source_path = source_path.resolve()
    files, bytes_read, source_equal = 0, 0, source_path is not None
    with CompactSource(archive_path) as compact:
        if compact.archive is None:
            raise ValueError('Verification requires a compact archive')
        if source_path is not None and set(source_files(source_path)) != set(compact.entries):
            raise ValueError('Source/package file set differs')
        original_context = CompactSource(source_path) if source_path is not None else nullcontext(None)
        with original_context as original:
            for rel, expected in compact.entries.items():
                checksum, count = hashlib.sha256(), 0
                with compact.open(rel) as incoming:
                    if original is not None:
                        with original.open(rel) as plain:
                            while chunk := incoming.read(BLOCK):
                                if chunk != plain.read(len(chunk)):
                                    raise ValueError(f'Roundtrip/source bytes differ: {rel}')
                                checksum.update(chunk)
                                count += len(chunk)
                            if plain.read(1):
                                raise ValueError(f'Source is longer than packed member: {rel}')
                    else:
                        while chunk := incoming.read(BLOCK):
                            checksum.update(chunk)
                            count += len(chunk)
                if count != expected['bytes'] or checksum.hexdigest() != expected['sha256']:
                    raise ValueError(f'Roundtrip hash/size mismatch: {rel}')
                files += 1
                bytes_read += count
    return dict(status='PASS', archive=str(Path(archive_path).resolve()), files=files,
                decoded_bytes=bytes_read, decode_verify_seconds=time.perf_counter()-started,
                all_sha256_match=True, source_byte_comparison=source_equal,
                decompression_is_lossless=True, extracted_files=0, source_deletions=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    pack = actions.add_parser('pack')
    pack.add_argument('--source', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--level', type=int, choices=range(1, 10), default=6)
    pack.add_argument('--receipt', type=Path)
    verify = actions.add_parser('verify')
    verify.add_argument('--archive', type=Path, required=True)
    verify.add_argument('--source', type=Path)
    verify.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.action == 'pack':
        result = pack_source(args.source, args.output, args.level)
        receipt = args.receipt or args.output.with_name(args.output.name+'.pack.json')
    else:
        result = verify_source(args.archive, args.source)
        receipt = args.receipt or args.archive.with_name(args.archive.name+'.verify.json')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
