"""Portable, explicit-scope ZIP backup and fresh-directory restore verification.

Only standard library; never follows nested links or overwrites a destination.
The inventory proves byte preservation, not scientific dependency completeness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or path.is_absolute()
            or '..' in path.parts or name != path.as_posix()):
        raise ValueError(f'Unsafe member: {name}')
    return path


def linked(path):
    attributes = getattr(path.lstat(), 'st_file_attributes', 0)
    return path.is_symlink() or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def create(root, selection, archive):
    root = Path(root).absolute()
    files = {}
    for name in selection['paths']:
        safe_name(name)
        source = root / name
        if not source.exists():
            raise FileNotFoundError(source)
        # artifacts.local is the explicitly allowed canonical junction. Reject
        # every other traversed ancestor as well as links inside selected trees.
        for ancestor in (source, *source.parents):
            if ancestor == root:
                break
            if linked(ancestor) and ancestor != root / 'artifacts.local':
                raise ValueError(f'Linked input: {ancestor}')
        candidates = [source]
        if source.is_dir():
            candidates = []
            for base, dirs, leaves in os.walk(source, followlinks=False):
                for child in dirs + leaves:
                    if linked(Path(base) / child):
                        raise ValueError(f'Linked input: {Path(base) / child}')
                candidates.extend(Path(base) / leaf for leaf in leaves)
        for path in candidates:
            name = path.relative_to(root).as_posix()
            if name == 'backup-manifest.json':
                raise ValueError('Reserved manifest name')
            files[name] = path
    manifest = {'schema': 'blindassist-scoped-backup-v1', 'scope': selection,
                'files': [], 'dependency_complete': False}
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=1) as out:
        for name, path in sorted(files.items()):
            before = path.stat()
            sha = digest(path)
            out.write(path, name)
            after = path.stat()
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise RuntimeError(f'Input changed during backup: {name}')
            manifest['files'].append({'path': name, 'bytes': before.st_size, 'sha256': sha})
        manifest['bytes'] = sum(f['bytes'] for f in manifest['files'])
        out.writestr('backup-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def restore(archive, destination):
    display_destination = str(Path(destination).absolute())
    # Python/Windows deployments may lack the machine-wide long-path option.
    # Extend only the filesystem destination; ZIP member names stay portable.
    if os.name == 'nt' and not display_destination.startswith('\\\\?\\'):
        extended = ('\\\\?\\UNC\\' + display_destination[2:] if display_destination.startswith('\\\\')
                    else '\\\\?\\' + display_destination)
        destination = Path(extended)
    else:
        destination = Path(display_destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as incoming:
        names = incoming.namelist()
        for name in names:
            safe_name(name)
        if len(set(names)) != len(names) or len({n.casefold() for n in names}) != len(names):
            raise ValueError('Duplicate archive members')
        manifest = json.loads(incoming.read('backup-manifest.json'))
        entries = manifest['files']
        expected = {f['path'] for f in entries}
        if len(expected) != len(entries) or set(names) != expected | {'backup-manifest.json'}:
            raise ValueError('Archive inventory mismatch')
        for item in entries:
            name = item['path']
            target = destination / safe_name(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with incoming.open(name) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if target.stat().st_size != item['bytes'] or digest(target) != item['sha256']:
                raise ValueError(f'Restored hash mismatch: {name}')
        (destination / 'backup-manifest.json').write_bytes(incoming.read('backup-manifest.json'))
    return {'status': 'PASS', 'files': len(entries), 'bytes': sum(f['bytes'] for f in entries),
            'archive_sha256': digest(archive), 'destination': display_destination,
            'dependency_complete': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['create', 'restore'])
    parser.add_argument('--archive', required=True)
    parser.add_argument('--root')
    parser.add_argument('--selection')
    parser.add_argument('--destination')
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    if Path(args.receipt).exists():
        raise FileExistsError(args.receipt)
    if args.action == 'create':
        result = create(args.root, json.loads(Path(args.selection).read_text(encoding='utf-8')), args.archive)
    else:
        result = restore(args.archive, args.destination)
    with Path(args.receipt).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'action': args.action, 'files': len(result['files']) if isinstance(result['files'], list) else result['files'], 'bytes': result['bytes']}))


if __name__ == '__main__':
    main()
