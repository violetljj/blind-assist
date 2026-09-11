"""Explicit lossless WebP RGB view; originals and existing output roots are immutable."""
import argparse
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import time

from PIL import Image, features, __version__ as PILLOW_VERSION
from data_lightweight import CompactSource, relative_path

ENCODER = dict(format='WEBP', lossless=True, quality=100, method=0, exact=True)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def rgba(data):
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.size, image.convert('RGBA').tobytes()


def build(manifest, output):
    """Accept explicit file or compact-archive members; emit an ordered filename map.

    Fresh output only: an existing root is rejected even if bytes match, avoiding
    implicit resume/overwrite. Failures preserve partial evidence for inspection.
    """
    start = time.perf_counter()
    manifest = Path(manifest).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists; nothing overwritten: {output}')
    manifest_sha = sha(manifest)
    document = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if document.get('schema') != 'compact-rgb-view-input-v1' or not document.get('images'):
        raise ValueError('Expected nonempty compact-rgb-view-input-v1 images')
    if not features.check('webp'):
        raise RuntimeError('Installed Pillow lacks WebP support; no installation attempted')
    images = document['images']
    paths = [relative_path(row['output']) for row in images]
    if any(not p.lower().endswith('.webp') for p in paths):
        raise ValueError('Every output path must have .webp suffix')
    if len(set(p.casefold() for p in paths)) != len(paths):
        raise ValueError('Output paths collide under case folding')
    if len({r['frame_id'] for r in images}) != len(images):
        raise ValueError('Duplicate frame_id')
    # Validate identities and source kind before creating the view.
    for row in images:
        source = row['source']
        kind = set(source)
        if kind not in ({'path', 'sha256'}, {'archive', 'archive_sha256', 'member', 'sha256'}):
            raise ValueError('Source must explicitly name one PNG file or archive member')
        container = Path(source.get('path', source.get('archive')))
        if not container.is_absolute():
            raise ValueError('Source container paths must be absolute')
        name = source.get('member', container.name)
        if 'member' in source:
            relative_path(name)
        if not name.lower().endswith('.png'):
            raise ValueError('Only explicitly listed original PNGs are accepted')
    output.mkdir(parents=True)
    rows, inputs, sources = [], {str(manifest): manifest_sha}, {}
    timing = dict(source_read=0., decode=0., encode=0., verify_decode=0., write=0.)
    write_json(output/'start.json', dict(schema=document['schema'], manifest_sha256=manifest_sha,
        requested_frames=len(images), encoder=ENCODER, code_sha256=sha(__file__),
        pillow=PILLOW_VERSION, webp=features.version('webp'), sequential=True))
    try:
        with ExitStack() as stack:
            for row, destination in zip(images, paths):
                source = row['source']
                container = Path(source.get('path', source.get('archive'))).resolve(strict=True)
                expected = source.get('archive_sha256', source['sha256'])
                if str(container) not in inputs:
                    if sha(container) != expected:
                        raise ValueError(f'Source container SHA mismatch: {container}')
                    inputs[str(container)] = expected
                elif inputs[str(container)] != expected:
                    raise ValueError('Conflicting SHA authority for one source container')
                t = time.perf_counter()
                if 'archive' in source:
                    if container not in sources:
                        sources[container] = stack.enter_context(CompactSource(container))
                    data = sources[container].read_bytes(source['member'])
                else:
                    data = container.read_bytes()
                timing['source_read'] += time.perf_counter()-t
                if digest(data) != source['sha256']:
                    raise ValueError(f'Original PNG SHA mismatch: {row["frame_id"]}')
                t = time.perf_counter()
                with Image.open(io.BytesIO(data)) as image:
                    if image.format != 'PNG' or image.mode not in ('RGB', 'RGBA') or getattr(image, 'n_frames', 1) != 1:
                        raise ValueError('Only single-frame 8-bit RGB/RGBA PNGs are supported')
                    image.load()
                    size, mode = image.size, image.mode
                    reference = image.convert('RGBA').tobytes()
                    timing['decode'] += time.perf_counter()-t
                    buffer = io.BytesIO()
                    t = time.perf_counter(); image.save(buffer, **ENCODER)
                    encode_seconds = time.perf_counter()-t; timing['encode'] += encode_seconds
                    encoded = buffer.getvalue()
                t = time.perf_counter()
                restored_size, restored = rgba(encoded)
                if restored_size != size or restored != reference:
                    raise ValueError(f'Decoded RGBA differs: {row["frame_id"]}')
                timing['verify_decode'] += time.perf_counter()-t
                path = output/destination
                path.parent.mkdir(parents=True, exist_ok=True)
                t = time.perf_counter()
                with path.open('xb') as stream:
                    stream.write(encoded)
                if sha(path) != digest(encoded):
                    raise ValueError(f'Written output SHA differs: {destination}')
                timing['write'] += time.perf_counter()-t
                record = dict(frame_id=row['frame_id'], source=source, output=destination,
                    source_sha256=digest(data), output_sha256=digest(encoded),
                    decoded_rgba_sha256=digest(reference), size=list(size), original_mode=mode,
                    original_bytes=len(data), output_bytes=len(encoded), encode_seconds=encode_seconds,
                    exact_decoded_rgba=True, encoded_byte_identity_preserved=encoded==data)
                rows.append(record)
                with (output/'progress.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write(json.dumps(record)+'\n')
        for path, expected in inputs.items():
            if sha(path) != expected:
                raise ValueError(f'Original input changed during build: {path}')
        mapping = dict(schema='compact-rgb-view-v1', images=rows,
            rgb_files=[r['output'] for r in rows], frame_ids=[r['frame_id'] for r in rows],
            authority='Opt-in alternate encoding only; not an existing frozen predictor manifest or full dataset.')
        write_json(output/'mapping.json', mapping)
        result = dict(status='PASS', frames=len(rows), original_bytes=sum(r['original_bytes'] for r in rows),
            output_bytes=sum(r['output_bytes'] for r in rows), exact_pixels_all=True,
            source_mutations=0, timings_seconds=timing, seconds=time.perf_counter()-start)
        write_json(output/'result.json', result)
        write_json(output/'receipt.json', dict(status='PASS', inputs=inputs, code_sha256=sha(__file__),
            reader_sha256=sha(Path(__file__).with_name('data_lightweight.py')), encoder=ENCODER,
            outputs={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()},
            resource_release='All image/archive handles closed; no subprocess or background cache created.'))
        return result
    except Exception as error:
        write_json(output/'failure.json', dict(status='FAIL', error=repr(error), completed_frames=len(rows),
            manifest_sha256=manifest_sha, code_sha256=sha(__file__), source_mutations=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output)))
