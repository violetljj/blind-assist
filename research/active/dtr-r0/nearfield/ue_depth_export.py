"""Compatible depth export for opt-in UE acquisition, including same-buffer probes.

EXR mode never exposes the pixel array to Python. Probe mode deliberately
does extra work and must not be reported as production acquisition throughput.
"""
import array
import math
from pathlib import Path
import struct
import time

MODES = ('legacy', 'single_access', 'exr', 'exr_probe')


def npy_header(width, height):
    header = str({'descr': '<f4', 'fortran_order': False, 'shape': (height, width)})
    header += ' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + '\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header.encode('ascii')


def export_depth(u, world, target, filename, mode, index=0):
    if mode not in MODES:
        raise ValueError('Unknown depth export mode: ' + mode)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    width, height = target.size_x, target.size_y
    if mode == 'exr':
        started = time.perf_counter()
        # Export uses native RGBA32F/linear/RCM_MinMax and lossless FLOAT EXR.
        # Publish only after the synchronous native writer has closed the file.
        partial = path.with_suffix('.partial.exr')
        ready = path.with_suffix('.exr')
        if partial.exists() or ready.exists():
            raise FileExistsError(ready)
        u.RenderingLibrary.export_render_target(world, target, str(path.parent), partial.name)
        if not partial.is_file() or partial.stat().st_size < 4:
            raise RuntimeError('Native EXR export failed')
        with partial.open('rb') as stream:
            if stream.read(4) != b'\x76\x2f\x31\x01':
                raise RuntimeError('Native export is not OpenEXR')
        partial.replace(ready)
        return dict(mode=mode, total_s=time.perf_counter()-started)
    if mode == 'exr_probe':
        rows = {}
        # Alternate reference ordering; do not publish EXR until references exist.
        modes = ['legacy', 'single_access'] if index % 2 == 0 else ['single_access', 'legacy']
        for candidate in modes:
            reference = path if candidate == 'single_access' else path.with_suffix('.legacy.npy')
            rows[candidate] = export_depth(u, world, target, reference, candidate, index)
        # EXR has a distinct transport filename because the reference .npy exists.
        rows['exr'] = export_depth(u, world, target, path.with_suffix('.transport.npy'), 'exr', index)
        return dict(mode=mode, order=modes + ['exr'], measurements=rows)
    started = time.perf_counter()
    values = u.RenderingLibrary.read_render_target_raw(world, target, normalize=False)
    read_done = time.perf_counter()
    if len(values) != width * height:
        raise ValueError('Unexpected depth pixel count')
    if mode == 'legacy':
        packed = array.array('f', (v.r / 100 if math.isfinite(v.r) and 0 < v.r < 10000 else 0. for v in values))
    else:
        packed = array.array('f', (r / 100 if math.isfinite(r) and 0 < r < 10000 else 0. for r in (v.r for v in values)))
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('xb') as stream:
        stream.write(npy_header(width, height))
        packed.tofile(stream)
    partial.replace(path)
    return dict(mode=mode, total_s=time.perf_counter()-started,
                readback_s=read_done-started, marshal_write_s=time.perf_counter()-read_done)
