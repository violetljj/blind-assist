"""Bounded host-side native EXR -> compatible depth NPY transport.

One in-flight conversion, plus the producer's fixed queue bound; never enqueue
an entire dataset. Production removes intermediate EXRs only after NPY commit;
the small parity probe retains EXRs and both reference files as evidence.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
from threading import Event

import numpy as np
import OpenEXR
from ue_depth_export import npy_header


def convert(source, output, probe=False):
    started = time.perf_counter()
    with OpenEXR.File(str(source), separate_channels=True) as image:
        red = image.channels()['R'].pixels
        if red.dtype != np.float32 or red.ndim != 2 or red.shape != (360, 640):
            raise ValueError(f'Expected 640x360 FLOAT R channel, got {red.dtype} {red.shape}')
        # Python reference performs float64 arithmetic then rounds to float32.
        valid = np.isfinite(red) & (red > 0) & (red < 10000)
        meters = np.zeros(red.shape, dtype='<f4')
        meters[valid] = red[valid].astype(np.float64) / 100.0
    payload = npy_header(red.shape[1], red.shape[0]) + meters.tobytes(order='C')
    if probe:
        reference = output.read_bytes()
        legacy = output.with_suffix('.legacy.npy').read_bytes()
        if payload != reference or payload != legacy:
            raise ValueError('EXR conversion differs from same-target Python references')
    else:
        if output.exists():
            raise FileExistsError(output)
        partial = output.with_suffix('.npy.partial')
        with partial.open('xb') as stream:
            stream.write(payload)
        partial.replace(output)
    return dict(index=int(output.stem), conversion_s=time.perf_counter()-started,
                depth_sha256=hashlib.sha256(payload).hexdigest(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_bytes=source.stat().st_size, depth_bytes=len(payload),
                bytes_equal=True if probe else None)


class Transport:
    def __init__(self, out, frames, probe=False):
        self.out, self.frames, self.probe = out, frames, probe
        self.index = 0
        self.rows = []
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ue-depth-io')
        self.started = time.perf_counter()
        self.source_complete, self.stop = Event(), Event()
        self.future = self.pool.submit(self.consume)

    def consume(self):
        while self.index < self.frames:
            if self.stop.is_set():
                raise RuntimeError('Transport cancelled')
            stem = f'{self.index:04d}'
            folder = self.out/'evaluator/native'
            source = folder/(stem + ('.transport.exr' if self.probe else '.exr'))
            if not source.is_file():
                if self.source_complete.is_set():
                    raise RuntimeError(f'Missing complete EXR at frame {self.index}')
                self.stop.wait(.005)
                continue
            self.rows.append(convert(source, folder/(stem+'.npy'), self.probe))
            if not self.probe:
                # Only this producer's intermediate, after atomic NPY commit.
                source.unlink()
            self.index += 1

    def pump(self):
        if self.future.done():
            self.future.result()

    def finish(self):
        self.source_complete.set()
        self.future.result()
        return dict(status='PASS', frames=self.frames, rows=self.rows,
                    wall_s=time.perf_counter()-self.started, workers=1,
                    backend='CPU_NATIVE_OPENEXR_NUMPY', scope='Transport conversion, not model inference')

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
