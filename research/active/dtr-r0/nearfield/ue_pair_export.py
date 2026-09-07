"""Bounded native GPU readback of one RGB/depth observation pair."""
from pathlib import Path
import time


class PairExporter:
    def __init__(self, u, mode='native_async', limit=4):
        self.u, self.mode, self.limit = u, mode, limit
        if mode not in ('native_async', 'native_probe'):
            raise ValueError('Unknown pair mode: ' + mode)
        self.rows = []

    def profile(self, drain=False):
        api = self.u.BlindAssistCaptureLibrary
        p = api.drain_capture_pairs() if drain else api.poll_capture_pairs()
        return {key: getattr(p, key) for key in
                ('pending', 'peak_pending', 'submitted', 'completed', 'failed', 'last_error',
                 'readback_seconds', 'encode_seconds', 'write_seconds', 'submit_seconds', 'gpu_ready_seconds')}

    def ready(self):
        p = self.profile()
        if p['failed']:
            raise RuntimeError('GPU pair export failed: ' + p['last_error'])
        return p['pending'] < self.limit

    def export(self, world, rgb, depth, png_path, npy_path, index):
        png, npy = Path(png_path), Path(npy_path)
        for path in (png, npy):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
        if self.mode == 'native_probe':
            reference = png.with_name(png.stem + '.reference.png')
            depth_reference = npy.with_name(npy.stem + '.reference.npy')
            if reference.exists() or depth_reference.exists():
                raise FileExistsError('Pair probe reference already exists')
            self.u.RenderingLibrary.export_render_target(world, rgb, str(reference.parent), reference.name)
            if not self.u.BlindAssistCaptureLibrary.export_depth_npy(world, depth, str(depth_reference)):
                raise RuntimeError('Reference depth failed')
        started = time.perf_counter()
        if not self.u.BlindAssistCaptureLibrary.submit_capture_pair(rgb, depth, str(png), str(npy), self.limit):
            raise RuntimeError('GPU pair submission failed: ' + str(png))
        row = dict(sample_index=index, mode='gpu_async', total_s=time.perf_counter()-started)
        self.rows.append(row)
        return row

    def finish(self):
        p = self.profile(drain=True)
        if p['pending'] or p['failed'] or p['completed'] != len(self.rows):
            raise RuntimeError('Incomplete GPU pairs: ' + str(p))
        return dict(mode=self.mode, queue_limit=self.limit, profile=p, rows=self.rows)


def create_pair_exporter(u, mode):
    if mode == 'auto':
        mode = 'native_async' if hasattr(getattr(u, 'BlindAssistCaptureLibrary', None), 'submit_capture_pair') else 'off'
    return None if mode == 'off' else PairExporter(u, mode)
