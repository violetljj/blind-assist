"""Bounded native RGB encoding; engine readback remains synchronous.

Probe writes an additional reference from the same target, without another render.
The host verifies decoded pixels after all background writes have drained.
"""
from pathlib import Path
import time


class RgbExporter:
    def __init__(self, u, mode='legacy', limit=4):
        available = hasattr(getattr(u, 'BlindAssistCaptureLibrary', None), 'export_rgb_png')
        if mode == 'auto':
            mode = 'native_async' if available else 'legacy'
        if mode not in ('legacy', 'native_async', 'native_probe'):
            raise ValueError('Unknown RGB export mode: ' + mode)
        if mode != 'legacy' and not available:
            raise RuntimeError('Rebuild the capture plugin to enable native RGB export')
        self.u, self.mode, self.limit = u, mode, limit
        self.rows = []

    def profile(self, drain=False):
        if self.mode == 'legacy':
            return dict(pending=0, completed=len(self.rows), failed=0)
        api = self.u.BlindAssistCaptureLibrary
        p = api.drain_rgb_writes() if drain else api.poll_rgb_writes()
        return {key: getattr(p, key) for key in
                ('pending', 'peak_pending', 'submitted', 'completed', 'failed', 'last_error', 'readback_seconds', 'encode_seconds', 'write_seconds')}

    def ready(self):
        p = self.profile()
        if p['failed']:
            raise RuntimeError('Native RGB write failed: ' + p['last_error'])
        return p['pending'] < self.limit

    def export(self, world, target, path, index):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        start = time.perf_counter()
        if self.mode == 'legacy':
            self.u.RenderingLibrary.export_render_target(world, target, str(path.parent), path.name)
            if not path.is_file():
                raise RuntimeError('RGB export did not publish ' + str(path))
        else:
            if self.mode == 'native_probe':
                reference = path.with_name(path.stem + '.reference.png')
                if reference.exists():
                    raise FileExistsError(reference)
                self.u.RenderingLibrary.export_render_target(world, target, str(path.parent), reference.name)
            if not self.u.BlindAssistCaptureLibrary.export_rgb_png(world, target, str(path), self.limit):
                raise RuntimeError('Native RGB submission failed: ' + str(path))
        self.rows.append(dict(sample_index=index, submit_s=time.perf_counter()-start))

    def finish(self):
        p = self.profile(drain=True)
        if p['failed'] or p['pending'] or p['completed'] != len(self.rows):
            raise RuntimeError('Incomplete RGB exports: ' + str(p))
        return dict(mode=self.mode, queue_limit=self.limit, profile=p, rows=self.rows)
