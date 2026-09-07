"""Tick-polled readiness after an explicit synchronous streaming demand update.

The plugin must expose prepare_capture_readiness(component) and
poll_capture_readiness(world). Call poll only on a subsequent editor callback. Asset registry alone does
not establish compilation or texture readiness. This gate preserves capture
settling and only covers the currently requested texture/view workload.
"""
import math
import time


class CaptureReadiness:
    COUNTERS = ('asset_compilation_remaining', 'shader_jobs_remaining',
                'pending_render_assets')

    def __init__(self, u, timeout=120, clock=time.monotonic):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('Readiness timeout must be finite and positive')
        self.u, self.timeout, self.clock = u, timeout, clock
        self.state = 'NOT_STARTED'
        self.last = None
        self.error = None
        self.started = None
        self.poll_count = 0
        self.ended = None

    def _fail(self, message, error_type=RuntimeError):
        self.state, self.error = 'FAIL', message
        self.ended = self.clock()
        raise error_type(message)

    def _snapshot(self):
        try:
            native = self.u.BlindAssistCaptureLibrary.poll_capture_readiness(self.world)
            get = (lambda key: native[key]) if isinstance(native, dict) else (lambda key: getattr(native, key))
            if get('ready_supported') is not True:
                raise ValueError('Native readiness observation unsupported')
            completed = get('streaming_update_completed')
            if type(completed) is not bool:
                raise ValueError('Invalid streaming update state')
            row = {key: get(key) for key in self.COUNTERS}
            if any(type(value) is not int or value < 0 for value in row.values()):
                raise ValueError('Invalid native readiness counters')
            loading = self.u.AssetRegistryHelpers.get_asset_registry().is_loading_assets()
            if type(loading) is not bool:
                raise ValueError('Invalid asset registry readiness')
            row['asset_registry_loading'] = loading
            row['streaming_update_completed'] = completed
            return row
        except Exception as exc:
            self._fail('Readiness observation failed: ' + str(exc))

    def begin(self, world, capture_component):
        if self.state == 'WAITING':
            raise RuntimeError('Readiness already waiting')
        self.world, self.started = world, self.clock()
        self.capture_component = capture_component
        self.state, self.error, self.poll_count = 'WAITING', None, 0
        self.ended = None
        self.prepared = self._prepare()
        self.last = self._snapshot()

    def _prepare(self):
        try:
            result = self.u.BlindAssistCaptureLibrary.prepare_capture_readiness(self.capture_component)
            if type(result) is not bool:
                raise ValueError('Invalid native preparation result')
            return result
        except Exception as exc:
            self._fail('Readiness preparation failed: ' + str(exc))

    def poll(self):
        if self.state == 'READY':
            return True
        if self.state != 'WAITING':
            raise RuntimeError('Readiness is not waiting: ' + self.state)
        if self.clock() - self.started >= self.timeout:
            self._fail('Capture readiness timed out', TimeoutError)
        row = self._snapshot()
        if row['asset_registry_loading']:
            self.prepared = False
        if not row['asset_registry_loading'] and (not self.prepared or not row['streaming_update_completed']
                or row['asset_compilation_remaining'] or row['shader_jobs_remaining']):
            # False means compilation is still busy. Retry at most once in this
            # editor callback; native code invalidates its update on new work.
            self.prepared = self._prepare()
            row = self._snapshot()
        self.last = row
        self.poll_count += 1
        idle = (self.prepared and row['streaming_update_completed'] and not row['asset_registry_loading']
                and all(row[key] == 0 for key in self.COUNTERS))
        if idle:
            self.state = 'READY'
            self.ended = self.clock()
        return self.state == 'READY'

    def receipt(self):
        return dict(status=self.state, elapsed_s=None if self.started is None else (self.ended if self.ended is not None else self.clock())-self.started,
                    timeout_s=self.timeout, poll_count=self.poll_count,
                    observation=None if self.last is None else dict(self.last), error=self.error,
                    scope='CURRENT_REQUESTED_ASSETS_NOT_ALL_FUTURE_CAMERA_VIEWS',
                    settling_unchanged=True)
