import unittest
from types import SimpleNamespace as NS
from ue_capture_readiness import CaptureReadiness


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.row = dict(ready_supported=True, asset_compilation_remaining=0,
                        shader_jobs_remaining=0, pending_render_assets=0,
                        streaming_update_completed=True)
        self.loading = False
        registry = NS(is_loading_assets=lambda: self.loading)
        self.u = NS(BlindAssistCaptureLibrary=NS(poll_capture_readiness=lambda world: self.row,
                        prepare_capture_readiness=lambda component: True),
                    AssetRegistryHelpers=NS(get_asset_registry=lambda: registry))
        self.gate = CaptureReadiness(self.u, timeout=10, clock=lambda: self.now)

    def test_prepare_then_subsequent_poll_can_pass_without_ids(self):
        calls = []
        self.u.BlindAssistCaptureLibrary.prepare_capture_readiness = lambda component: calls.append(component) or True
        component = object()
        self.gate.begin(object(), component)
        self.assertEqual(calls, [component])
        self.assertEqual(self.gate.receipt()['status'], 'WAITING')
        self.assertTrue(self.gate.poll())

    def test_each_actual_pending_counter_blocks(self):
        for field in ('asset_compilation_remaining', 'shader_jobs_remaining', 'pending_render_assets'):
            with self.subTest(field=field):
                self.setUp()
                self.row[field] = 1
                self.gate.begin(object(), object())
                self.assertFalse(self.gate.poll())
                self.row[field] = 0
                self.assertTrue(self.gate.poll())

    def test_streaming_update_not_completed_blocks(self):
        self.row['streaming_update_completed'] = False
        self.gate.begin(object(), object())
        self.assertFalse(self.gate.poll())

    def test_busy_prepare_retries_and_then_passes(self):
        calls = []
        self.u.BlindAssistCaptureLibrary.prepare_capture_readiness = lambda component: calls.append(1) or len(calls) >= 3
        self.gate.begin(object(), object())
        self.assertFalse(self.gate.poll())
        self.assertEqual(len(calls), 2)
        self.assertTrue(self.gate.poll())
        self.assertEqual(len(calls), 3)

    def test_invalidated_update_prepares_again(self):
        calls = []
        def prepare(component):
            calls.append(1)
            self.row['streaming_update_completed'] = True
            return True
        self.u.BlindAssistCaptureLibrary.prepare_capture_readiness = prepare
        self.gate.begin(object(), object())
        self.row['streaming_update_completed'] = False
        self.assertTrue(self.gate.poll())
        self.assertEqual(len(calls), 2)

    def test_invalid_prepare_result_fails(self):
        self.u.BlindAssistCaptureLibrary.prepare_capture_readiness = lambda component: None
        with self.assertRaises(RuntimeError):
            self.gate.begin(object(), object())
        self.assertEqual(self.gate.receipt()['status'], 'FAIL')

    def test_loading_assets_blocks(self):
        self.gate.begin(object(), object())
        self.loading = True
        for i in (11, 12, 13):
            self.assertFalse(self.gate.poll())

    def test_timeout_fails_closed(self):
        self.gate.begin(object(), object())
        self.now = 10
        with self.assertRaises(TimeoutError):
            self.gate.poll()
        self.assertEqual(self.gate.receipt()['status'], 'FAIL')

    def test_missing_helper_and_unsupported_fail_closed(self):
        del self.u.BlindAssistCaptureLibrary.poll_capture_readiness
        with self.assertRaises(RuntimeError):
            self.gate.begin(object(), object())
        self.assertEqual(self.gate.receipt()['status'], 'FAIL')
        self.setUp()
        self.row['ready_supported'] = False
        with self.assertRaises(RuntimeError):
            self.gate.begin(object(), object())

    def test_invalid_counter_fails_closed(self):
        self.row['shader_jobs_remaining'] = -1
        with self.assertRaises(RuntimeError):
            self.gate.begin(object(), object())


if __name__ == '__main__':
    unittest.main()
