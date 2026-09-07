"""Python session boundaries; native pending-job rejection needs engine coverage."""
import unittest
from types import SimpleNamespace as NS
from ue_pair_export import PairExporter
from ue_rgb_export import RgbExporter


class ExporterResetTests(unittest.TestCase):
    def test_each_new_native_exporter_resets_completed_session(self):
        for cls, reset_name in ((PairExporter, 'reset_capture_pairs'),
                                (RgbExporter, 'reset_rgb_writes')):
            with self.subTest(exporter=cls.__name__):
                calls = []
                api = NS(export_rgb_png=lambda *a: True)
                setattr(api, reset_name, lambda: calls.append('reset') or True)
                u = NS(BlindAssistCaptureLibrary=api)
                first = cls(u, mode='native_async')
                first.rows.append({'old': True})
                second = cls(u, mode='native_async')
                self.assertEqual(calls, ['reset', 'reset'])
                self.assertEqual(second.rows, [])
                self.assertEqual(first.rows, [{'old': True}])

    def test_native_pending_rejection_prevents_new_exporter(self):
        for cls, reset_name in ((PairExporter, 'reset_capture_pairs'),
                                (RgbExporter, 'reset_rgb_writes')):
            with self.subTest(exporter=cls.__name__):
                api = NS(export_rgb_png=lambda *a: True)
                setattr(api, reset_name, lambda: False)
                with self.assertRaisesRegex(RuntimeError, 'still pending'):
                    cls(NS(BlindAssistCaptureLibrary=api), mode='native_async')

    def test_old_plugin_without_reset_retains_one_shot_compatibility(self):
        u = NS(BlindAssistCaptureLibrary=NS(export_rgb_png=lambda *a: True))
        self.assertEqual(PairExporter(u).rows, [])
        self.assertEqual(RgbExporter(u, mode='native_async').rows, [])

    def test_legacy_rgb_does_not_touch_native_reset(self):
        def forbidden():
            raise AssertionError('Legacy exporter called native reset')
        u = NS(BlindAssistCaptureLibrary=NS(reset_rgb_writes=forbidden))
        self.assertEqual(RgbExporter(u, mode='legacy').mode, 'legacy')
        self.assertEqual(RgbExporter(NS(), mode='auto').mode, 'legacy')


if __name__ == '__main__':
    unittest.main()
