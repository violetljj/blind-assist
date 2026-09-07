from types import SimpleNamespace
import unittest
from ue_pair_export import PairExporter, create_pair_exporter


class PairTests(unittest.TestCase):
    def exporter(self, **changes):
        fields = dict(pending=0,peak_pending=0,submitted=0,completed=0,failed=0,
                      last_error='',readback_seconds=0,encode_seconds=0,write_seconds=0,submit_seconds=0,gpu_ready_seconds=0)
        fields.update(changes)
        profile = SimpleNamespace(**fields)
        api = SimpleNamespace(poll_capture_pairs=lambda:profile,drain_capture_pairs=lambda:profile)
        return PairExporter(SimpleNamespace(BlindAssistCaptureLibrary=api))

    def test_queue_bound(self):
        self.assertFalse(self.exporter(pending=4).ready())
        self.assertTrue(self.exporter(pending=3).ready())

    def test_auto_supports_old_plugin(self):
        self.assertIsNone(create_pair_exporter(SimpleNamespace(), 'auto'))
        u=SimpleNamespace(BlindAssistCaptureLibrary=SimpleNamespace(submit_capture_pair=lambda:True))
        self.assertIsInstance(create_pair_exporter(u,'auto'),PairExporter)

    def test_failure_is_not_backpressure(self):
        with self.assertRaisesRegex(RuntimeError,'copy failed'):
            self.exporter(failed=1,last_error='copy failed').ready()

    def test_drain_requires_all_outputs(self):
        ex = self.exporter(completed=0)
        ex.rows = [{}]
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):
            ex.finish()
        ex = self.exporter(completed=1)
        ex.rows = [{}]
        self.assertEqual(ex.finish()['profile']['completed'],1)


if __name__ == '__main__':
    unittest.main()
