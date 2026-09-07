import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from ue_rgb_export import RgbExporter


class RgbTests(unittest.TestCase):
    def exporter(self, **changes):
        values = dict(pending=0, peak_pending=0, submitted=0, completed=0, failed=0, last_error='',
                      readback_seconds=0., encode_seconds=0., write_seconds=0.)
        values.update(changes)
        profile = SimpleNamespace(**values)
        api = SimpleNamespace(poll_rgb_writes=lambda: profile, drain_rgb_writes=lambda: profile,
                              export_rgb_png=lambda *args: True)
        return RgbExporter(SimpleNamespace(BlindAssistCaptureLibrary=api), 'native_async')

    def test_backpressure(self):
        self.assertFalse(self.exporter(pending=4).ready())
        self.assertTrue(self.exporter(pending=3).ready())

    def test_auto_supports_old_plugins(self):
        self.assertEqual(RgbExporter(SimpleNamespace(), 'auto').mode, 'legacy')
        self.assertEqual(RgbExporter(self.exporter().u, 'auto').mode, 'native_async')

    def test_background_failure_propagates(self):
        with self.assertRaisesRegex(RuntimeError, 'disk full'):
            self.exporter(failed=1, last_error='disk full').ready()

    def test_drain_requires_all_submissions(self):
        ex = self.exporter(completed=0)
        ex.rows.append(dict(sample_index=0))
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            ex.finish()
        ex = self.exporter(completed=1)
        ex.rows.append(dict(sample_index=0))
        self.assertEqual(ex.finish()['profile']['completed'], 1)

    def test_reject_existing_before_native_call(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'frame.png'
            path.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                self.exporter().export(None, None, path, 0)
            self.assertEqual(path.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main()
