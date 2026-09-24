import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ue_attribute_export import AttributePairExporter, write_rgb_reference, validate_npy_payload


class Api:
    def __init__(self):
        self.pending = 0; self.submitted = 0; self.failed = 0
    def reset_capture_pairs(self): return True
    def submit_capture_npy_batch(self, *args): return True
    def poll_capture_pairs(self):
        return SimpleNamespace(pending=self.pending, peak_pending=self.pending, submitted=self.submitted,
            completed=self.submitted, failed=self.failed, last_error='fixture', readback_seconds=0.,
            encode_seconds=0., write_seconds=0., submit_seconds=0., gpu_ready_seconds=0.)
    def drain_capture_pairs(self): self.pending = 0; return self.poll_capture_pairs()


class AttributeExporterTests(unittest.TestCase):
    def test_reserves_three_jobs_in_shared_queue_four(self):
        api = Api(); export = AttributePairExporter(SimpleNamespace(BlindAssistCaptureLibrary=api))
        api.pending = 1; self.assertTrue(export.ready())
        api.pending = 2; self.assertFalse(export.ready())
        api.failed = 1
        with self.assertRaises(RuntimeError): export.ready()

    def test_finish_counts_pair_and_attribute_jobs(self):
        api = Api(); export = AttributePairExporter(SimpleNamespace(BlindAssistCaptureLibrary=api))
        export.rows = [{}, {}]; export.attribute_batches = [dict(files=[])]
        api.submitted = 3
        self.assertEqual(export.finish()['profile']['completed'], 3)
        api.submitted = 2
        with self.assertRaises(RuntimeError): export.finish()

    def test_no_silent_fallback_to_old_plugin(self):
        with self.assertRaises(RuntimeError):
            AttributePairExporter(SimpleNamespace(BlindAssistCaptureLibrary=object()))

    def test_reference_preserves_signed_rgb_and_exact_payload_count(self):
        import struct
        target = SimpleNamespace(size_x=1, size_y=1)
        pixel = SimpleNamespace(r=-.5, g=.25, b=1.)
        api = SimpleNamespace(RenderingLibrary=SimpleNamespace(read_render_target_raw=lambda *a, **k:[pixel]))
        temp_root = Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='attribute-export-test-', dir=temp_root) as tmp:
            path = Path(tmp)/'normal.npy'
            write_rgb_reference(api, None, target, path)
            self.assertEqual(validate_npy_payload(path, (1, 1, 3))['dtype'], '<f4')
            self.assertEqual(struct.unpack('<fff', path.read_bytes()[-12:]), (-.5, .25, 1.))
            with path.open('ab') as stream: stream.write(b'x')
            with self.assertRaises(ValueError): validate_npy_payload(path, (1, 1, 3))


if __name__ == '__main__':
    unittest.main()
