"""Transport precision, publication, failure and bounded-worker checks."""
import array
from pathlib import Path
import tempfile
import unittest

import numpy as np
import OpenEXR
from ue_depth_export import npy_header
from ue_exr_transport import convert, Transport


class ExportTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='ue-depth-test-', dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, red, path=None):
        path = path or self.root/'0000.exr'
        OpenEXR.File({}, {'R': red}).write(str(path))
        return path

    def test_extremes_match_legacy_bytes(self):
        values = np.array([np.nan, np.inf, -np.inf, -0., 0., -1., 1e-40,
                           1., 99.999, 100., 9999., np.nextafter(np.float32(10000), np.float32(0)),
                           10000., 10001.], dtype=np.float32)
        red = np.resize(values, (360, 640)).copy()
        path = self.root/'0000.npy'
        convert(self.source(red), path)
        legacy = array.array('f', (float(r)/100 if np.isfinite(r) and 0<r<10000 else 0. for r in red.flat))
        self.assertEqual(path.read_bytes(), npy_header(640,360)+legacy.tobytes())
        self.assertEqual(np.load(path).shape, (360,640))
        self.assertFalse(path.with_suffix('.npy.partial').exists())

    def test_no_overwrite(self):
        path=self.root/'0000.npy';path.write_bytes(b'preserved')
        with self.assertRaises(FileExistsError):
            convert(self.source(np.ones((360,640),dtype=np.float32)),path)
        self.assertEqual(path.read_bytes(),b'preserved')

    def test_half_depth_rejected(self):
        source=self.source(np.ones((360,640),dtype=np.float16))
        with self.assertRaises(ValueError):convert(source,self.root/'0000.npy')

    def test_corrupt_source_rejected(self):
        source=self.root/'0000.exr';source.write_bytes(b'broken')
        with self.assertRaises(Exception):convert(source,self.root/'0000.npy')
        self.assertFalse((self.root/'0000.npy').exists())

    def test_missing_frame_fails_without_hanging(self):
        worker=Transport(self.root,1)
        try:
            with self.assertRaises(RuntimeError):worker.finish()
        finally:worker.close()

    def test_stream_completes_and_releases_intermediates(self):
        folder=self.root/'evaluator/native';folder.mkdir(parents=True)
        worker=Transport(self.root,2)
        try:
            for i in range(2):
                partial=folder/f'{i:04d}.partial.exr'
                self.source(np.full((360,640),100*(i+1),dtype=np.float32),partial)
                partial.replace(folder/f'{i:04d}.exr')
            result=worker.finish()
            self.assertEqual(result['frames'],2)
            self.assertEqual(len(result['rows']),2)
            self.assertEqual(float(np.load(folder/'0001.npy')[0,0]),2.)
            self.assertFalse(list(folder.glob('*.exr')))
        finally:worker.close()

    def test_probe_detects_mismatch(self):
        path=self.root/'0000.npy';path.write_bytes(b'wrong')
        path.with_suffix('.legacy.npy').write_bytes(b'wrong')
        with self.assertRaises(ValueError):
            convert(self.source(np.ones((360,640),dtype=np.float32)),path,probe=True)
        self.assertEqual(path.read_bytes(),b'wrong')


if __name__ == '__main__':
    unittest.main()
