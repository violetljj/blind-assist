"""Independent decoder and wrapper lifecycle tests; never launches CARLA."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import capture_dtr_carla_c2_fast_png as wrapper


class FastPngCaptureTest(unittest.TestCase):
    def image(self):
        raw=np.random.default_rng(907).integers(0,256,(18,32,4),dtype=np.uint8)
        return SimpleNamespace(raw_data=raw.tobytes(),width=32,height=18,frame=42,timestamp=1.25),raw

    def test_metadata_and_independent_rgba_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            journal=io.StringIO()
            save=wrapper.FastPngWriter(root,journal)
            sensor,raw=self.image()
            path=root/'episodes/ep_01/payload/000000.png'
            metadata=save(sensor,path)
            self.assertEqual(set(metadata),{'path','bytes','sha256','world_frame','sensor_timestamp_s','width','height'})
            self.assertEqual(metadata,{'path':str(path),'bytes':path.stat().st_size,'sha256':wrapper.sha(path),
                'world_frame':42,'sensor_timestamp_s':1.25,'width':32,'height':18})
            decoded=np.array(Image.open(path).convert('RGBA'))
            np.testing.assert_array_equal(decoded,raw[:,:,[2,1,0,3]])
            receipt=json.loads(journal.getvalue())
            self.assertEqual(receipt['raw_rgba_sha256'],hashlib.sha256(decoded.tobytes()).hexdigest().upper())
            with self.assertRaises(FileExistsError):
                save(sensor,path)
            with self.assertRaisesRegex(ValueError,'outside_owned_shard'):
                save(sensor,root.parent/'escape.png')

    def test_failure_restores_base_and_preserves_separate_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            base_path=root/'unchanged.py'
            base_path.write_text('# original bytes\n')
            protocol=root/'protocol.json'
            protocol.write_text('{}')
            original=object()
            base=SimpleNamespace(__file__=str(base_path),save_image=original,
                parse_args=lambda:SimpleNamespace(output_root=root,sensor='depth',protocol=protocol))
            def fail():
                self.assertIsInstance(base.save_image,wrapper.FastPngWriter)
                raise RuntimeError('synthetic failure')
            base.main=fail
            original_hash=wrapper.sha(base_path)
            with patch.object(wrapper.importlib,'import_module',return_value=base):
                with self.assertRaisesRegex(RuntimeError,'synthetic failure'):
                    wrapper.main()
            self.assertIs(base.save_image,original)
            self.assertEqual(wrapper.sha(base_path),original_hash)
            receipt=root/'encoder-receipts/depth'
            outcome=json.loads((receipt/'outcome.json').read_text())
            self.assertEqual(outcome['status'],'FAILED')
            self.assertEqual(outcome['payload_count'],0)
            self.assertIsNone(outcome['base_result_sha256'])
            identity=json.loads((receipt/'implementation.json').read_text())
            self.assertEqual(identity['files']['base_capture']['sha256'],original_hash)
            self.assertFalse((root/'shards/depth/result.json').exists())


if __name__=='__main__':
    unittest.main()
