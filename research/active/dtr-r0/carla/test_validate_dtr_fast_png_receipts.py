"""Three-shard composite layout and independent pixel checks; synthetic only."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import capture_dtr_carla_c2_fast_png as capture
import validate_dtr_fast_png_receipts as validator


class CompositeReceiptTests(unittest.TestCase):
    def fixture(self,root):
        identity=root/'synthetic-implementation.py'
        identity.write_text('# synthetic identity only\n')
        identity_hash=validator.sha(identity)
        for group,sensor in validator.SHARDS:
            group_root=root/'raw'/group
            shard=group_root/'shards'/sensor
            shard.mkdir(parents=True)
            receipt=group_root/'encoder-receipts'/sensor
            receipt.mkdir(parents=True)
            with (receipt/'raw-pixels.jsonl').open('x',encoding='utf-8') as journal:
                writer=capture.FastPngWriter(shard,journal)
                image=SimpleNamespace(raw_data=bytes([1,2,3,4,5,6,7,8]),
                                      width=2,height=1,frame=10,timestamp=.1)
                writer(image,shard/'episodes/ep_01/payload/000000.png')
            implementation={'schema':'dtr-c2-fast-png-implementation-v1','sensor':sensor,
                'authority':'SYNTHETIC_DEVELOPMENT_COMPOSITE_NOT_FRESH_CONFIRMATION',
                'files':{name:{'path':str(identity),'sha256':identity_hash}
                         for name in ('base_capture','encoder','wrapper','protocol')}}
            capture.write_json(receipt/'implementation.json',implementation)
            capture.write_json(shard/'result.json',{'sensor':sensor,
                'status':'DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE','payload_count':1,
                'checks':{'complete':True},'capture_script_sha256':identity_hash,
                'protocol_sha256':identity_hash})
            capture.write_json(receipt/'outcome.json',{'schema':'dtr-c2-fast-png-outcome-v1',
                'sensor':sensor,'status':'COMPLETE','exit_code':0,'payload_count':1,
                'implementation_receipt_sha256':validator.sha(receipt/'implementation.json'),
                'raw_pixel_journal_sha256':validator.sha(receipt/'raw-pixels.jsonl'),
                'base_result_sha256':validator.sha(shard/'result.json')})

    def temporary(self):
        parent=Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        parent.mkdir(parents=True,exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='fast-png-composite-test-',dir=parent)

    def test_full_composite_layout_and_exclusive_report(self):
        with self.temporary() as temporary:
            root=Path(temporary).resolve()
            self.fixture(root)
            result=validator.validate(root)
            self.assertEqual(result['shard_count'],3)
            self.assertEqual(result['validated_payload_count'],3)
            self.assertEqual([(row['group'],row['sensor']) for row in result['shards']],list(validator.SHARDS))
            self.assertTrue((root/'fast-png-validation.json').is_file())
            self.assertFalse((root/'raw/fast-png-validation.json').exists())
            self.assertFalse((root/'FINAL_A').exists())
            with self.assertRaisesRegex(ValueError,'already_exists'):
                validator.validate(root)

    def test_pixel_corruption_rejected_before_success_receipt(self):
        with self.temporary() as temporary:
            root=Path(temporary).resolve()
            self.fixture(root)
            group_root=root/'raw/FINAL_A'
            receipt=group_root/'encoder-receipts/depth'
            payload=group_root/'shards/depth/episodes/ep_01/payload/000000.png'
            # Substitute another valid PNG, and update encoded-file/journal hashes.
            # Preserve the original raw-buffer digest: the independent decoder
            # must detect pixel change even when file-integrity checks pass.
            from PIL import Image
            Image.new('RGBA',(2,1),(200,100,50,255)).save(payload)
            journal=receipt/'raw-pixels.jsonl'
            row=json.loads(journal.read_text())
            row['png_sha256']=validator.sha(payload)
            row['png_bytes']=payload.stat().st_size
            journal.write_text(json.dumps(row)+'\n')
            outcome_path=receipt/'outcome.json'
            outcome=json.loads(outcome_path.read_text())
            outcome['raw_pixel_journal_sha256']=validator.sha(journal)
            outcome_path.write_text(json.dumps(outcome))
            with self.assertRaisesRegex(ValueError,'decoded_rgba_hash'):
                validator.validate(root)
            self.assertFalse((root/'fast-png-validation.json').exists())


if __name__=='__main__':
    unittest.main()
