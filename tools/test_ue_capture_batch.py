import argparse
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import ue_capture_batch as batch


class BatchTests(unittest.TestCase):
    def setUp(self):
        parent = batch.capture.REPO/'artifacts.local/tmp'
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.out = self.repo/'artifacts.local/batch'
        self.out.mkdir(parents=True)
        source = self.repo/'research/active/dtr-r0/nearfield'
        source.mkdir(parents=True)
        (source/'ue_capture_session.py').write_text('# bootstrap')
        spec = self.repo/'artifacts.local/spec.json'
        spec.write_text('{}')
        self.identity = dict(fingerprint='f', jobs=[dict(id=x, capture='grounding', spec=str(spec),
            spec_sha256=batch.capture.file_hash(spec)) for x in ('a','b')],
            source_hashes={'ue_capture_session.py': batch.capture.file_hash(source/'ue_capture_session.py')})
        self.args = argparse.Namespace(output=self.out, resume=True, engine=None, plugin=None,
            timeout=30, pair_export='native_async', settling_policy='full', startup_policy='ready', standard_init=False)
        self.items = []
        self.repo_patch = patch.object(batch.capture, 'REPO', self.repo)
        self.repo_patch.start()
        self.addCleanup(self.repo_patch.stop)

    def prepare(self, options):
        out = options.output
        (out/'source').mkdir(parents=True)
        (out/'source/spec.json').write_bytes(options.spec.read_bytes())
        item = dict(out=out, script=out/'source/capture.py', env={}, command=['editor','-ExecCmds=py old'])
        self.items.append(item)
        return item

    def publish(self, out):
        (out/'payload').write_text('verified')
        batch.atomic_write(out/'payload-hashes.json', {'payload':batch.capture.file_hash(out/'payload')})
        batch.atomic_write(out/'completion.json',dict(status='PASS',payload_hashes_sha256=batch.capture.file_hash(out/'payload-hashes.json')))
        return dict(frame_count=1)

    def cached(self, name):
        out=self.out/'jobs'/name/'old'
        out.mkdir(parents=True)
        self.publish(out)
        return dict(status='PASS',output=str(out))

    def run_fake(self, launch, validate=None):
        with patch.object(batch.capture,'capture',side_effect=self.prepare), \
             patch.object(batch.capture,'validate_capture',side_effect=validate or self.publish), \
             patch.object(batch.capture,'run_owned',side_effect=launch), \
             patch.object(batch,'request_identity',return_value=self.identity):
            batch.run_batch(self.args,self.identity,self.out)

    def launch_success(self, command, env, session, timeout, pump):
        ledger=batch.read(self.out/'completed.json')
        self.assertTrue(all(ledger[item['id']]['status']=='PENDING' for item in self.items))
        for item in self.items:
            batch.atomic_write(item['out']/'receipt.json',dict(status='PASS'))
            pump()
        batch.atomic_write(session/'receipt.json',dict(status='PASS'))

    def test_all_cached_never_launches(self):
        batch.atomic_write(self.out/'completed.json',{x:self.cached(x) for x in ('a','b')})
        self.run_fake(lambda *a: self.fail('launched cached batch'))
        self.assertEqual(self.items,[])

    def test_corruption_reacquires_only_damaged_block(self):
        records={x:self.cached(x) for x in ('a','b')}
        (Path(records['b']['output'])/'payload').write_text('corrupt')
        batch.atomic_write(self.out/'completed.json',records)
        self.run_fake(self.launch_success)
        self.assertEqual([x['id'] for x in self.items],['b'])
        ledger=batch.read(self.out/'completed.json')
        self.assertEqual(ledger['a'],records['a'])
        self.assertTrue(batch.reusable(ledger['b']))

    def test_input_drift_rejected_before_launch(self):
        batch.atomic_write(self.out/'identity.json',dict(fingerprint='old'))
        with patch.object(batch,'request_identity',return_value=self.identity), patch.object(batch,'run_batch') as run:
            with self.assertRaisesRegex(ValueError,'changed'):
                batch.batch(self.args)
            run.assert_not_called()
        self.assertFalse((self.out/'owner.lock').exists())

    def test_failed_block_record_and_raw_receipt_are_retained(self):
        def validate(out):
            if out.parent.name=='b': raise ValueError('bad payload')
            return self.publish(out)
        with self.assertRaises(RuntimeError):
            self.run_fake(self.launch_success,validate)
        ledger=batch.read(self.out/'completed.json')
        self.assertEqual(ledger['a']['status'],'PASS')
        self.assertEqual(ledger['b']['status'],'FAIL')
        self.assertTrue((Path(ledger['b']['output'])/'receipt.json').exists())
        self.assertEqual(len(list((self.out/'sessions').glob('*/b-validation.json'))),1)

    def test_validation_overlaps_next_capture(self):
        started,release=threading.Event(),threading.Event()
        def validate(out):
            if out.parent.name=='a':
                started.set()
                if not release.wait(3): raise RuntimeError('No pipeline overlap')
            return self.publish(out)
        def launch(command,env,session,timeout,pump):
            try:
                batch.atomic_write(self.items[0]['out']/'receipt.json',dict(status='PASS'))
                pump()
                self.assertTrue(started.wait(3))
                batch.atomic_write(self.items[1]['out']/'receipt.json',dict(status='PASS'))
                pump()
            finally:
                release.set()
            batch.atomic_write(session/'receipt.json',dict(status='PASS'))
        self.run_fake(launch,validate)


if __name__=='__main__': unittest.main()
