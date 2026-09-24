import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cnh_street_capture_queue import CaptureQueue

HASH='a'*64


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'queue.json'
        self.q=CaptureQueue(self.path)

    def add(self,task='one',category='street',layouts=None):
        self.q.add(task,configuration_sha256=HASH,category=category,layouts=layouts or ['L0','L1'])

    def claim(self,machine='5060',output='F:/capture/one',**kwargs):
        return self.q.claim(machine_id=machine,spec_path='F:/spec.json',output_path=output,
                            configuration_sha256=kwargs.get('configuration_sha256',HASH))

    def finish_layouts(self,claim):
        for layout in claim['remaining_layouts']:
            self.q.checkpoint(claim['task_id'],claim['token'],layout_id=layout,outcome='completed',evidence='receipt/'+layout)
        self.q.finish(claim['task_id'],claim['token'],outcome='completed',evidence='capture/receipt')

    def test_restart_keeps_claim_and_checkpoints(self):
        self.add();claim=self.claim()
        self.q.checkpoint('one',claim['token'],layout_id='L0',outcome='completed',evidence='L0.json')
        restarted=CaptureQueue(self.path)
        with self.assertRaises(ValueError):
            restarted.claim(machine_id='5060',spec_path='spec',output_path='different',configuration_sha256=HASH)
        self.assertEqual(restarted.snapshot()['tasks']['one']['attempts'][0]['token'],claim['token'])
        self.assertEqual(restarted.snapshot()['tasks']['one']['checkpoints']['L0']['evidence'],'L0.json')

    def test_quality_quarantine_never_retries(self):
        self.add();claim=self.claim()
        self.q.checkpoint('one',claim['token'],layout_id='L0',outcome='quarantined',evidence='quality-fail.json')
        with self.assertRaises(ValueError):
            self.q.checkpoint('one',claim['token'],layout_id='L0',outcome='completed',evidence='tuned.json')
        self.q.checkpoint('one',claim['token'],layout_id='L1',outcome='completed',evidence='L1.json')
        self.q.finish('one',claim['token'],outcome='completed',evidence='done.json')
        self.assertEqual(self.q.snapshot()['tasks']['one']['status'],'quarantined')
        with self.assertRaises(ValueError): self.q.retry_mechanical('one',evidence='no')

    def test_one_retry_new_path_remaining_layouts_only(self):
        self.add();first=self.claim()
        self.q.checkpoint('one',first['token'],layout_id='L0',outcome='completed',evidence='L0.json')
        self.q.finish('one',first['token'],outcome='mechanical_failed',evidence='crash.log')
        self.q.retry_mechanical('one',evidence='repair.json')
        with self.assertRaises(ValueError): self.claim(output='f:/capture/temp/../one')
        second=self.claim(machine='3060',output='E:/new-attempt')
        self.assertEqual(second['remaining_layouts'],['L1'])
        self.assertEqual(second['number'],2)
        with self.assertRaises(ValueError): self.q.finish('one',first['token'],outcome='mechanical_failed',evidence='stale')
        self.q.finish('one',second['token'],outcome='mechanical_failed',evidence='second-crash.log')
        with self.assertRaises(ValueError): self.q.retry_mechanical('one',evidence='third')
        self.assertEqual(len(self.q.snapshot()['tasks']['one']['attempts']),2)

    def test_category_rotation_and_no_machine_source_binding(self):
        self.add('a','city');self.add('b','city');self.add('c','street')
        first=self.claim();self.finish_layouts(first)
        second=self.claim(output='F:/capture/two')
        self.assertEqual(second['task_id'],'c')
        third=self.claim(machine='3060',output='E:/capture/three')
        self.assertEqual(third['task_id'],'b')

    def test_atomic_write_failure_preserves_old_file_and_releases_lock(self):
        self.add();before=self.path.read_bytes()
        with patch('cnh_street_capture_queue.os.replace',side_effect=OSError('injected')):
            with self.assertRaises(OSError): self.add('two')
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(list(self.path.parent.glob('*.partial')),[])
        self.assertFalse(self.path.with_name('queue.json.lock').exists())
        self.add('two')

    def test_lock_prevents_duplicate_claim_no_stale_stealing(self):
        self.add();lock=self.path.with_name('queue.json.lock');lock.write_text('{}')
        with self.assertRaises(RuntimeError): self.claim()
        self.assertEqual(self.q.snapshot()['tasks']['one']['status'],'pending')
        lock.unlink();self.claim()
        with self.assertRaises(ValueError): self.claim(output='other')

    def test_hash_mismatch_and_partial_finish_rejected(self):
        self.add()
        with self.assertRaises(ValueError): self.claim(configuration_sha256='b'*64)
        claim=self.claim()
        with self.assertRaises(ValueError): self.q.finish('one',claim['token'],outcome='completed',evidence='exit0')

    def test_existing_job_registration_honest_origin(self):
        self.add()
        observed=self.q.register_running('one',machine_id='3060',spec_path='remote/spec',output_path='remote/capture',
            configuration_sha256=HASH,evidence='existing-launch-receipt.json')
        self.assertEqual(observed['origin'],'OBSERVED_EXISTING_JOB')
        self.assertEqual(self.q.snapshot()['tasks']['one']['attempts'][0]['registration_evidence'],'existing-launch-receipt.json')


if __name__=='__main__': unittest.main()
