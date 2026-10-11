"""Focused v2 change check: paired 4/4b, one-session resume and redo only.

Earlier complete v1 startup/30s/integrity checks are retained at Git a03b1715.
No COM connection, network request, acquisition, flashing or real annotation.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import real_labeled_capture as guided

REPO = Path(__file__).resolve().parents[4]
PAYLOAD = REPO / 'artifacts.local/work/real-labeled-capture-paired-prep-20261011'
RAW = REPO / 'artifacts.local/hardware-bringup/captures/cnh-h3-1mhz-20260927/raw.bin'

class FakeClock:
    def __init__(self): self.value = 0.
    def sleep(self, seconds): self.value += seconds
    def monotonic(self): return self.value
    def monotonic_ns(self): return int(self.value * 1e9)
    def utcnow(self):
        return (datetime(2026, 10, 11, tzinfo=timezone.utc) + timedelta(seconds=self.value)).isoformat()

class PairedCaptureFocusedTest(unittest.TestCase):
    def test_focused_ten_segments_pair_resume_and_redo_4b(self):
        PAYLOAD.mkdir(parents=True, exist_ok=True)
        workspace = tempfile.TemporaryDirectory(prefix='focused-paired-', dir=PAYLOAD)
        workspace._finalizer.detach()  # preserve captures and interruption evidence
        root = Path(workspace.name)
        source_hash = hashlib.sha256(RAW.read_bytes()).hexdigest()
        plan = guided.segment_plan()
        self.assertEqual([1, 2, 3, 4, '4b', 5, 6, 7, 8, 9], [s['condition_index'] for s in plan])
        self.assertEqual('simple-dorm-v2', guided.PROTOCOL_VERSION)
        def make(interrupt=False):
            clock, devices, console = FakeClock(), [], []
            def factory():
                device = guided.ReplaySerial(RAW, clock)
                if interrupt:
                    original = device.read
                    def read(size):
                        if clock.monotonic() >= 15.8:
                            raise KeyboardInterrupt('synthetic host interruption')
                        return original(size)
                    device.read = read
                devices.append(device)
                return device
            app = guided.GuidedCapture(root, factory, clock=clock,
                                       input_fn=lambda prompt: '', print_fn=console.append,
                                       probe_seconds=.3, segment_seconds=.6, replay=True)
            return app, devices, console
        interrupted, devices, _ = make(interrupt=True)
        with self.assertRaises(KeyboardInterrupt): interrupted.run_session()
        state = json.loads((root/'capture.json').read_text(encoding='utf-8'))
        self.assertEqual('interrupted', state['status'])
        self.assertEqual('interrupted', state['attempts'][0]['status'])
        partial = root/state['attempts'][0]['path']/'raw.bin'
        self.assertTrue(partial.stat().st_size > 0)
        self.assertTrue(all(device.closed for device in devices))
        resumed, devices, console = make()
        state = resumed.run_session(resume=True)
        self.assertEqual('complete', state['status'])
        self.assertTrue(state['session_complete'])
        self.assertTrue(all(device.closed for device in devices))
        active = [a for a in state['attempts'] if a['status'] in ('complete', 'needs_review')]
        self.assertEqual([1, 2, 3, 4, '4b', 5, 6, 7, 8, 9], [a['condition_index'] for a in active])
        pair = [a for a in active if a['condition_index'] in (4, '4b')]
        self.assertEqual(['high', 'low'], [a['ambient_level'] for a in pair])
        self.assertEqual(['empty-corridor-fixed-setup']*2, [a['ambient_pair_id'] for a in pair])
        for attempt in pair:
            directory = root/attempt['path']
            metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(attempt['ambient_pair_id'], metadata['ambient_pair_id'])
            self.assertEqual(attempt['ambient_level'], metadata['ambient_level'])
            self.assertGreater(metadata['summary']['frames'], 0)
            self.assertGreater(metadata['ambient_summary']['frames_with_ambient'], 0)
            self.assertIsNotNone(metadata['ambient_summary']['median_of_zone_medians'])
        self.assertTrue(any('支架和场景保持第4段原位' in text for text in console))
        old_dark = root/pair[1]['path']/'raw.bin'
        old_bytes = old_dark.read_bytes()
        before = len(state['attempts'])
        redo, devices, _ = make()
        state = redo.run_session(redo=['4b'])
        self.assertEqual(before+1, len(state['attempts']))
        self.assertEqual('4b', state['attempts'][-1]['condition_index'])
        self.assertEqual('low', state['attempts'][-1]['ambient_level'])
        old_record = next(a for a in state['attempts'] if a['path'] == pair[1]['path'])
        self.assertEqual('superseded', old_record['status'])
        self.assertEqual(old_bytes, old_dark.read_bytes())
        self.assertTrue(all(device.closed for device in devices))
        self.assertEqual(source_hash, hashlib.sha256(RAW.read_bytes()).hexdigest())
        receipt = {'protocol_version': guided.PROTOCOL_VERSION, 'source_sha256': source_hash,
                   'evidence_scope': 'EXISTING_SERIAL_REPLAY_WITH_SYNTHETIC_HOST_CLOCK_AND_INTERRUPTION',
                   'case_count': 1, 'result': 'PASS', 'active_segments': 10,
                   'probe_virtual_seconds': .3, 'capture_virtual_seconds': .6,
                   'ambient_pair_id': pair[0]['ambient_pair_id'], 'redo_4b': True,
                   'interrupted_resume': True, 'device_access': False, 'network_access': False}
        (root/'focused-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')

if __name__ == '__main__': unittest.main(verbosity=2)
