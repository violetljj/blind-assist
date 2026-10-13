import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from check_frozen_fusion_inputs import check, check_inputs_only, read_json


class CompletenessTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[4] / 'artifacts.local/work/frozen-fusion-input-tests'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.labels = {'capture-a': {'class': 'ahead'}, 'capture-q': {'class': 'quiet'}}
        self.label_file = self.root / 'labels.json'
        self.write_labels()
        for name in self.labels:
            self.make_capture(self.raw / name)

    def write_labels(self, encoding='utf8'):
        self.label_file.write_text(json.dumps(self.labels), encoding=encoding)

    @staticmethod
    def write_rows(path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')

    def make_capture(self, folder):
        folder.mkdir()
        frame = dict(type='cnh_frame', rows=8, cols=8, bins=16, seq=1, ms=100,
                     hist_raw=[[1] * 16 for _ in range(64)], hist_scaler=[[0] * 16 for _ in range(64)],
                     ambient_raw=[1] * 64, ambient_scaler=[0] * 64,
                     distance_mm=[1000] * 64, target_status=[5] * 64)
        payload = (json.dumps(frame) + '\n').encode()
        (folder / 'tof.raw.jsonl').write_bytes(payload)
        self.write_rows(folder / 'tof.index.jsonl', [dict(index=1, valid_frame=True, sequence=1,
                        device_timestamp_ms=100, received_elapsed_ns=100_000_000,
                        raw_byte_offset=0, raw_byte_length=len(payload), imu_alignment={'imu_record_index': 1})])
        self.write_rows(folder / 'imu.index.jsonl', [dict(index=1, valid_frame=True,
                        received_elapsed_ns=99_000_000, gravity_unit=[0, -1, 0])])
        # Payload-integrity fixture only; the checker explicitly does not decode images.
        image = b'engineering fixture, not a decoded camera image'
        (folder / 'frame.jpg').write_bytes(image)
        self.write_rows(folder / 'rgb.jsonl', [dict(index=1, received_elapsed_ns=100_000_000,
                        file='frame.jpg', bytes=len(image), sha256=hashlib.sha256(image).hexdigest())])

    def result(self, reports=None):
        return check(self.raw, self.label_file, reports)

    def test_missing_session_and_zero_ready_cannot_disappear(self):
        self.assertEqual(self.result()['status'], 'COMPLETE')
        self.labels['capture-missing'] = {'class': 'head'}
        self.write_labels()
        self.assertEqual(self.result()['status'], 'INCOMPLETE')
        del self.labels['capture-missing']
        self.write_labels()
        rows = {name: dict(cls=label['class'], ready_frames=3, **{'pass': False}) for name, label in self.labels.items()}
        for version in (1, 2):
            (self.root / f'frozen_fusion_eval_v{version}.json').write_text(json.dumps({'sessions': rows}), encoding='utf8')
        self.assertEqual(self.result(self.root)['status'], 'COMPLETE')  # Failed science can be complete.
        rows['capture-a']['ready_frames'] = 0
        rows['capture-a']['pass'] = None
        (self.root / 'frozen_fusion_eval_v2.json').write_text(json.dumps({'sessions': rows}), encoding='utf8')
        self.assertEqual(self.result(self.root)['status'], 'INCOMPLETE')

    def test_v3_report_checked_when_present(self):
        rows = {name: dict(cls=label['class'], ready_frames=3, **{'pass': True}) for name, label in self.labels.items()}
        for version in (1, 2):
            (self.root / f'frozen_fusion_eval_v{version}.json').write_text(json.dumps({'sessions': rows}), encoding='utf8')
        result = self.result(self.root)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertTrue(any('v3 report absent' in w for w in result['warnings']))
        bad = {name: dict(row) for name, row in rows.items()}
        bad['capture-a'].update(ready_frames=0, **{'pass': None})
        (self.root / 'frozen_fusion_eval_v3.json').write_text(json.dumps({'sessions': bad}), encoding='utf8')
        self.assertEqual(self.result(self.root)['status'], 'INCOMPLETE')

    def test_duplicate_labels_rejected(self):
        self.label_file.write_text('{"capture-a": {}, "capture-a": {}}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            read_json(self.label_file)

    def test_bom_labels_rejected_before_frozen_reader(self):
        self.write_labels('utf-8-sig')
        with self.assertRaisesRegex(ValueError, 'without BOM'):
            self.result()

    def test_malformed_index_does_not_silently_drop_a_record(self):
        path = self.raw / 'capture-a/rgb.jsonl'
        path.write_text(path.read_text() + '{truncated\n', encoding='utf8')
        result = self.result()
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertIn('rgb.jsonl:2', result['errors'][0])
        self.assertIn('capture-q', result['input_diagnostics'])

    def test_nonempty_placeholder_is_not_a_valid_index(self):
        (self.raw / 'capture-a/tof.index.jsonl').write_text('{}\n', encoding='utf8')
        self.assertEqual(self.result()['status'], 'INCOMPLETE')

    def test_raw_byte_range_and_cnh_shape(self):
        folder = self.raw / 'capture-a'
        path = folder / 'tof.index.jsonl'
        row = json.loads(path.read_text())
        row['raw_byte_length'] += 1
        self.write_rows(path, [row])
        self.assertIn('raw byte range', self.result()['errors'][0])
        frame = json.loads((folder / 'tof.raw.jsonl').read_bytes())
        frame['hist_raw'] = [[1] * 16 for _ in range(63)]
        payload = (json.dumps(frame) + '\n').encode()
        (folder / 'tof.raw.jsonl').write_bytes(payload)
        row['raw_byte_length'] = len(payload)
        self.write_rows(path, [row])
        self.assertIn('hist_raw', self.result()['errors'][0])

    def test_indexed_invalid_tail_is_retained(self):
        folder = self.raw / 'capture-a'
        raw = folder / 'tof.raw.jsonl'
        payload = raw.read_bytes()
        tail = b'{"type":"cnh_frame",'
        raw.write_bytes(payload + tail)
        path = folder / 'tof.index.jsonl'
        self.write_rows(path, [json.loads(path.read_text()), dict(index=2, valid_frame=False,
                        raw_byte_offset=len(payload), raw_byte_length=len(tail))])
        result = self.result()
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['input_diagnostics']['capture-a']['invalid_tof_records_retained'], 1)
        self.assertEqual(result['input_diagnostics']['capture-a']['valid_cnh_frames'], 1)

    def test_missing_and_corrupted_rgb_are_actionable(self):
        folder = self.raw / 'capture-a'
        path = folder / 'rgb.jsonl'
        row = json.loads(path.read_text())
        row['file'] = 'missing.jpg'
        self.write_rows(path, [row])
        self.assertIn('missing/empty RGB payload', self.result()['errors'][0])
        row['file'] = 'frame.jpg'
        row['sha256'] = '0' * 64
        self.write_rows(path, [row])
        self.assertIn('SHA256 differs', self.result()['errors'][0])

    def test_broken_imu_reference_differs_from_recorded_unavailability(self):
        path = self.raw / 'capture-a/tof.index.jsonl'
        row = json.loads(path.read_text())
        row['imu_alignment'] = {'imu_record_index': 999}
        self.write_rows(path, [row])
        self.assertIn('IMU reference 999', self.result()['errors'][0])
        row['imu_alignment'] = {'imu_record_index': None}
        self.write_rows(path, [row])
        rgb_path = self.raw / 'capture-a/rgb.jsonl'
        rgb = json.loads(rgb_path.read_text())
        rgb['received_elapsed_ns'] = 300_000_000
        self.write_rows(rgb_path, [rgb])
        result = self.result()
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['input_diagnostics']['capture-a']['cnh_without_imu_reference'], 1)
        self.assertEqual(result['input_diagnostics']['capture-a']['cnh_without_rgb_receipt_match'], 1)
        self.assertTrue(any('RGB branch unavailable if ready' in note for note in result['warnings']))

    def test_input_only_cannot_claim_label_or_result_completeness(self):
        result = check_inputs_only(self.raw)
        self.assertEqual(result['status'], 'INPUTS_ONLY_COMPLETE')
        self.assertFalse(result['labels_checked'])
        self.assertFalse(result['evaluator_reports_checked'])
        self.assertEqual(result['sessions'], 2)

    def test_legacy_imu_without_additive_gravity_is_unavailable_not_corrupt(self):
        path = self.raw / 'capture-a/imu.index.jsonl'
        old = dict(index=1, valid_frame=True, received_elapsed_ns=99_000_000)
        for legacy in (old, dict(old, gravity_unit=None)):
            with self.subTest(legacy=legacy):
                self.write_rows(path, [legacy])
                result = self.result()
                self.assertEqual(result['status'], 'COMPLETE')
                diag = result['input_diagnostics']['capture-a']
                self.assertEqual(diag['cnh_without_imu_reference'], 0)
                self.assertEqual(diag['imu_without_gravity'], 1)
                self.assertEqual(diag['cnh_without_usable_gravity'], 1)
                self.assertTrue(any('supported legacy schema' in note for note in result['warnings']))
        self.write_rows(path, [dict(old, gravity_unit=[0, -1, 0]), dict(old, index=2)])
        result = self.result()
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['input_diagnostics']['capture-a']['cnh_without_usable_gravity'], 0)


if __name__ == '__main__':
    unittest.main()
