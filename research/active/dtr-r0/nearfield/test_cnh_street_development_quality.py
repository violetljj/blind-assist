import copy
import unittest
from cnh_street_development_quality import build_ledger, SCOPE


def fixture():
    layouts = [dict(layout_id=lid, physical_site_id='Street200V7-single-street-block',
        environment_category='sidewalk', clips=[dict(id='centre', poses=[{}])]) for lid in ('a', 'b')]
    rows = [dict(layout_id=r['layout_id'], id=r['layout_id'], folder=r['layout_id'],
        physical_site_id=r['physical_site_id'], environment_category='sidewalk',
        clip_id='centre', pose_index=0, nominal_time_s=0, data_role='Development', asset_ids=[1,254]) for r in layouts]
    transport = dict(status='PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY', frame_count=2,
        source_gate='NOT_ADMITTED', frames=[dict(r, hashes={'depth':'abc'}) for r in rows])
    return dict(scope=SCOPE, data_role='Development', layouts=layouts), dict(frames=rows), transport


class QualityTests(unittest.TestCase):
    def test_pending_does_not_block_or_promote(self):
        ledger = build_ledger(*fixture(), 'hash')
        self.assertFalse(ledger['formal_eligible'])
        self.assertEqual(ledger['physical_site_count'], 1)
        for row in ledger['layouts']:
            self.assertEqual(row['disposition'], 'AVAILABLE_FOR_PROVISIONAL_DEVELOPMENT')
            self.assertEqual(len(row['pending_checks']), 3)
            self.assertNotEqual(row['asset_isolation']['status'], 'PASS')

    def test_fail_quarantines_only_affected_layout_and_is_sticky(self):
        result = dict(capture_manifest_sha256='hash', results=[dict(layout_id='a', metric='geometry',
            status='FAIL', scope='original gate', evidence='receipt.json')])
        later = copy.deepcopy(result); later['results'][0]['status'] = 'PASS'
        ledger = build_ledger(*fixture(), 'hash', [result, later])
        self.assertEqual(ledger['layouts'][0]['disposition'], 'QUARANTINE_AFFECTED_LAYOUT')
        self.assertEqual(ledger['layouts'][1]['disposition'], 'AVAILABLE_FOR_PROVISIONAL_DEVELOPMENT')
        self.assertEqual(len(ledger['layouts'][0]['checks']['geometry']['evidence']), 2)

    def test_missing_transport_frame_is_local(self):
        spec, manifest, transport = fixture()
        transport['frames'].pop(0); transport['frame_count'] = 1
        ledger = build_ledger(spec, manifest, transport, 'hash')
        self.assertEqual(ledger['layouts'][0]['transport']['status'], 'FAIL')
        self.assertEqual(ledger['layouts'][1]['transport']['status'], 'PASS_RECEIPT_MATCH_ONLY')

    def test_foreign_result_and_test_frames_rejected(self):
        with self.assertRaises(ValueError):
            build_ledger(*fixture(), 'hash', [dict(capture_manifest_sha256='other', results=[])])
        spec, manifest, transport = fixture(); manifest['frames'][0]['split'] = 'test'
        with self.assertRaises(ValueError):
            build_ledger(spec, manifest, transport, 'hash')


if __name__ == '__main__':
    unittest.main()
