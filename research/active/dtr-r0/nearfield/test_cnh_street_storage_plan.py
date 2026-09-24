import unittest
import json
from pathlib import Path
import tempfile
from cnh_street_storage_plan import crosscheck_ids, TRANSPORT_REPLACEMENTS, CANONICAL, plan, sha


class StoragePlanTests(unittest.TestCase):
    def test_crosschecks_are_per_layout_clip_not_global(self):
        rows = [dict(id=f'{layout}-{clip}-{pose}', layout_id=layout, clip_id=clip, pose_index=pose)
            for layout in ('a', 'b') for clip in ('centre', 'removed') for pose in (2, 0, 1)]
        expected = {r['id'] for r in rows if r['pose_index'] in (0, 2)}
        self.assertEqual(crosscheck_ids(rows), expected)

    def test_replacements_are_canonical_and_never_ids_or_rgb(self):
        self.assertTrue(all(name in CANONICAL for names in TRANSPORT_REPLACEMENTS.values() for name in names))
        self.assertTrue(all(name.endswith('.transport.npy') for name in TRANSPORT_REPLACEMENTS))
        self.assertNotIn('instance_left.png', TRANSPORT_REPLACEMENTS)

    def test_dry_run_hash_inventory_and_no_mutation(self):
        temp_root = Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='storage-plan-test-', dir=temp_root) as temp:
            root = Path(temp); rows = []; reports = []
            for pose in range(3):
                folder = root/f'frames/f{pose}'; folder.mkdir(parents=True)
                for name in (*CANONICAL, *TRANSPORT_REPLACEMENTS, 'isolated_depth_1.transport.npy', 'camera.json'):
                    (folder/name).write_bytes(b'fixture only')
                row = dict(id=f'f{pose}', folder=f'frames/f{pose}', layout_id='a', clip_id='centre', pose_index=pose)
                rows.append(row); reports.append(dict(row, hashes={p.name:sha(p) for p in folder.iterdir()}))
            (root/'raw-manifest.json').write_text(json.dumps(dict(frames=rows)))
            (root/'format-receipt.json').write_text(json.dumps(dict(status='PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY', frames=reports)))
            before = {str(p):sha(p) for p in root.rglob('*') if p.is_file()}
            result = plan(root)
            self.assertEqual(result['files_deleted'], 0)
            self.assertEqual(result['crosscheck_frame_count'], 2)
            self.assertEqual(before, {str(p):sha(p) for p in root.rglob('*') if p.is_file()})
            mid = {Path(p['path']).name:p for p in result['frames'][1]['files']}
            self.assertEqual(mid['isolated_depth_1.transport.npy']['action'], 'PLAN_REMOVE_BLOCKED')
            self.assertEqual(mid['camera.json']['action'], 'KEEP')
            self.assertTrue(all(p['original_transport_hash_status']=='MATCH' for p in mid.values()))


if __name__ == '__main__':
    unittest.main()
