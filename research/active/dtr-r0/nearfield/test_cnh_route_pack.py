"""Small transport fixtures only; no simulated rendering evidence is created."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from cnh_route_pack import pack, sha, write


class LosslessPackagingTest(unittest.TestCase):
    def test_all_members_read_back_originals_survive_and_rerun_cannot_overwrite(self):
        parent = Path(__file__).resolve().parents[4]/'artifacts.local/work/cnh-route-comparison-20260924/tests/pack'
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='synthetic-format-fixture-', dir=parent))
        folder = root/'frames/example'; folder.mkdir(parents=True)
        payload = bytes(range(256))*100
        (folder/'transport.bin').write_bytes(payload)
        (folder/'metadata.json').write_text('{"test_fixture_only":true}\n')
        write(root/'raw-manifest.json', dict(frames=[dict(id='example', folder='frames/example')]))
        write(root/'format-receipt.json', dict(status='PASS_FORMAT_ONLY_GEOMETRIC_CANARY_REQUIRED', frame_count=1))
        before = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
        result = pack(root)
        self.assertEqual(result['status'], 'PASS_LOSSLESS_ARCHIVES_VERIFIED_ORIGINALS_RETAINED')
        self.assertEqual(result['frame_count'], 1)
        self.assertTrue(result['originals_retained'])
        for name, digest in before.items():
            self.assertEqual(sha(root/name), digest)
        members = result['frames'][0]['source_files']
        self.assertEqual({row['path'] for row in members}, {'frames/example/transport.bin', 'frames/example/metadata.json'})
        with zipfile.ZipFile(root/result['frames'][0]['archive_path']) as archive:
            self.assertEqual(archive.read('frames/example/transport.bin'), payload)
        manifest_sha = sha(root/'packed-raw/manifest.json')
        with self.assertRaises(FileExistsError):
            pack(root)
        self.assertEqual(sha(root/'packed-raw/manifest.json'), manifest_sha)
        self.assertGreater(result['budget_forecast']['originals_plus_archives_gib'], result['budget_forecast']['archive_payload_gib'])
        # Keep this tiny task-owned fixture under canonical artifacts as diagnostic
        # evidence; it is explicitly not an engine/capture acceptance result.
        write(root/'test-result.json', dict(status='PASS', fixture_only=True, bytes=len(payload)))


if __name__ == '__main__':
    unittest.main()
