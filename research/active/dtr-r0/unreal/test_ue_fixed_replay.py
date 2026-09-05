import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import dataclass
import ue_fixed_replay as fixed


class FixedReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = fixed.REPO / 'artifacts.local' / 'tmp'
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def test_full_causal_prefixes_never_include_future(self):
        @dataclass
        class Episode:
            observations: tuple
        self.assertEqual([e.observations for e in fixed.causal_prefixes(Episode((0, 1, 2)), 2)], [(0,), (0, 1)])

    def test_motion_replay_is_rejected_before_input_access(self):
        with self.assertRaisesRegex(ValueError, 'new motion trajectory'):
            fixed.replay_dataset('absent', 'absent', mode='motion')

    def test_boundary_rejects_names_truth_and_incomplete_prefix(self):
        item = {'episode_id': 'scenario_collision', 'route_frame': {}, 'plan_path': 'plan.json', 'frames': []}
        with self.assertRaisesRegex(ValueError, 'opaque'):
            fixed.validate_episode(item)
        item['episode_id'] = 'episode_0000'
        item['truth'] = True
        with self.assertRaisesRegex(ValueError, 'schema'):
            fixed.validate_episode(item)

    def test_content_hash_and_extra_file_detection(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            root = Path(directory)
            fixed.write(root / 'manifest.json', {'episodes': []})
            fixed.write(root / 'integrity.json', {'files': {'manifest.json': fixed.sha(root / 'manifest.json')}})
            (root / 'integrity.sha256').write_text(fixed.sha(root / 'integrity.json'))
            fixed.verify_dataset(root)
            (root / 'responses.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Unexpected file'):
                fixed.verify_dataset(root)
            (root / 'responses.json').unlink()
            (root / 'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                fixed.verify_dataset(root)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            root = Path(directory)
            (root / 'inside').mkdir()
            (root / 'outside').write_text('x')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                fixed.contained(root / 'inside', '../outside')


if __name__ == '__main__':
    unittest.main()
