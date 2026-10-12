import json
from pathlib import Path
import tempfile
import unittest

from check_frozen_fusion_inputs import REQUIRED, check, read_json


class CompletenessTests(unittest.TestCase):
    def test_missing_session_and_zero_ready_cannot_disappear(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / 'raw'; raw.mkdir()
            labels = {'capture-a': {'class': 'ahead'}, 'capture-q': {'class': 'quiet'}}
            label_file = root / 'labels.json'
            label_file.write_text(json.dumps(labels), encoding='utf8')
            for name in labels:
                session = raw / name; session.mkdir()
                for filename in REQUIRED:
                    (session / filename).write_text('{}\n', encoding='utf8')
            self.assertEqual(check(raw, label_file)['status'], 'COMPLETE')
            labels['capture-missing'] = {'class': 'head'}
            label_file.write_text(json.dumps(labels), encoding='utf8')
            self.assertEqual(check(raw, label_file)['status'], 'INCOMPLETE')
            del labels['capture-missing']
            label_file.write_text(json.dumps(labels), encoding='utf8')
            rows = {name: dict(cls=label['class'], ready_frames=3, **{'pass': False}) for name, label in labels.items()}
            for version in (1, 2):
                (root / f'frozen_fusion_eval_v{version}.json').write_text(json.dumps({'sessions': rows}), encoding='utf8')
            self.assertEqual(check(raw, label_file, root)['status'], 'COMPLETE')  # Failed science can be complete.
            rows['capture-a']['ready_frames'] = 0
            rows['capture-a']['pass'] = None
            (root / 'frozen_fusion_eval_v2.json').write_text(json.dumps({'sessions': rows}), encoding='utf8')
            self.assertEqual(check(raw, label_file, root)['status'], 'INCOMPLETE')

    def test_duplicate_labels_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'labels.json'
            path.write_text('{"capture-a": {}, "capture-a": {}}', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_json(path)


if __name__ == '__main__':
    unittest.main()
