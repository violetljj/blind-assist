"""A consistent Git checkout can still violate an independent frozen lock."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from check_frozen_carla_bytes import check, MANIFEST


class FrozenBytesTest(unittest.TestCase):
    def test_raw_bytes_normalization_corruption_and_missing_target(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / 'tools').mkdir()
            raw = b'{\r\n  "frozen": true\r\n}\r\n'
            target = root / 'protocol.json'
            source = root / 'lock.json'
            source.write_text(json.dumps({'locks': [{'sha256': hashlib.sha256(raw).hexdigest()}]}))
            (root / MANIFEST).write_text(json.dumps({'locks': [{
                'path': 'protocol.json', 'lock_source': 'lock.json',
                'lock_pointer': ['locks', 0, 'sha256'],
            }]}))
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                target.write_bytes(raw)
                self.assertEqual(0, check(root))
                for damaged in (raw.replace(b'\r\n', b'\n'), raw.replace(b'true', b'null')):
                    target.write_bytes(damaged)
                    self.assertEqual(1, check(root))
                    self.assertEqual(damaged, target.read_bytes())
                target.unlink()
                self.assertEqual(1, check(root))
                target.write_bytes(raw)
                source.write_text('{}')
                self.assertEqual(1, check(root))

    def test_empty_manifest_is_not_success(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / 'tools').mkdir()
            (root / MANIFEST).write_text('{"locks": []}')
            with self.assertRaisesRegex(ValueError, 'empty'):
                check(root)
