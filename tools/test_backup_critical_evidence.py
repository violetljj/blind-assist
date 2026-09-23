import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

from tools.backup_critical_evidence import create, linked, restore


class BackupTests(unittest.TestCase):
    def test_long_restore_member(self):
        temporary = tempfile.TemporaryDirectory()
        if os.name == 'nt':
            temporary.name = '\\\\?\\' + str(Path(temporary.name).absolute())
        with temporary as tmp:
            name = '/'.join(['segment' * 5] * 8) + '/data'
            import hashlib
            archive = Path(tmp) / 'long.zip'
            with zipfile.ZipFile(archive, 'w') as out:
                out.writestr(name, b'long')
                out.writestr('backup-manifest.json', json.dumps({'files': [{'path': name, 'bytes': 4, 'sha256': hashlib.sha256(b'long').hexdigest()}]}))
            self.assertEqual(restore(archive, Path(tmp) / 'restore')['status'], 'PASS')

    def test_windows_reparse_point_without_path_is_junction(self):
        path = Mock(spec=['lstat', 'is_symlink'])
        path.is_symlink.return_value = False
        path.lstat.return_value.st_file_attributes = 0x400
        self.assertTrue(linked(path))
        path.lstat.return_value.st_file_attributes = 0
        self.assertFalse(linked(path))

    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'source').mkdir()
            (root / 'source/data').write_bytes(b'\x00\xff exact\r\n')
            archive = root / 'backup.zip'
            create(root, {'paths': ['source']}, archive)
            result = restore(archive, root / 'restored')
            self.assertEqual(result['files'], 1)
            self.assertEqual((root / 'restored/source/data').read_bytes(), b'\x00\xff exact\r\n')
            with self.assertRaises(FileExistsError):
                restore(archive, root / 'restored')
            with self.assertRaises(FileExistsError):
                create(root, {'paths': ['source']}, archive)

    def test_missing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                create(Path(tmp), {'paths': ['missing']}, Path(tmp) / 'b.zip')

    def test_tamper_and_traversal(self):
        for name, data, expected_hash in [('data', b'bad', '0' * 64), ('../escape', b'x', '0' * 64)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / 'bad.zip'
                with zipfile.ZipFile(archive, 'w') as out:
                    out.writestr(name, data)
                    out.writestr('backup-manifest.json', json.dumps({'files': [{'path': name, 'bytes': len(data), 'sha256': expected_hash}]}))
                with self.assertRaises(ValueError):
                    restore(archive, Path(tmp) / 'restored')
                self.assertFalse((Path(tmp) / 'escape').exists())


if __name__ == '__main__':
    unittest.main()
