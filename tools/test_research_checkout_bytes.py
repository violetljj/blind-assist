"""The portability check must reject changed bytes without repairing evidence."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import subprocess
import tempfile
import unittest

from check_research_checkout_bytes import check


class CheckoutBytesTest(unittest.TestCase):
    def test_exact_checkout_and_corruption(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            path = root / 'research/active/dtr-r0/carla/frozen.py'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'one\ntwo\n')
            (root / '.gitattributes').write_bytes(b'* text eol=lf\n')
            commands = [
                ['init', '--quiet'], ['add', '.'],
                ['-c', 'user.name=Test', '-c', 'user.email=test@invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'fixture'],
            ]
            for command in commands:
                subprocess.run(['git', *command], cwd=root, check=True, capture_output=True)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(0, check(root))
                for changed in (b'one\r\ntwo\r\n', b'changed\n'):
                    path.write_bytes(changed)
                    self.assertEqual(1, check(root))
                    self.assertEqual(changed, path.read_bytes())
                path.unlink()
                self.assertEqual(1, check(root))
