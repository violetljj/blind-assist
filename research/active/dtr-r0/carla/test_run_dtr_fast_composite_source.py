"""Owned capture subprocess exits promptly after the server dies."""
import subprocess
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace
import run_dtr_fast_composite_source as runner


class ProcessCleanupTest(unittest.TestCase):
    def test_incomplete_startup_pixels_cannot_create_source_run(self):
        parent=Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root=Path(directory);probe=root/'probe';probe.mkdir()
            (probe/'result.json').write_text(json.dumps({'status':'CAPTURE_PASS_PENDING_INDEPENDENT_PIXELS'}))
            (probe/'pixel-validation.json').write_text(json.dumps({'status':'PASS','verified_images':200}))
            output=root/'source'
            with self.assertRaisesRegex(ValueError,'Startup pixels incomplete'):
                runner.run(SimpleNamespace(output=output,previous=root,startup_probe=probe))
            self.assertFalse(output.exists())

    def test_dead_server_terminates_only_its_capture_child(self):
        parent=Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            children=[]
            popen=subprocess.Popen
            def launch(*args,**kwargs):
                child=popen(*args,**kwargs);children.append(child);return child
            with mock.patch.object(runner.subprocess,'Popen',side_effect=launch):
                with self.assertRaisesRegex(RuntimeError,'server exited'):
                    runner.command([sys.executable,'-c','import time; time.sleep(60)'],
                                   Path(directory)/'capture.log',mock.Mock(poll=lambda:1))
            self.assertEqual(len(children),1)
            self.assertIsNotNone(children[0].poll())


if __name__=='__main__':unittest.main()
