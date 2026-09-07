"""Completion publication must include the host transport, not just UE exit."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import OpenEXR
import ue_native_capture as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        folder=runner.REPO/'artifacts.local/tmp'
        folder.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='ue-runner-test-',dir=folder)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.spec=self.root/'spec.json'
        self.spec.write_text(json.dumps({'cases':[{}]}))

    def args(self):
        return argparse.Namespace(spec=self.spec,output=self.root/'capture',capture='grounding',
                                  engine=Path('unused'),depth_export='exr',timeout=1)

    def fake_process(self,command,env,out,timeout,on_poll=None,corrupt=False):
        self.assertIn(str(out/'source'/'grounding_capture.py').replace('\\','/'),command[2])
        self.assertEqual(Path(env['BA_NEARFIELD_SPEC']),out/'source/spec.json')
        self.assertTrue((out/'source/ue_exr_transport.py').is_file())
        folder=out/'evaluator/native';folder.mkdir(parents=True)
        shape=(1,1) if corrupt else (360,640)
        source=folder/'0000.partial.exr'
        OpenEXR.File({}, {'R':np.ones(shape,dtype=np.float32)}).write(str(source))
        source.replace(folder/'0000.exr')
        runner.write(out/'receipt.json',dict(status='PASS',source_unchanged=True,
                                            frame_count=1,wall_elapsed_s=1))
        runner.write(out/'process-release.json',dict(released=True))

    def test_completed_transport_and_frozen_inputs(self):
        with patch.object(runner,'run_owned',side_effect=self.fake_process):runner.capture(self.args())
        out=self.root/'capture'
        self.assertEqual(json.loads((out/'completion.json').read_text())['status'],'PASS')
        self.assertTrue((out/'evaluator/native/0000.npy').is_file())
        self.assertEqual((out/'source/spec.json').read_bytes(),self.spec.read_bytes())

    def test_transport_failure_overrides_engine_pass(self):
        def fail(*args,**kwargs):return self.fake_process(*args,**kwargs,corrupt=True)
        with patch.object(runner,'run_owned',side_effect=fail):
            with self.assertRaises(ValueError):runner.capture(self.args())
        out=self.root/'capture'
        self.assertEqual(json.loads((out/'receipt.json').read_text())['status'],'FAIL')
        self.assertEqual(json.loads((out/'engine-receipt.json').read_text())['status'],'PASS')
        self.assertEqual(json.loads((out/'completion.json').read_text())['status'],'FAIL')


if __name__=='__main__':unittest.main()
