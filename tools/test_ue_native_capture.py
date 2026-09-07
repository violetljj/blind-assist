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
        self.assertIn(str(out/'source').replace('\\','/'),command[2])
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

    def test_auto_without_plugin_uses_exr(self):
        args=self.args();args.depth_export='auto'
        with patch.object(runner,'run_owned',side_effect=self.fake_process):runner.capture(args)
        launch=json.loads((self.root/'capture/launch.json').read_text())
        self.assertEqual(launch['depth_export'],'exr')

    def test_native_missing_plugin_fails_before_creating_output(self):
        args=self.args();args.depth_export='native'
        with self.assertRaises(ValueError):runner.capture(args)
        self.assertFalse(args.output.exists())

    def test_auto_preserves_motion_adapter_tick_cadence(self):
        args=self.args();args.capture='whisker';args.cadence='auto'
        with patch.object(runner,'run_owned',side_effect=self.fake_process):runner.capture(args)
        launch=json.loads((self.root/'capture/launch.json').read_text())
        self.assertEqual(launch['cadence'],'tick')

    def test_native_rgb_requires_plugin_even_with_exr_depth(self):
        args=self.args();args.rgb_export='native_async'
        with self.assertRaises(ValueError):runner.capture(args)
        self.assertFalse(args.output.exists())

    def test_rgb_probe_checks_pixels_not_only_file_presence(self):
        from PIL import Image
        folder=self.root/'model/sample';folder.mkdir(parents=True)
        Image.new('RGBA',(640,360),(1,2,3,255)).save(folder/'0000.png')
        Image.new('RGBA',(640,360),(1,2,3,255)).save(folder/'0000.reference.png')
        runner.verify_rgb(self.root,1,probe=True)
        Image.new('RGBA',(640,360),(1,2,4,255)).save(folder/'0000.reference.png')
        with self.assertRaisesRegex(ValueError,'differs'):
            runner.verify_rgb(self.root,1,probe=True)

    def test_settling_auto_is_scoped_to_declared_static_source(self):
        self.spec.write_text(json.dumps({'cases':[{}], 'sampling':'THREE_STATIC_SETTLED_POSES_SIMULATED_5HZ_NOT_MOTION_TEST'}))
        for i,(cadence,policy,expected) in enumerate([('burst','auto','reuse'),('tick','auto','full'),('burst','full','full')]):
            args=self.args();args.output=self.root/f'policy{i}';args.cadence=cadence;args.settling_policy=policy
            with patch.object(runner,'run_owned',side_effect=self.fake_process):runner.capture(args)
            self.assertEqual(json.loads((args.output/'launch.json').read_text())['settling_policy'],expected)

    def test_pair_rejects_exr_and_motion_before_launch(self):
        args=self.args();args.pair_export='native_async'
        with self.assertRaisesRegex(ValueError,'depth'):
            runner.capture(args)
        args.depth_export='native';args.capture='whisker'
        with self.assertRaisesRegex(ValueError,'settled'):
            runner.capture(args)
        self.assertFalse(args.output.exists())


if __name__=='__main__':unittest.main()
