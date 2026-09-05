import tempfile
from pathlib import Path
import unittest
from unittest import mock
import run_obstacle_research as lab


class RoutingTest(unittest.TestCase):
    def test_replay_cannot_silently_become_motion_trial(self):
        args=lab.parser().parse_args(['replay','--output',str(lab.DATA/'test-output')])
        target,argv=lab.command(args)
        self.assertEqual(target.name,'ue_fixed_replay.py')
        self.assertIn('incremental',argv)
        with self.assertRaises(SystemExit):
            lab.parser().parse_args(['replay','--output',str(lab.DATA/'test-output'),'--controller-mode','JOINT'])

    def test_closed_loop_defaults_to_v4_depth_reference(self):
        with mock.patch.object(lab,'engine_root',return_value=Path('engine')),mock.patch.object(Path,'is_file',return_value=True):
            args=lab.parser().parse_args(['closed-loop','--output',str(lab.DATA/'test-output')])
            target,argv=lab.command(args)
        self.assertEqual(target.name,'run_street_closed_loop.py')
        self.assertEqual(argv[argv.index('--map')+1],'StreetLabV4')
        self.assertEqual(argv[argv.index('--controller-mode')+1],'DEPTH_ONLY')
        self.assertEqual(argv[argv.index('--scenario-split')+1],'regression')

    def test_external_output_is_rejected_before_launch(self):
        args=lab.parser().parse_args(['replay','--output',str(lab.REPO/'forbidden-output')])
        with self.assertRaisesRegex(ValueError,'artifacts.local'):lab.command(args)


if __name__=='__main__':unittest.main()
