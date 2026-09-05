"""Crash-boundary checks for the live worker's durable policy state."""
import json
from pathlib import Path
import tempfile
import unittest

from street_live_policy import MotionPolicy
from street_live_server import restore_checkpoints


class LiveResumeTest(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[4]/'artifacts.local/tmp'
        root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='street-resume-test-',dir=root)
        self.output=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def row(self,index,**extra):
        return {'prediction':{'episode_id':'episode_0000','sample_index':index},**extra}

    def checkpoint(self,index):
        policy=MotionPolicy('CANDIDATE_DTR')
        policy.target_y=.72
        value={'model_sha256':'weights','controller_mode':policy.mode,'prediction_engine':'incremental',
               'last_sample_index':index,'policy':policy.__dict__,'last_response':self.row(index,authority='checkpoint')}
        (self.output/'episode_0000-checkpoint.json').write_text(json.dumps(value))

    def restore(self):
        return restore_checkpoints(self.output,'CANDIDATE_DTR','incremental','weights')

    def test_uncommitted_journal_step_is_not_reused(self):
        self.checkpoint(1)
        (self.output/'responses.jsonl').write_text('\n'.join(json.dumps(self.row(i)) for i in range(3))+'\n{"truncated":')
        policies,responses=self.restore()
        self.assertEqual(set(responses),{('episode_0000',0),('episode_0000',1)})
        self.assertEqual(responses[('episode_0000',1)]['authority'],'checkpoint')
        self.assertEqual(policies['episode_0000'].target_y,.72)

    def test_journal_without_checkpoint_reconstructs_from_zero(self):
        (self.output/'responses.jsonl').write_text(json.dumps(self.row(0)))
        self.assertEqual(self.restore(),({},{}))

    def test_checkpoint_recovers_last_response_without_journal(self):
        self.checkpoint(1)
        self.assertEqual(set(self.restore()[1]),{('episode_0000',1)})

    def test_changed_model_or_engine_is_rejected(self):
        self.checkpoint(1)
        for engine,weights in [('batch','weights'),('incremental','changed')]:
            with self.assertRaises(RuntimeError):
                restore_checkpoints(self.output,'CANDIDATE_DTR',engine,weights)

    def test_changed_action_support_contract_is_rejected(self):
        self.checkpoint(1)
        with self.assertRaisesRegex(RuntimeError,'action footprint'):
            restore_checkpoints(self.output,'CANDIDATE_DTR','incremental','weights','cadence')


if __name__=='__main__': unittest.main()
