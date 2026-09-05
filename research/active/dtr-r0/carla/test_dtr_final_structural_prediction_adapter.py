"""Frozen composition and empty synthetic causal input; no source or detector runs."""
import ast
import copy
from pathlib import Path
import unittest
import dtr_final_structural_prediction_adapter as adapter


class StructuralAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock=adapter.dependency_manifest()
        adapter._load_locked(cls.lock)

    def test_composition_matches_retained_runner_statements(self):
        old=ast.parse((adapter.HERE/'run_dtr_carla_c42_x96_dropout_survival.py').read_text())
        new=ast.parse(Path(adapter.__file__).read_text())
        before=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='_core_x93')
        after=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_core_x93_and_x73')
        self.assertEqual([ast.dump(n) for n in before.body[:-1]],[ast.dump(n) for n in after.body[:-1]])

    def test_dependency_lock_rejects_mutation(self):
        changed=copy.deepcopy(self.lock)
        changed['files']['dtr_carla_x24_plan_adherent_predictor.py']['sha256']='0'*64
        with self.assertRaises(ValueError):adapter._load_locked(changed)
        self.assertFalse(self.lock['historical_WORK_modules_required'])

    def synthetic(self):
        m=adapter.x24.adapter
        image=m.ImageReference(Path('NONEXISTENT_SYNTHETIC_NO_SENSOR_READ'),'',0,1280,720)
        transform={'x':0.,'y':0.,'z':0.,'yaw':0.,'pitch':0.,'roll':0.}
        wearer={'transform':transform,'command_velocity':{'x':1.,'y':0.,'z':0.},'bounding_box_extent':{'x':.3,'y':.3,'z':1.}}
        frames=tuple(m.FrameObservation('synthetic',i,i*.1,i,'s',transform,image,image,wearer,{'authority':'NO_PLAN','path':None}) for i in range(3))
        return m.Episode('synthetic',m.AnchorFrame((0.,0.),0.,(1.,0.),(0.,1.)),frames),m.CameraCalibration(1280,720,90.,1000.)

    def test_empty_synthetic_episode_preserves_structural_frames(self):
        episode,calibration=self.synthetic()
        candidates=[{'sample_index':i,'candidates':[]} for i in range(3)]
        result=adapter.predict_episode(episode,candidates,calibration,dependency_lock=self.lock)
        self.assertEqual(set(result['arms']),{'X24_CORE','X73_STRUCTURAL_GEOMETRY','X94_EVIDENCE_MODEL'})
        for name,rows in result['arms'].items():
            self.assertEqual(len(rows),3)
            self.assertTrue(all(row['route_risk'] is False for row in rows))
            self.assertIn('tracks',result['core_episodes'][name]['frames'][0])
        self.assertIn(adapter.x94.ARM_X94,result['core_episodes']['X94_EVIDENCE_MODEL']['frames'][0]['arms'])
        self.assertEqual(candidates,[{'sample_index':i,'candidates':[]} for i in range(3)])

    def test_privileged_candidates_are_rejected_before_predictors(self):
        episode,calibration=self.synthetic()
        candidates=[{'candidates':[],'truth':False} for _ in range(3)]
        with self.assertRaises((ValueError,RuntimeError)):
            adapter.predict_episode(episode,candidates,calibration,dependency_lock=self.lock)


if __name__=='__main__':unittest.main()
