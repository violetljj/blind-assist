"""CPU analytic checks for all-visible-surface geometry and dynamic groups."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from worlds_verify import camera_points,corridor_masks,visible_support,verified_spec,sha
from worlds_evaluate import all_states,localization


class WorldsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_spec_identity_allows_serialization_only(self):
        artifact_root=Path(__file__).resolve().parents[4]/'artifacts.local/nearfield/worlds-20260908'
        with tempfile.TemporaryDirectory(dir=artifact_root) as directory:
            source=Path(directory)/'source.json'; evaluator=Path(directory)/'evaluator.json'
            value=dict(schema='nf-g14-worlds-spec-v1',cases=[dict(x=1)])
            source.write_text(json.dumps(value,indent=2)+'\n')
            evaluator.write_text(json.dumps(value))
            digest=sha(source)
            self.assertEqual(verified_spec(source,evaluator,digest),value)
            evaluator.write_text(json.dumps(dict(value,cases=[dict(x=2)])))
            with self.assertRaisesRegex(ValueError,'semantic identity'):
                verified_spec(source,evaluator,digest)
            evaluator.write_text(json.dumps(value))
            source.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'byte identity'):
                verified_spec(source,evaluator,digest)

    def test_axial_depth_and_camera_projection(self):
        camera=dict(x=1000.,y=100.,z=1.82,pitch=0.,yaw=0.,roll=0.)
        points=camera_points(torch.full((3,3),2.,dtype=torch.float64),camera)
        torch.testing.assert_close(points[1,1],torch.tensor([1002.,100.,1.82],dtype=torch.float64))
        self.assertAlmostEqual(points[0,0,0].item(),1002.)
        self.assertLess(points[0,0,1].item(),100.)
        self.assertGreater(points[0,0,2].item(),1.82)
        self.assertGreater(float(torch.linalg.vector_norm(points[0,0]-torch.tensor([1000.,100.,1.82]))),2.)

    def test_shape_independent_query_bounds_and_distractors(self):
        wearer=dict(x=0.,y=0.,z=.12,pitch=0.,yaw=0.,roll=0.)
        points=torch.tensor([[2.,0.,1.],[2.,0.,1.7],[2.,.7,1.7],[4.,0.,1.7],[2.,0.,.12],[2.,0.,1.7]])
        valid=torch.tensor([True,True,True,True,True,False])
        masks=corridor_masks(points,valid,wearer)
        self.assertEqual(masks[0].tolist(),[True,False,False,False,False,False])
        self.assertEqual(masks[1].tolist(),[False,True,False,False,False,False])
        # Asset identity/shape/name is absent from this geometric API.
        translated=points+torch.tensor([1000.,100.,0.])
        shifted=dict(wearer,x=1000.,y=100.)
        torch.testing.assert_close(masks,corridor_masks(translated,valid,shifted))
        with self.assertRaisesRegex(ValueError,'zero yaw'):corridor_masks(points,valid,dict(wearer,yaw=5.))

    def test_three_native_pixels_and_thin_support_preserved(self):
        camera=dict(x=0.,y=0.,z=1.82,pitch=0.,yaw=0.,roll=0.)
        wearer=dict(x=0.,y=0.,z=.12,pitch=0.,yaw=0.,roll=0.)
        native=torch.zeros(360,640)
        native[250,318:321]=2.;native[180,318:320]=2.
        target,support,masks,valid=visible_support(native,camera,wearer)
        self.assertEqual(target.tolist(),[1,0]);self.assertEqual(tuple(support.shape),(2,18,32))
        self.assertTrue(bool(support[0].any()));self.assertEqual(int(valid.sum()),5)
        native[180,320]=2.
        self.assertEqual(visible_support(native,camera,wearer)[0].tolist(),[1,1])

    def test_three_state_truth_not_inferred_from_relation_and_dynamic_denominator(self):
        samples=[dict(sample_id=r,group_id='world_layout_group_with_underscores') for r in ('BODY','HEAD','NEITHER')]
        metadata={r:dict(relation=r) for r in ('BODY','HEAD','NEITHER')}
        targets={'BODY':[1,0,-1,-1],'HEAD':[0,1,-1,-1],'NEITHER':[1,1,-1,-1]}
        rows={r:np.array(targets[r],float) for r in targets};thresholds=[dict(value=.5)]*4
        result=all_states(rows,samples,targets,thresholds,metadata)
        self.assertEqual(result['BODY_HEAD_joint_correct'],1)
        maps={r:np.ones((2,18,32),np.float32) for r in targets};truth={r:np.ones((2,18,32),np.uint8) for r in targets}
        metrics=localization(maps,samples,targets,truth,[dict(value=.5)]*2)
        self.assertEqual(metrics['HEAD']['fixed_0.5']['positive_samples'],2)
        self.assertEqual(metrics['HEAD']['fixed_0.5']['peak_hits'],2)


if __name__=='__main__':unittest.main()
