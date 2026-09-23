"""Pure depth-composition controls for inserted asset instance transport."""
import unittest
import numpy as np
from cnh_route_insert_launch import compose_instance_ids, validate_spec


class InsertInstanceTests(unittest.TestCase):
    def test_visible_occluded_and_background(self):
        scene = np.array([[2,1,4,3]],dtype=np.float32)
        isolated = np.array([[2,2,0,100]],dtype=np.float32)
        np.testing.assert_array_equal(compose_instance_ids(scene,isolated,254),[[254,0,0,0]])

    def test_invalid_scene_remains_unknown_even_with_target(self):
        scene=np.array([[0,np.nan,np.inf,100,-1]],dtype=np.float32)
        isolated=np.full_like(scene,2)
        np.testing.assert_array_equal(compose_instance_ids(scene,isolated,1),np.full((1,5),65535,dtype=np.uint16))

    def test_invalid_isolated_does_not_invent_foreground(self):
        scene=np.full((1,4),2,dtype=np.float32)
        isolated=np.array([[np.nan,0,np.inf,-1]],dtype=np.float32)
        np.testing.assert_array_equal(compose_instance_ids(scene,isolated,1),np.zeros((1,4),dtype=np.uint16))

    def test_fixed_one_mm_depth_agreement(self):
        scene=np.array([[2,2,2]],dtype=np.float32)
        isolated=np.array([[2.0005,2.002,1.998]],dtype=np.float32)
        np.testing.assert_array_equal(compose_instance_ids(scene,isolated,1),[[1,0,65535]])

    def test_rejects_shape_or_unregistered_id(self):
        a=np.ones((2,2),dtype=np.float32)
        for b,identity in ((a[:1],1),(a,255),(a,0)):
            with self.assertRaises(ValueError):
                compose_instance_ids(a,b,identity)

    def test_spec_refuses_benchmark_and_wrong_ids(self):
        spec=dict(scene_layer='INSERTED_ASSET_ENGINEERING_CANARY',benchmark_eligible=False,assets=[dict(id=1,mesh_asset='/Game/a'),dict(id=254,mesh_asset='/Game/b')])
        validate_spec(spec)
        with self.assertRaises(ValueError):
            validate_spec(dict(spec,benchmark_eligible=True))
        with self.assertRaises(ValueError):
            validate_spec(dict(spec,assets=[dict(id=1,mesh_asset='/Game/a')]))


if __name__=='__main__':
    unittest.main()
