"""Pure NumPy SE3/FOV/core/OBSTACLE contract tests; no model calls."""
import unittest
import numpy as np
import cnh_candidate_query_coverage_dev as C


def pose(yaw=0.,translation=(0.,0.,0.)):
    p=np.eye(4);p[:3,:3]=C.R.rotation(yaw);p[:3,3]=translation
    return p


class CandidateCoverageChecks(unittest.TestCase):
    def test_complete_se3_target_and_historical_sensor_mapping(self):
        sensor=pose(23.,(.3,.1,-.6));target=pose(-17.,(-.2,0.,-.8))
        query=np.linalg.inv(target)@sensor
        recovered=C.candidate_frame(sensor,query)
        np.testing.assert_allclose(recovered,target,atol=1e-12)
        points=np.array([[.29,.42,1.2],[-.1,.8,1.8]])
        world=C.world_points(points,recovered)
        historical=pose(11.,(.1,.02,-1.1))
        observed=C.local_from_world(world,historical)
        full=np.linalg.inv(historical)@sensor@np.linalg.inv(query)
        np.testing.assert_allclose(observed,points@full[:3,:3].T+full[:3,3],atol=1e-12)
        np.testing.assert_allclose(C.local_from_world(world,recovered),points,atol=1e-12)

    def test_fov_and_range_boundary_crossings(self):
        check=C.boundary_countercheck()
        self.assertEqual(check['actual'],check['expected'])
        edge=np.tan(np.pi/8)
        points=np.array([[edge,0,1],[0,edge,1],[0,0,-1]])
        np.testing.assert_array_equal(C.fov_local(points),[True,True,False])
        self.assertFalse(C.fov_local(points[:1],margin_deg=3.)[0])

    def test_query_rotation_can_change_sparse_coverage(self):
        sensor=np.repeat(np.eye(4)[None],16,axis=0)
        forward=np.repeat(np.eye(4)[None],16,axis=0)
        side=np.repeat(pose(-30.)[None],16,axis=0)
        points=np.array([[0.,0.,1.5]])
        first=C.coverage_probe(sensor,forward,10,points=points)
        second=C.coverage_probe(sensor,side,10,points=points)
        self.assertTrue(first['fresh'][0,0])
        self.assertFalse(second['fresh'][0,0])

    def test_equal_query_equal_masks_and_different_query_may_equal_masks(self):
        sensor=np.repeat(np.eye(4)[None],16,axis=0)
        query=sensor.copy()
        points=np.array([[0.,0.,1.]])
        first=C.coverage_probe(sensor,query,10,points=points)
        same=C.coverage_probe(sensor,query.copy(),10,points=points)
        other=C.coverage_probe(sensor,np.repeat(pose(-1.)[None],16,axis=0),10,points=points)
        for key in ('core','fresh','missing'):
            np.testing.assert_array_equal(first[key],same[key])
            np.testing.assert_array_equal(first[key],other[key])

    def test_nominal_heading_uses_candidate_not_sensor_displacement(self):
        sensor=pose(75.,(.2,0.,.3));candidate=pose(-20.,(0.,0.,.3))
        nominal=C.nominal_from_candidate(sensor,candidate)
        np.testing.assert_allclose(nominal[:3,:3],C.R.rotation(-20.)@C.R.rotation(-10.,'x'),atol=1e-12)
        np.testing.assert_array_equal(nominal[:3,3],sensor[:3,3])

    def test_future_history_cannot_change_current_masks(self):
        sensor=np.repeat(np.eye(4)[None],16,axis=0);query=sensor.copy()
        first=C.coverage_probe(sensor,query,5)
        sensor[6:]=pose(90.,(10.,0.,10.));query[6:]=pose(110.)
        second=C.coverage_probe(sensor,query,5)
        for key in ('world','core','fresh','missing'):
            np.testing.assert_array_equal(first[key],second[key])

    def test_obstacle_priority_and_sparse_or_invalid_inputs_cannot_clear(self):
        for covered in (False,True):
            for valid in (None,False,True):
                self.assertEqual(C.partition_state(True,True,covered,valid),'OBSTACLE')
        self.assertEqual(C.partition_state(False,True,False,True),'UNKNOWN')
        self.assertEqual(C.partition_state(False,True,True,None),'UNKNOWN')
        self.assertEqual(C.partition_state(False,True,True,True),'CLEAR')
        sensor=np.repeat(np.eye(4)[None],16,axis=0)
        sparse=C.coverage_probe(sensor,sensor.copy(),10)
        self.assertFalse(sparse['full_query_coverage_verified'])
        self.assertEqual(C.partition_state(False,True,sparse['full_query_coverage_verified'],True),'UNKNOWN')


if __name__=='__main__':unittest.main()
