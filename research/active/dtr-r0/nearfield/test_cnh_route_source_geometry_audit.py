"""Numerical denominators, radial conversion and fixed-grid gate checks."""
import unittest
import numpy as np
from cnh_route_capture import camera_record
from cnh_route_source_geometry_audit import ray_grid,zone_coverage,distance_gate,thin_layer,aggregate_status


class SourceGeometryAuditTests(unittest.TestCase):
    def test_fixed_grid_and_zone_coverage(self):
        camera=camera_record(dict(x=0,y=0,z=1.6,pitch=0,yaw=0,roll=0),dict(width=640,height=360,hfov_deg=100,baseline_m=.06))
        yy,xx,rays,weights,zones=ray_grid(camera)
        self.assertTrue(np.all(yy%2==1));self.assertTrue(np.all(xx%2==1))
        self.assertEqual(set(zones.tolist()),set(range(64)))
        self.assertTrue(np.all(weights>0));self.assertTrue(np.all(np.abs(rays[:,:2])<=np.tan(np.deg2rad(22.5))))
        self.assertEqual(len(xx),12321)

    def test_all_ray_weight_denominator_includes_missing(self):
        native=np.array([1.,1.,np.inf,np.inf]);pred=np.array([1.,np.inf,1.,np.inf])
        rows,_,_=zone_coverage(native,pred,np.ones(4),np.array([1.,2.,3.,4.]),np.zeros(4,dtype=int))
        zone=rows[0]
        self.assertEqual(zone['angular_weight_sr'],10)
        self.assertEqual(zone['missing_fraction'],.2);self.assertEqual(zone['extra_fraction'],.3)
        self.assertEqual(zone['status'],'FAIL')

    def test_radial_window_not_axial_window(self):
        rows,native,mesh=zone_coverage(np.array([4.]),np.array([4.]),np.array([1.3]),np.ones(1),np.zeros(1,dtype=int))
        self.assertFalse(native[0]);self.assertFalse(mesh[0])
        self.assertEqual(rows[0]['missing_fraction'],0)

    def test_distance_thresholds_and_no_common(self):
        self.assertEqual(distance_gate(np.zeros(10))['status'],'PASS')
        self.assertEqual(distance_gate(np.full(10,.031))['status'],'FAIL')
        self.assertEqual(distance_gate(np.r_[np.zeros(100),.076])['status'],'FAIL')
        self.assertEqual(distance_gate(np.array([]))['status'],'NOT_RUN')

    def test_visible_layer_sampling_gap_and_loss(self):
        depth=np.array([2.,2.]);norm=np.ones(2)
        self.assertEqual(thin_layer(3,np.array([False,False]),depth,depth,norm)['status'],'NOT_RUN')
        self.assertEqual(thin_layer(3,np.array([True,False]),depth,np.array([np.inf,2.]),norm)['status'],'FAIL')
        self.assertEqual(thin_layer(3,np.array([True,False]),depth,np.array([2.01,2.]),norm)['status'],'PASS')
        self.assertEqual(thin_layer(3,np.array([True,False]),depth,depth,norm,np.array([False,False]))['status'],'FAIL')
        self.assertEqual(thin_layer(0,np.array([False,False]),depth,depth,norm)['status'],'NOT_APPLICABLE')

    def test_failure_precedence_and_no_evidence(self):
        self.assertEqual(aggregate_status(['FAIL','NOT_RUN']),'FAIL')
        self.assertEqual(aggregate_status(['PASS','NOT_RUN']),'NOT_RUN')
        self.assertEqual(aggregate_status([]),'NOT_RUN')


if __name__=='__main__':unittest.main()
