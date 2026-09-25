import unittest
import numpy as np
from cnh_track_a_geometry import (box_mesh,cylinder_mesh,raycast,signed_margin,
                                 clip_triangles,triangle_areas,surface_quadrature)

class GeometryTests(unittest.TestCase):
    def test_first_hit_plane_and_nohit(self):
        mesh=np.concatenate([box_mesh([-1,-1,2],[1,1,2.1]),box_mesh([-1,-1,4],[1,1,4.1])])
        hit=raycast([0,0,0],[[0,0,3],[2,0,1]],mesh,np.repeat([7,8],12),.4)
        self.assertAlmostEqual(hit['distance'][0],2)
        self.assertEqual(hit['object_id'].tolist(),[7,-1])
        self.assertEqual(hit['rho'][0],.4)
        self.assertAlmostEqual(hit['cos'][0],1)
        self.assertTrue(np.isinf(hit['distance'][1]))

    def test_cylinder_analytic_chord_bound(self):
        radius=.2; mesh=cylinder_mesh([0,0,2],radius,1)
        angle=(np.arange(32)+.5)*2*np.pi/64
        # Orthogonal radial probes: chord sagitta is the metric error bound.
        for a in angle:
            radial=np.array([np.cos(a),0,np.sin(a)])
            h=raycast(np.array([0,0,2])+radial,[-radial],mesh,3,.6)
            self.assertLessEqual(abs(h['distance'][0]-(1-radius)),radius*(1-np.cos(np.pi/64))+1e-10)

    def test_surface_margin_inside_thin_rod_contact_separation(self):
        low=np.array([-1,-1,-1]); high=-low
        rod=cylinder_mesh([0,0,0],.005,.4)
        self.assertAlmostEqual(signed_margin(rod,low,high),1-.005/np.sqrt(2),places=7)
        self.assertAlmostEqual(signed_margin(box_mesh([1,-.2,-.2],[1.2,.2,.2]),low,high),0)
        self.assertAlmostEqual(signed_margin(box_mesh([1.1,-.2,-.2],[1.2,.2,.2]),low,high),-.1)

    def test_triangle_interior_not_vertices_margin(self):
        tri=np.array([[[-2,0,0],[2,0,0],[0,2,0]]])
        self.assertAlmostEqual(signed_margin(tri,[-1,-1,-1],[1,1,1]),1)

    def test_zero_area_contact_witness(self):
        tri=np.array([[[1,1,1],[2,1,1],[1,2,1]]],float)
        value,point=signed_margin(tri,[-1,-1,-1],[1,1,1],return_witness=True)
        self.assertAlmostEqual(value,0)
        np.testing.assert_allclose(point,[1,1,1],atol=1e-9)
        self.assertTrue(np.all(point>=-1-1e-9) and np.all(point<=1+1e-9))
        # The only contact is the vertex, not a positive-area clipped surface.
        self.assertEqual(len(clip_triangles(tri,[-1,-1,-1],[1,1,1])),0)
        self.assertEqual(signed_margin(tri,[-1,-1,-1],[1,1,1]),value)

    def test_clip_area_and_quadrature(self):
        plane=np.array([[[-2,-2,0],[2,-2,0],[2,2,0]],[[-2,-2,0],[2,2,0],[-2,2,0]]])
        clipped=clip_triangles(plane,[-1,-1,-1],[1,1,1])
        self.assertAlmostEqual(triangle_areas(clipped).sum(),4)
        points,weights=surface_quadrature(clipped,.2)
        self.assertAlmostEqual(weights.sum(),4)
        self.assertTrue(np.all(np.abs(points)<=1))

    def test_camera_pitch_roll_do_not_enter_query_margin(self):
        # Q geometry is fixed; camera extrinsics only change observation rays.
        mesh=box_mesh([-.1,.1,1],[.1,.2,1.1]); low=[-.3,-.2,.3]; high=[.3,.42,3]
        before=signed_margin(mesh,low,high)
        for theta in [-.25,0,.2]:
            rotation=np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],
                               [0,np.sin(theta),np.cos(theta)]])
            raycast([0,0,0],np.array([[0,0,1]])@rotation.T,mesh,1,.5)
            self.assertEqual(before,signed_margin(mesh,low,high))

if __name__=='__main__': unittest.main()

