import unittest
import numpy as np
import surface_contact_core as s


def geometry(points,edges=(),triangles=()):
    p=np.asarray(points,dtype=np.float64).reshape(-1,3)
    return dict(raw_points=p.copy(),fitted_points=p,edges=np.asarray(edges,np.int32).reshape(-1,2),
                triangles=np.asarray(triangles,np.int32).reshape(-1,3))


class SurfaceTests(unittest.TestCase):
    def empty(self):return np.full((360,640),np.nan)

    def test_plane_fit_moves_noisy_vertex(self):
        z=self.empty();z[179:182,319:322]=1.;z[180,320]=1.005
        g=s.depth_to_geometry(z)
        center=np.argmin(np.abs(g['raw_points'][:,1])+np.abs(g['raw_points'][:,2]))
        self.assertAlmostEqual(g['raw_points'][center,0],1.005)
        self.assertLess(g['fitted_points'][center,0],1.001)
        self.assertEqual(g['metadata']['fitted_vertices'],1)

    def test_exact_inverse_plane_preserved(self):
        yy,xx=np.indices((360,640));z=1./(1.+.0001*(xx-320)+.00015*(yy-180))
        fitted,mask=s.fit_numpy(z)
        self.assertTrue(mask[180,320])
        np.testing.assert_allclose(fitted,z,rtol=1e-14,atol=1e-14)

    def test_hole_and_isolated_thin_evidence_retained(self):
        z=self.empty();z[179:182,319:322]=1.;z[180,320]=np.nan;z[200,330]=2.
        g=s.depth_to_geometry(z)
        self.assertEqual(len(g['raw_points']),9)
        self.assertEqual(g['metadata']['fitted_vertices'],0)
        self.assertEqual(len(g['triangles']),0)
        np.testing.assert_array_equal(g['raw_points'],g['fitted_points'])

    def test_discontinuity_not_connected(self):
        z=self.empty();z[180,320]=1.;z[180,321]=1.051
        g=s.depth_to_geometry(z)
        self.assertEqual(len(g['edges']),0)
        self.assertEqual(len(g['raw_points']),2)

    def test_surface_recovers_between_pixel_contact(self):
        z=self.empty();z[180,320:322]=1.
        g=s.depth_to_geometry(z);mid=.5/s.FOCAL
        lo=[.5,mid-.0001,-.0001];hi=[3,mid+.0001,.0001]
        self.assertIsNone(s.contact(g,lo,hi,representation='raw'))
        self.assertIsNone(s.contact(g,lo,hi,representation='fitted'))
        self.assertAlmostEqual(s.contact(g,lo,hi),1.)

    def test_clipped_triangle_interior_and_zero_width(self):
        g=geometry([[1.,-.1,-.1],[1.,.1,-.1],[1.,0.,.1]],triangles=[[0,1,2]])
        lo=[.5,-.01,-.01];hi=[3,.01,.01]
        self.assertIsNone(s.contact(g,lo,hi,representation='raw'))
        self.assertAlmostEqual(s.contact(g,lo,hi),1.)
        self.assertAlmostEqual(s.contact(g,lo,hi,axis='abs_y'),0.)

    def test_edge_clipping_uses_intersection_not_endpoint(self):
        g=geometry([[1.,-.2,0.],[2.,.2,0.]],edges=[[0,1]])
        self.assertAlmostEqual(s.contact(g,[.5,-.1,-.1],[3,.1,.1]),1.25)
        self.assertAlmostEqual(s.contact(g,[.5,.05,-.1],[3,.1,.1],axis='abs_y'),.05)

    def test_empty_is_unknown_and_axial_domain_enforced(self):
        self.assertIsNone(s.contact(geometry([]),[.5,-1,-1],[3,1,1]))
        self.assertIsNone(s.contact(geometry([[.4,0.,0.]]),[0,-1,-1],[3,1,1]))

    def test_fov_and_range_are_shared_for_all_representations(self):
        z=self.empty();z[180,0]=1.;z[180,320]=.49;z[180,321]=4.01;z[180,322]=1.
        g=s.depth_to_geometry(z)
        self.assertEqual(len(g['raw_points']),1)
        np.testing.assert_array_equal(g['raw_points'],g['fitted_points'])


if __name__=='__main__':unittest.main()
