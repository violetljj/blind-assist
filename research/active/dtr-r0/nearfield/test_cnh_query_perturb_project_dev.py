"""Three focused contracts, including NR-cache and original GPU projector parity."""
import unittest
import numpy as np
import torch
import cnh_query_perturb_project_dev as P


class QueryPerturbationProjection(unittest.TestCase):
    def test_identity_and_signed_query_rotation(self):
        query=np.eye(4);query[:3,3]=[.2,.1,.4]
        np.testing.assert_array_equal(P.perturb_query(query,0.),query)
        transform=P.perturb_query(np.eye(4),10.)
        point=transform@np.array([0.,0.,1.,1.])
        self.assertLess(point[0],0.)
        self.assertAlmostEqual(np.degrees(np.arctan2(point[0],point[2])),-10.)
        m=np.broadcast_to(query,(2,4,4,4)).copy()
        actual=P.perturb_matrices(m,[0.,10.])
        np.testing.assert_array_equal(actual[0],m[0])
        np.testing.assert_allclose(actual[1],np.broadcast_to(P.perturb_query(query,10.),(4,4,4)),atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for original projection parity')
    def test_nr_cache_and_original_sequence_bitwise(self):
        from cnh_cvr_v2_materialize import BatchedProjector
        projector=BatchedProjector();helper=P.QueryPerturbProjector(projector)
        try:
            for frame in (3,10):
                z,m,cache=P.nr_fixture(frame)
                actual=helper.project(z,m).cpu().numpy()
                reference=np.stack([projector.sequence(row,t).half().cpu().numpy() for row,t in zip(z,m)])
                np.testing.assert_array_equal(actual,reference)
                np.testing.assert_array_equal(actual,cache)
        finally:
            helper=None;projector=None;torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for perturbed projection parity')
    def test_perturbed_histories_match_original_sequence(self):
        from cnh_cvr_v2_materialize import BatchedProjector
        projector=BatchedProjector();helper=P.QueryPerturbProjector(projector)
        try:
            z,m,_=P.nr_fixture(10);changed=P.perturb_matrices(m,[10.,-10.])
            actual=helper.project(z,changed).cpu().numpy()
            expected=np.stack([projector.sequence(row,t).half().cpu().numpy() for row,t in zip(z,changed)])
            np.testing.assert_array_equal(actual,expected)
        finally:
            helper=None;projector=None;torch.cuda.empty_cache()


if __name__ == '__main__':unittest.main()
