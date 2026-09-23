"""Analytic geometry fixtures independent of the first-contact label helper."""
import unittest

import numpy as np

from exact_contact_labels import metric_targets


def box(xlo, xhi, ylo, yhi, zlo, zhi):
    return (np.array([xlo, ylo, zlo]), np.array([xhi, yhi, zhi]))


class ExactContactLabelsTests(unittest.TestCase):
    def test_arbitrary_horizon_independence(self):
        queries = [[.6, h, 0] for h in (.3, .6, 1.8, 3.)]
        np.testing.assert_allclose(metric_targets([box(-.1,.1,.5,.7,2.,2.2)], queries), [2.]*4)

    def test_multiple_surfaces_and_layer_selection(self):
        boxes = [box(-.1,.1,.5,.7,2.,2.2), box(-.1,.1,.5,.7,1.,1.2),
                 box(-.1,.1,0.,.2,.7,.8)]
        np.testing.assert_allclose(metric_targets(boxes, [[.6,3.,0],[.6,3.,1]]), [1.,.7])

    def test_no_contact_and_out_of_range(self):
        query = [[.6,3.,0]]
        for boxes in ([], [box(.4,.5,.5,.7,1.,2.)], [box(-.1,.1,1.,1.2,1.,2.)],
                      [box(-.1,.1,.5,.7,3.001,4.)], [box(-.1,.1,.5,.7,.1,.299)]):
            with self.subTest(boxes=boxes):
                self.assertTrue(np.isinf(metric_targets(boxes,query)[0]))

    def test_left_censoring_and_start_touch(self):
        for lo,hi in ((.1,.5),(.3,.5),(.1,.3)):
            np.testing.assert_allclose(metric_targets([box(-.1,.1,.5,.7,lo,hi)], [[.6,1.,0]]), [.3])

    def test_exact_endpoint_is_contact(self):
        np.testing.assert_allclose(metric_targets([box(-.1,.1,.5,.7,3.,3.5)], [[.6,.6,0]]), [3.])

    def test_width_grazing_and_exclusion(self):
        boxes=[box(.3,.5,.5,.7,1.,1.5)]
        target=metric_targets(boxes,[[.6,3.,0],[.599,3.,0],[1.,3.,0]])
        np.testing.assert_allclose(target[[0,2]],[1.,1.])
        self.assertTrue(np.isinf(target[1]))

    def test_shared_band_edge_is_inclusive(self):
        boxes=[box(-.1,.1,.42,.42,1.,1.5)]
        np.testing.assert_allclose(metric_targets(boxes,[[.6,3.,0],[.6,3.,1]]),[1.,1.])

    def test_does_not_mutate_queries(self):
        query=np.array([[.6,.6,0]],dtype=np.float32); before=query.copy()
        metric_targets([],query)
        np.testing.assert_array_equal(query,before)

    def test_invalid_queries_rejected(self):
        for query in ([[.6,3.1,0]],[[.6,.2,0]],[[.6,1.,2]],[[0.,1.,0]],[[float('nan'),1.,0]],[[.6,1.]]):
            with self.subTest(query=query),self.assertRaises(ValueError): metric_targets([],query)


if __name__=='__main__': unittest.main()
