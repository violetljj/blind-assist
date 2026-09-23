"""Independent mathematical and interface checks for the censored contact head."""
import copy
import unittest

import torch
from torch.nn import functional as F

from metric_contact_model import MetricContactModel


class MetricContactTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(931)
        torch.set_num_threads(2)
        self.model = MetricContactModel().double()
        self.x = torch.randn(2, 4864, dtype=torch.float64)
        self.queries = torch.tensor([[.6, .3, 0], [.6, 1.2, 0],
                                     [.6, 3., 0], [.8, 2., 1]], dtype=torch.float64)

    def test_normalized_cdf_and_endpoint(self):
        q, mu, scale = self.model.components(self.x, self.queries)
        expected = q * torch.sigmoid((self.queries[:, 1] - mu) / scale) / torch.sigmoid((3 - mu) / scale)
        actual = self.model(self.x, self.queries).sigmoid()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual[:, 2], q[:, 2])
        self.assertTrue(bool(((mu > .3) & (mu < 3.) & (scale > .01)).all()))
        self.assertTrue(bool((actual[:, 0] > 0).all()))  # left-censored mass remains possible

    def test_horizon_monotonic_and_components_horizon_independent(self):
        queries = torch.stack((torch.full((101,), .6), torch.linspace(.3, 3, 101),
                               torch.zeros(101)), dim=1).double()
        probability = self.model(self.x, queries).sigmoid()
        self.assertTrue(bool((probability.diff(dim=1) >= -1e-12).all()))
        for value in self.model.components(self.x, queries):
            torch.testing.assert_close(value, value[:, :1].expand_as(value))

    def test_shared_and_per_image_forward_and_gradients(self):
        other = copy.deepcopy(self.model)
        per_image = self.queries[None].expand(2, -1, -1).clone()
        left = self.model(self.x, self.queries)
        right = other(self.x, per_image)
        torch.testing.assert_close(left, right)
        labels = torch.tensor([[0., 1., 1., 0.], [0., 0., 1., 1.]], dtype=torch.float64)
        for value in (left, right):
            F.binary_cross_entropy_with_logits(value, labels).backward()
        for a, b in zip(self.model.parameters(), other.parameters()):
            self.assertTrue(bool(torch.isfinite(a.grad).all()))
            torch.testing.assert_close(a.grad, b.grad)

    def test_per_image_distinct_queries_match_individual_calls(self):
        queries = self.queries[None].repeat(2, 1, 1)
        queries[1, :, 0] = .9
        actual = self.model(self.x, queries)
        expected = torch.cat([self.model(self.x[i:i+1], queries[i]) for i in range(2)])
        torch.testing.assert_close(actual, expected)

    def test_analytic_crossing_and_right_censoring(self):
        q, _, _ = self.model.components(self.x, self.queries)
        threshold = .3
        crossing = self.model.continuous_crossing(self.x, self.queries, threshold)
        query = self.queries[None].repeat(2, 1, 1)
        query[..., 1] = crossing
        score = self.model(self.x, query).sigmoid()
        interior = (crossing > .3) & (crossing < 3)
        self.assertTrue(bool(interior.any()))
        torch.testing.assert_close(score[interior], torch.full_like(score[interior], threshold))
        censored = self.model.continuous_crossing(self.x, self.queries, float(q.detach().max()) + .01)
        self.assertTrue(bool(torch.isinf(censored).all()))

    def test_endpoint_equal_threshold_and_left_censoring(self):
        # Fixed head q=.5, mu=1.65, scale=.01+log(2).
        with torch.no_grad():
            for parameter in self.model.head.parameters():
                parameter.zero_()
        end = self.model.continuous_crossing(self.x, self.queries, .5)
        torch.testing.assert_close(end, torch.full_like(end, 3.))
        start = self.model.continuous_crossing(self.x, self.queries, .001)
        torch.testing.assert_close(start, torch.full_like(start, .3))

    def test_invalid_queries_rejected(self):
        invalid = [torch.zeros(4), torch.zeros(3, 4, 3),
                   torch.tensor([[.6, .29, 0.]]), torch.tensor([[1.3, 1., 0.]]),
                   torch.tensor([[.6, 1., 2.]]), torch.tensor([[.6, float('nan'), 0.]])]
        for query in invalid:
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.model(self.x, query)


if __name__ == '__main__':
    unittest.main()
