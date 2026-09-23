"""Synthetic adapter equivalence checks; no experiment fitting or source data."""
import unittest

import torch

from contact_boundary_model import ContactBoundaryModel
from contact_boundary_sampling_model import ContactBoundarySamplingModel, operator_counterexample


class SamplingModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(20260923)
        self.features = torch.randn(3, 4864, dtype=torch.float64)
        self.queries = torch.tensor([[[.4, .8, 0], [.7, 1.35, 1], [1.1, 2.8, 0]],
                                     [[.6, .3, 1], [.3, .65, 0], [.8, 1.75, 1]],
                                     [[.9, 2.1, 1], [.5, 1.1, 0], [.2, .55, 1]]], dtype=torch.float64)

    def models(self, mode):
        original = ContactBoundaryModel(mode).double()
        adapter = ContactBoundarySamplingModel(mode).double()
        adapter.load_state_dict(original.state_dict(), strict=True)
        return original, adapter

    def test_identical_state_keys_and_counts(self):
        for mode in ("direct", "geometry"):
            original, adapter = self.models(mode)
            self.assertEqual(list(original.state_dict()), list(adapter.state_dict()))
            self.assertEqual(original.parameter_counts(), adapter.parameter_counts())

    def test_logits_and_gradients_match_original_per_image_reference(self):
        for mode in ("direct", "geometry"):
            with self.subTest(mode=mode):
                original, adapter = self.models(mode)
                x1 = self.features.clone().requires_grad_()
                x2 = self.features.clone().requires_grad_()
                q1 = self.queries.clone().requires_grad_()
                q2 = self.queries.clone().requires_grad_()
                expected = torch.cat([original(x1[i:i+1], q1[i]) for i in range(3)])
                actual = adapter(x2, q2)
                torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
                weights = torch.tensor([[1., 2., 3.], [2., 1., 4.], [3., 2., 1.]])
                (expected * weights).sum().backward()
                (actual * weights).sum().backward()
                torch.testing.assert_close(x2.grad, x1.grad, rtol=1e-10, atol=1e-12)
                torch.testing.assert_close(q2.grad, q1.grad, rtol=1e-10, atol=1e-12)
                for (name1, p1), (name2, p2) in zip(original.named_parameters(), adapter.named_parameters()):
                    self.assertEqual(name1, name2)
                    self.assertEqual(p1.grad is None, p2.grad is None)
                    if p1.grad is not None:
                        self.assertTrue(torch.isfinite(p2.grad).all())
                        torch.testing.assert_close(p2.grad, p1.grad, rtol=1e-10, atol=1e-11)

    def test_shared_query_super_path_is_exact_and_per_image_replication_matches(self):
        for mode in ("direct", "geometry"):
            original, adapter = self.models(mode)
            shared = self.queries[0]
            expected = original(self.features, shared)
            torch.testing.assert_close(adapter(self.features, shared), expected, rtol=0, atol=0)
            replicated = adapter(self.features, shared.expand(3, -1, -1))
            torch.testing.assert_close(replicated, expected, rtol=1e-12, atol=1e-12)

    def test_batch_and_per_image_query_permutations_are_equivariant(self):
        for mode in ("direct", "geometry"):
            _, model = self.models(mode)
            expected = model(self.features, self.queries)
            batch = torch.tensor([2, 0, 1])
            torch.testing.assert_close(model(self.features[batch], self.queries[batch]), expected[batch])
            permutations = torch.tensor([[2, 0, 1], [1, 2, 0], [0, 2, 1]])
            permuted = self.queries.gather(1, permutations[:, :, None].expand(-1, -1, 3))
            torch.testing.assert_close(model(self.features, permuted), expected.gather(1, permutations))

    def test_query_shape_and_domain_rejections(self):
        model = ContactBoundarySamplingModel("geometry").double()
        for queries in (self.queries[:2], self.queries[..., :2], self.queries.unsqueeze(0)):
            with self.assertRaises(ValueError):
                model(self.features, queries)
        bad = self.queries.clone()
        bad[1, 1, 2] = .5
        with self.assertRaises(ValueError):
            model(self.features, bad)

    def test_operator_counterexample_is_full_row_rank_but_novel_decisions_differ(self):
        proof = operator_counterexample()
        self.assertEqual(proof["shape"], [72, 1200])
        self.assertEqual(proof["rank"], 72)
        self.assertEqual(proof["nominal_decimal_operator_rank"], 72)
        self.assertLess(proof["nominal_decimal_max_mass_difference"], 1e-12)
        self.assertLess(proof["original_max_probability_difference"], 1e-5)
        self.assertEqual(proof["original_threshold_decision_disagreements"], 0)
        self.assertEqual(proof["novel_decisions"], [True, False])
        self.assertGreater(proof["novel_probability_difference"], .9)


if __name__ == "__main__":
    unittest.main()
