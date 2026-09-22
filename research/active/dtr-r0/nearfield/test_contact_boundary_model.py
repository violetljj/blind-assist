"""Synthetic engineering checks; these are not predictive-quality evidence."""
import unittest
import torch
from torch.nn import functional as F

from contact_boundary_model import ContactBoundaryModel


class ContactBoundaryModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(19)
        self.features = torch.randn(2, 4864, generator=generator)

    def test_geometry_monotone_in_both_query_axes(self):
        features = self.features
        model = ContactBoundaryModel("geometry").eval()
        widths = torch.tensor([0.01, 0.059, 0.06, 0.061, 0.32, 0.60, 1.2])
        horizons = torch.tensor([0.3, 0.31, 0.399, 0.4, 0.401, 1.5, 3.0])
        for layer in (0, 1):
            queries = torch.cartesian_prod(widths, horizons, torch.tensor([float(layer)]))
            probabilities = model(features, queries).sigmoid().reshape(2, len(widths), len(horizons))
            assert torch.all(probabilities.diff(dim=1) >= -1e-7)
            assert torch.all(probabilities.diff(dim=2) >= -1e-7)
            assert torch.all(probabilities[:, :, 0] < 1e-30)
            assert torch.all(probabilities[:, -1, -1] > probabilities[:, 0, -1])


    def test_partial_cells_are_continuous_and_resolve_subcell_queries(self):
        features = self.features
        model = ContactBoundaryModel("geometry").double()
        # Constant latent mass allows an analytical area-ratio check independently
        # of learned weights: the second sweep has twice the width and same depth.
        with torch.no_grad():
            model.geometry_head.weight.zero_()
            model.geometry_head.bias.fill_(-7)
        queries = torch.tensor([[0.015, 0.35, 0], [0.03, 0.35, 0],
                                [0.06 - 1e-8, 0.40 - 1e-8, 0],
                                [0.06 + 1e-8, 0.40 + 1e-8, 0]], dtype=torch.float64)
        probabilities = model(features.double(), queries).sigmoid()
        mass = -torch.log1p(-probabilities)
        torch.testing.assert_close(mass[:, 1], 2 * mass[:, 0], rtol=1e-6, atol=1e-12)
        assert torch.all((probabilities[:, 3] - probabilities[:, 2]).abs() < 1e-8)
        assert torch.all(probabilities[:, 3] > probabilities[:, 2])


    def test_body_and_head_are_independent(self):
        features = self.features
        model = ContactBoundaryModel("geometry")
        queries = torch.tensor([[0.6, 2.0, 0], [0.6, 2.0, 1]])
        before = model(features, queries).detach()
        with torch.no_grad():
            model.geometry_head.bias[600:].add_(2.0)
        after = model(features, queries).detach()
        torch.testing.assert_close(before[:, 0], after[:, 0], rtol=0, atol=0)
        assert torch.all(after[:, 1] > before[:, 1])


    def check_query_permutation_finite_gradients_and_actual_unused_head_bypass(self, mode):
        features = self.features
        model = ContactBoundaryModel(mode)
        queries = torch.tensor([[0.6, 1.5, 0], [0.2, 0.3, 1], [1.2, 3.0, 1]])
        permutation = torch.tensor([2, 0, 1])
        logits = model(features, queries)
        assert logits.shape == (2, 3)
        assert torch.isfinite(logits).all()
        torch.testing.assert_close(model(features, queries[permutation]), logits[:, permutation])
        F.binary_cross_entropy_with_logits(logits, torch.tensor([[1., 0., 1.], [0., 1., 0.]])).backward()
        active = model.direct_head if mode == "direct" else model.geometry_head
        unused = model.geometry_head if mode == "direct" else model.direct_head
        for parameter in list(model.trunk.parameters()) + list(active.parameters()):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert any(torch.count_nonzero(p.grad) > 0 for p in active.parameters())
        assert all(p.grad is None for p in unused.parameters())
        # A bypassed head must not execute, even when its output is unusable.
        with torch.no_grad():
            for parameter in unused.parameters():
                parameter.fill_(float("nan"))
        torch.testing.assert_close(model(features, queries), logits.detach())


    def test_direct_gradients_permutation_and_bypass(self):
        self.check_query_permutation_finite_gradients_and_actual_unused_head_bypass("direct")


    def test_geometry_gradients_permutation_and_bypass(self):
        self.check_query_permutation_finite_gradients_and_actual_unused_head_bypass("geometry")


    def test_empty_geometry_sweep_has_clean_zero_feature_gradient(self):
        features = self.features
        model = ContactBoundaryModel("geometry")
        observations = features.clone().requires_grad_()
        logits = model(observations, torch.tensor([[1.2, 0.3, 0], [0.6, 0.3, 1]]))
        F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
        assert torch.isfinite(logits).all()
        assert torch.isfinite(observations.grad).all()
        assert torch.count_nonzero(observations.grad) == 0


    def test_identical_carrier_state_and_shared_trunk(self):
        features = self.features
        direct = ContactBoundaryModel("direct")
        geometry = ContactBoundaryModel("geometry")
        geometry.load_state_dict(direct.state_dict(), strict=True)
        assert direct.state_dict().keys() == geometry.state_dict().keys()
        torch.testing.assert_close(direct.encode_scene(features), geometry.encode_scene(features), rtol=0, atol=0)
        counts = direct.parameter_counts(), geometry.parameter_counts()
        assert counts[0]["carrier"] == counts[1]["carrier"]
        assert counts[0]["effective"] != counts[1]["effective"]
        assert counts[0]["effective"] + counts[0]["unused"] == counts[0]["carrier"]


    def test_invalid_queries_are_rejected(self):
        queries = [[0., 1., 0.], [1.21, 1., 0.], [0.6, 0.29, 0.],
                   [0.6, 3.1, 0.], [0.6, 1., 0.5], [0.6, 1., 2.],
                   [0.6, float("nan"), 0.]]
        model = ContactBoundaryModel("geometry")
        for query in queries:
            with self.subTest(query=query), self.assertRaises(ValueError):
                model(self.features, torch.tensor([query]))


if __name__ == "__main__":
    unittest.main()
