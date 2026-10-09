"""Independent CPU geometry and focused signed-boundary token checks."""
import unittest

import numpy as np
import torch

from cnh_boundary_token_model import (BoundaryTokenReadout, QUERY_BOXES, SOFTNESS,
                                      build_features, feature_geometry, parameter_count)
from cnh_cvr_projection import EDGE, WIDTH


def transforms(batch=1):
    return torch.eye(4).repeat(batch, 8, 1, 1)


def reference(t, mode):
    result = np.empty((2, 8, 8, 16, 11), np.float64)
    offsets = [.5] if mode == 'center' else [1/6, .5, 5/6]
    for y in range(8):
        for x in range(8):
            rays = []
            for dy in offsets:
                for dx in offsets:
                    ray = np.array([-EDGE+(x+dx)*2*EDGE/8,
                                    -EDGE+(y+dy)*2*EDGE/8, 1.])
                    rays.append(ray/np.linalg.norm(ray))
            xyz = np.asarray(rays)[None]*((np.arange(16)+.5)*WIDTH)[:, None, None]
            xyz = xyz@t[:3, :3].T+t[:3, 3]
            for q, box in enumerate(QUERY_BOXES):
                margins = np.stack((xyz[:, :, 0]-box[0], box[3]-xyz[:, :, 0],
                                    xyz[:, :, 1]-box[1], box[4]-xyz[:, :, 1],
                                    xyz[:, :, 2]-box[2], box[5]-xyz[:, :, 2]), -1)
                outside = np.maximum(-margins, 0).sum(-1)
                membership = np.exp(-outside/SOFTNESS)
                result[q, y, x] = np.concatenate((np.stack((membership.mean(-1), membership.min(-1), membership.max(-1)), -1),
                    margins.min(-2)/np.array([.6, .6, 1., 1., 3., 3.]),
                    np.stack((outside.min(-1), outside.max(-1)), -1)/.6), -1)
    return result


class BoundaryTokenTests(unittest.TestCase):
    def test_independent_geometry(self):
        t = transforms()
        a = np.deg2rad(11.)
        t[:, :, :3, :3] = torch.tensor([[np.cos(a), 0., np.sin(a)],
                                        [0., 1., 0.], [-np.sin(a), 0., np.cos(a)]])
        t[:, :, :3, 3] = torch.tensor([.03, -.04, .08])
        for mode in ('center', 'extent'):
            actual = feature_geometry(t, [8], mode)[0, :, -1].numpy()
            np.testing.assert_allclose(actual, reference(t[0, -1].numpy(), mode), atol=2e-6, rtol=2e-5)

    def test_side_specific_signed_margins(self):
        # Both translations keep a selected ray inside the query. Membership
        # is saturated, while signed x-low/x-high margins distinguish sides.
        left, right = transforms(), transforms()
        left[:, :, 0, 3], right[:, :, 0, 3] = -.08, .08
        l = feature_geometry(left, [8], 'center')[0, 0, -1, 4, 4, 3]
        r = feature_geometry(right, [8], 'center')[0, 0, -1, 4, 4, 3]
        torch.testing.assert_close(l[:3], r[:3], rtol=0, atol=0)
        assert l[0] == 1
        assert l[3] < r[3] and l[4] > r[4]
        # Extent changes the direction-specific envelope even inside the box.
        extent = feature_geometry(left, [8], 'extent')[0, 0, -1, 4, 4, 3]
        assert extent[3] < l[3] and extent[4] < l[4]

    def test_nan_padding_and_channels(self):
        torch.manual_seed(7)
        z, t, lengths = torch.randn(2, 8, 8, 8, 16), transforms(2), torch.tensor([2, 5])
        expected_geometry = feature_geometry(t, lengths, 'extent')
        expected = build_features(z, expected_geometry, lengths)
        for i, length in enumerate(lengths):
            z[i, :8-length], t[i, :8-length] = float('nan'), float('nan')
        g = feature_geometry(t, lengths, 'extent')
        torch.testing.assert_close(g, expected_geometry, rtol=0, atol=0)
        for i, length in enumerate(lengths):
            g[i, :, :8-length] = float('nan')
        actual = build_features(z, g, lengths)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == (2, 2, 16, 15, 8, 8)
        signed = z[:, -1].sign()*z[:, -1].abs().log1p()
        torch.testing.assert_close(actual[:, 0, :, 0], signed.permute(0, 3, 1, 2))
        radius = (torch.arange(16)+.5)*WIDTH/3
        torch.testing.assert_close(actual[0, 0, :, 14, 0, 0], radius)

    def test_same_state_and_gradient(self):
        torch.manual_seed(8)
        a, b = BoundaryTokenReadout(), BoundaryTokenReadout()
        b.load_state_dict(a.state_dict())
        assert parameter_count() == 8833
        z, t, lengths = torch.randn(2, 8, 8, 8, 16), transforms(2), torch.tensor([1, 8])
        for mode in ('center', 'extent'):
            features = build_features(z, feature_geometry(t, lengths, mode), lengths)
            torch.testing.assert_close(a(features, lengths), b(features, lengths), rtol=0, atol=0)
        features.requires_grad_()
        output = a(features, lengths)
        assert output.shape == (2, 2)
        output.square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())
        assert features.grad[:, :, :, 6:12].abs().sum() > 0
        torch.testing.assert_close(a(features.half(), lengths), a(features, lengths), rtol=.02, atol=.001)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_cpu(self):
        torch.manual_seed(9)
        z, t, lengths = torch.randn(2, 8, 8, 8, 16), transforms(2), torch.tensor([3, 8])
        for mode in ('center', 'extent'):
            cpu, gpu = feature_geometry(t, lengths, mode), feature_geometry(t.cuda(), lengths.cuda(), mode)
            torch.testing.assert_close(gpu.cpu(), cpu, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(build_features(z.cuda(), gpu, lengths.cuda()).cpu(),
                                       build_features(z, cpu, lengths), rtol=2e-5, atol=2e-6)
        model = BoundaryTokenReadout().cuda()
        model(build_features(z.cuda(), gpu, lengths.cuda()).half(), lengths.cuda()).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


if __name__ == '__main__':
    unittest.main()
