"""Focused query geometry, padding and same-network contract checks."""
import numpy as np
import torch
import unittest

from cnh_delayed_query_model import (DelayedQueryReadout, QUERY_BOXES, SOFTNESS,
                                     build_features, feature_geometry, parameter_count)
from cnh_cvr_projection import EDGE, WIDTH


def transforms(batch=1):
    return torch.eye(4).repeat(batch, 8, 1, 1)


def numpy_reference(t, mode):
    """Independent CPU scalar loops over zones, nodes and fixed query boxes."""
    result = np.empty((2, 8, 8, 16, 3), np.float64)
    offsets = [.5] if mode == 'center' else [1/6, .5, 5/6]
    for y in range(8):
        for x in range(8):
            rays = []
            for dy in offsets:
                for dx in offsets:
                    ray = np.array([-EDGE+(x+dx)*2*EDGE/8,
                                    -EDGE+(y+dy)*2*EDGE/8, 1.])
                    rays.append(ray/np.linalg.norm(ray))
            xyz = np.asarray(rays)[None] * ((np.arange(16)+.5)*WIDTH)[:, None, None]
            xyz = xyz @ t[:3, :3].T + t[:3, 3]
            for q, box in enumerate(QUERY_BOXES):
                box = np.asarray(box)
                outside = np.maximum(box[:3]-xyz, 0) + np.maximum(xyz-box[3:], 0)
                values = np.exp(-outside.sum(-1)/SOFTNESS)
                result[q, y, x] = np.stack((values.mean(-1), values.min(-1), values.max(-1)), -1)
    return result


def _check_geometry_independent_cpu_reference(mode):
    t = transforms()
    angle = np.deg2rad(13.)
    t[:, :, :3, :3] = torch.tensor([[np.cos(angle), 0., np.sin(angle)],
                                    [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    t[:, :, :3, 3] = torch.tensor([.03, -.04, .08])
    actual = feature_geometry(t, torch.tensor([8]), mode)[0, :, -1].numpy()
    np.testing.assert_allclose(actual, numpy_reference(t[0, -1].numpy(), mode), atol=2e-6, rtol=2e-5)
    assert np.all(actual[..., 1] <= actual[..., 0]+1e-7)
    assert np.all(actual[..., 0] <= actual[..., 2]+1e-7)


def _check_known_yaw_moves_right_body_ray_into_query():
    t = transforms()
    before = feature_geometry(t, [8], 'center')[0, 1, -1, 6, 6, 6, 0]
    angle = np.deg2rad(-15.)
    t[:, :, :3, :3] = torch.tensor([[np.cos(angle), 0., np.sin(angle)],
                                    [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    after = feature_geometry(t, [8], 'center')[0, 1, -1, 6, 6, 6, 0]
    assert after > .99
    assert after > before * 20


def _check_nan_padding_ignored_and_radial_evidence_retained():
    torch.manual_seed(1)
    z = torch.randn(2, 8, 8, 8, 16)
    t = transforms(2)
    lengths = torch.tensor([2, 5])
    clean_geometry = feature_geometry(t, lengths, 'extent')
    expected = build_features(z, clean_geometry, lengths)
    for sample, length in enumerate(lengths):
        z[sample, :8-length] = float('nan')
        t[sample, :8-length] = float('nan')
    geometry = feature_geometry(t, lengths, 'extent')
    torch.testing.assert_close(geometry, clean_geometry, rtol=0, atol=0)
    for sample, length in enumerate(lengths):
        geometry[sample, :, :8-length] = float('nan')
    actual = build_features(z, geometry, lengths)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 2, 128, 8, 8)
    # Native current remains one channel per radial bin, without bin averaging.
    signed = z[:, -1].sign()*z[:, -1].abs().log1p()
    torch.testing.assert_close(actual[:, 0, 112:], signed.permute(0, 3, 1, 2))


def _check_same_state_shape_and_finite_gradients():
    torch.manual_seed(2)
    a, b = DelayedQueryReadout(), DelayedQueryReadout()
    b.load_state_dict(a.state_dict())
    assert parameter_count() == 15713
    z = torch.randn(2, 8, 8, 8, 16)
    lengths = torch.tensor([1, 8])
    center = build_features(z, feature_geometry(transforms(2), lengths, 'center'), lengths)
    extent = build_features(z, feature_geometry(transforms(2), lengths, 'extent'), lengths)
    torch.testing.assert_close(a(center, lengths), b(center, lengths), rtol=0, atol=0)
    output = a(extent.half(), lengths)
    assert output.shape == (2, 2)
    output.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
def _check_gpu_matches_cpu_geometry_and_features():
    torch.manual_seed(3)
    z = torch.randn(2, 8, 8, 8, 16)
    lengths = torch.tensor([3, 8])
    t = transforms(2)
    for mode in ('center', 'extent'):
        cpu = feature_geometry(t, lengths, mode)
        gpu = feature_geometry(t.cuda(), lengths.cuda(), mode)
        torch.testing.assert_close(gpu.cpu(), cpu, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(build_features(z.cuda(), gpu, lengths.cuda()).cpu(),
                                   build_features(z, cpu, lengths), rtol=2e-5, atol=2e-6)


class DelayedQueryTests(unittest.TestCase):
    def test_geometry_center(self):
        _check_geometry_independent_cpu_reference('center')

    def test_geometry_extent(self):
        _check_geometry_independent_cpu_reference('extent')

    def test_known_yaw(self):
        _check_known_yaw_moves_right_body_ray_into_query()

    def test_padding_and_radial_channels(self):
        _check_nan_padding_ignored_and_radial_evidence_retained()

    def test_state_and_gradients(self):
        _check_same_state_shape_and_finite_gradients()

    def test_gpu_cpu(self):
        _check_gpu_matches_cpu_geometry_and_features()


if __name__ == '__main__':
    unittest.main()
