"""Focused geometry/causality/output checks, independent of NF-G7 outcomes."""
import copy
import math
import os
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from contact_retina_model import (CHANNELS, HISTORY, IMAGE_SIZE, META_SIZE,
    STRUCTURE_SIZE, ContactRetina, arm_features, cumulative_contact, frame_poses,
    motion_features, prepare_features, project_history, structure_features)


def observations(translation=True):
    result = []
    for index in range(HISTORY):
        camera = dict(x=index*.06 if translation else 0., y=0., z=1.82,
                      pitch=0., yaw=(index-7)*.4, roll=0.)
        wearer = dict(camera, z=.12, yaw=0.)
        result.append(dict(camera_transform=camera, wearer_transform=wearer,
                           speed_m_s=.6, time_s=index*.1))
    return result


class ContactRetinaTests(unittest.TestCase):
    def test_cumulative_hazards_and_gradients(self):
        logits = torch.tensor([[[-3., 0., 2.], [2., -1., -2.]]], requires_grad=True)
        scores = cumulative_contact(logits)
        self.assertTrue(bool(((scores >= 0) & (scores <= 1)).all()))
        self.assertTrue(bool((scores.diff(dim=-1) >= 0).all()))
        scores.sum().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_world_translation_does_not_become_identifier_feature(self):
        first = observations()
        shifted = copy.deepcopy(first)
        for frame in shifted:
            for key in ("camera_transform", "wearer_transform"):
                frame[key]["x"] += 1234.
                frame[key]["y"] -= 567.
                frame[key]["z"] += 2.
        np.testing.assert_allclose(motion_features(first), motion_features(shifted), atol=1e-6)

    def test_metric_lateral_projection(self):
        frames = observations(False)
        for frame in frames:
            frame["camera_transform"]["yaw"] = 0.
            frame["camera_transform"]["y"] = -.2
        frames[-1]["camera_transform"]["y"] = 0.
        r, t = frame_poses(frames, "cpu")
        uv, _ = project_history(r, t, 18, 32, 100., torch.tensor([2.]))
        focal = 32/(2*math.tan(math.radians(50.)))
        self.assertAlmostEqual(float(uv[0, 0, 8, 15, 0])-15., focal*.2/2., places=5)

    def test_pure_rotation_is_depth_invariant(self):
        r, t = frame_poses(observations(False), "cpu")
        uv, valid = project_history(r, t, 18, 32, 100., torch.tensor([.5, 2., 10.]))
        torch.testing.assert_close(uv[:, 0], uv[:, 2], atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.equal(valid[:, 0], valid[:, 2]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA geometry checks require CUDA")
    def test_pure_rotation_has_no_temporal_parallax_maps(self):
        frames = observations(False)
        r, t = frame_poses(frames, "cuda")
        # A shared textured distant world evaluated at the rotated camera rays.
        h, w = STRUCTURE_SIZE
        y, x = torch.meshgrid(torch.linspace(-.6, .6, h, device="cuda"),
                              torch.linspace(-1., 1., w, device="cuda"), indexing="ij")
        rays = torch.stack((x, y, torch.ones_like(x)), -1)
        world = torch.einsum("tij,hwj->thwi", r, rays)
        gray = .5 + .25*torch.sin(12*world[..., 0]) * torch.cos(15*world[..., 1])
        rgb = gray[:, None].repeat(1, 3, 1, 1)
        features = structure_features(rgb, r, t)
        self.assertGreater(float(features[:2].abs().sum()), 0.)
        self.assertEqual(float(features[2:].abs().max()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA history check requires CUDA")
    def test_history_copy_matches_actual_recomputed_features(self):
        torch.manual_seed(123)
        frames = observations()
        rgb = torch.rand((HISTORY, 3, *STRUCTURE_SIZE), device="cuda")
        visual, _, repeat_meta = prepare_features(rgb, frames)
        repeated_frames = copy.deepcopy(frames)
        for frame in repeated_frames:
            for key in ("camera_transform", "wearer_transform"):
                frame[key] = copy.deepcopy(frames[-1][key])
        repeated_rgb = rgb[-1:].repeat(HISTORY, 1, 1, 1)
        actual, actual_meta, _ = prepare_features(repeated_rgb, repeated_frames)
        expected = arm_features(visual[None], "temporal_structure", True)[0]
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(repeat_meta, actual_meta, atol=1e-6, rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA backward check requires CUDA")
    def test_real_backbone_deterministic_backward_shape(self):
        torch.use_deterministic_algorithms(True)
        model = ContactRetina().cuda()
        visual = torch.rand((2, CHANNELS, *IMAGE_SIZE), device="cuda")
        meta = torch.zeros((2, META_SIZE), device="cuda")
        output = model(visual, meta)
        self.assertEqual(tuple(output.shape), (2, 2, 3))
        output.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
