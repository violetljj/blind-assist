"""Focused NF-G8 architecture and protected-label boundary checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from whisker_model import ARMS, IMAGE_SIZE, WhiskerModel, masked_bce, support_bce, reichardt, repeat_history
from train_whisker import load_inputs, input_hashes


class WhiskerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shapes_and_current_branch_independence(self):
        torch.manual_seed(7)
        rgb = torch.rand(2, 3, 3, *IMAGE_SIZE)
        with torch.inference_mode():
            for arm in ARMS:
                model = WhiskerModel(arm).eval()
                logits, support = model(rgb)
                repeated_logits, repeated_support = model(repeat_history(rgb))
                self.assertEqual(tuple(logits.shape), (2, 4))
                self.assertEqual(tuple(support.shape), (2, 2, 18, 32))
                torch.testing.assert_close(logits[:, :2], repeated_logits[:, :2], rtol=0, atol=0)
                torch.testing.assert_close(support, repeated_support, rtol=0, atol=0)
                if arm == "single_frame":
                    torch.testing.assert_close(logits, repeated_logits, rtol=0, atol=0)
                else:
                    self.assertGreater(float((logits[:, 2:] - repeated_logits[:, 2:]).abs().max()), 1e-7)

    def test_reichardt_static_zero_and_time_reversal(self):
        rgb = torch.rand(1, 3, 3, 12, 20)
        static = reichardt(repeat_history(rgb))
        self.assertEqual(float(static.abs().max()), 0.)
        original = reichardt(rgb).reshape(1, 4, 2, 12, 20)
        reverse = reichardt(rgb.flip(1)).reshape(1, 4, 2, 12, 20)
        torch.testing.assert_close(reverse, -original.flip(2))
        self.assertGreater(float(original.abs().max()), 0.)

    def test_unknown_has_zero_gradient(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        masked_bce(logits, torch.tensor([[1., -1., 0., -1.]])).backward()
        self.assertEqual(logits.grad[0, 1].item(), 0.)
        self.assertEqual(logits.grad[0, 3].item(), 0.)
        self.assertLess(logits.grad[0, 0].item(), 0.)
        self.assertGreater(logits.grad[0, 2].item(), 0.)
        support = torch.zeros(1, 2, 18, 32, requires_grad=True)
        support_bce(support, torch.zeros_like(support), torch.tensor([[-1., 0.]])).backward()
        self.assertEqual(float(support.grad[:, 0].abs().max()), 0.)
        self.assertGreater(float(support.grad[:, 1].abs().max()), 0.)

    def test_sparse_support_balances_classes_and_handles_empty_classes(self):
        logits = torch.zeros(1, 2, 18, 32, requires_grad=True)
        target = torch.zeros_like(logits)
        target[0, 0, 0, 0] = 1.
        near = torch.tensor([[1., -1.]])
        support_bce(logits, target, near).backward()
        self.assertAlmostEqual(logits.grad[0, 0, 0, 0].item(), -.25)
        self.assertAlmostEqual(logits.grad[0, 0].sum().item(), 0., places=6)
        self.assertEqual(float(logits.grad[0, 1].abs().max()), 0.)
        for value in (0., 1.):
            single_class = torch.zeros(1, 2, 18, 32, requires_grad=True)
            loss = support_bce(single_class, torch.full_like(single_class, value), near)
            loss.backward()
            self.assertAlmostEqual(loss.item(), np.log(2.), places=6)
            self.assertAlmostEqual(single_class.grad.abs().sum().item(), .5, places=6)
        unknown = torch.zeros(1, 2, 18, 32, requires_grad=True)
        loss = support_bce(unknown, target, torch.tensor([[-1., -1.]]))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(float(unknown.grad.abs().max()), 0.)

    def fixture(self, root):
        (root / "model").mkdir()
        (root / "training").mkdir()
        samples, frames = [], []
        for k, split in enumerate(("train", "val", "test")):
            ids = list(range(k*3, k*3+3))
            samples.append(dict(sample_id=split, clip_id=split, group_id=split, split=split, frame_indices=ids))
            frames.extend(dict(sample_index=i, rgb_path=f"{i}.png", clip_id=split, frame_in_clip=j, time_s=j*.2)
                          for j, i in enumerate(ids))
        dataset = dict(calibration=dict(width=640, height=360, horizontal_fov_degrees=100), samples=samples, frames=frames)
        (root / "model" / "dataset.json").write_text(json.dumps(dataset))
        labels = dict(targets={split: [1, 0, -1, 0] for split in ("train", "val")})
        (root / "training" / "labels.json").write_text(json.dumps(labels))
        np.savez(root / "training" / "support.npz", **{split: np.zeros((2, 18, 32), np.uint8) for split in ("train", "val")})
        return dataset, labels

    def test_loader_accepts_unknown_without_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            dataset, labels, support, paths, inputs = load_inputs(root)
            self.assertEqual(labels["train"][2], -1)
            self.assertNotIn("test", support)
            self.assertEqual(len(paths), 9)

    def test_loader_rejects_test_labels_and_group_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, labels = self.fixture(root)
            labels["targets"]["test"] = [0]*4
            (root / "training" / "labels.json").write_text(json.dumps(labels))
            with self.assertRaisesRegex(ValueError, "test labels prohibited"):
                load_inputs(root)
            del labels["targets"]["test"]
            (root / "training" / "labels.json").write_text(json.dumps(labels))
            dataset["samples"][2]["group_id"] = "train"
            (root / "model" / "dataset.json").write_text(json.dumps(dataset))
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                load_inputs(root)

    def test_loader_rejects_speed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, _ = self.fixture(root)
            dataset["frames"][0]["speed_m_s"] = 1.5
            (root / "model" / "dataset.json").write_text(json.dumps(dataset))
            with self.assertRaisesRegex(ValueError, "extra metadata prohibited"):
                load_inputs(root)

    def test_input_hashes_normalizes_root_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            path = root / "model" / "dataset.json"
            path.write_text("{}")
            # Mock the junction alias's resolution without requiring OS privileges.
            from unittest.mock import Mock
            alias = Mock()
            alias.resolve.return_value = root.resolve()
            result = input_hashes(alias, [path.resolve()])
            self.assertEqual(list(result), ["model/dataset.json"])
            self.assertEqual(len(result["model/dataset.json"]), 64)


if __name__ == "__main__":
    unittest.main()
