from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.research.riskseg_r0_pidnet_preflight.modeling import (
    DeploymentWrapper,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    build_pidnet_s,
    load_imagenet_backbone,
    load_trained_deployment_checkpoint,
    set_deterministic_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REPO = REPO_ROOT / "artifacts.local" / "vendor" / "pidnet-official"
PRETRAINED = (
    REPO_ROOT
    / "artifacts.local"
    / "models"
    / "riskseg-r0"
    / "pretrained"
    / "PIDNet_S_ImageNet.pth.tar"
)
TRAINED_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local"
    / "evidence"
    / "riskseg-r0"
    / "training-v1"
    / "seed-20260801"
    / "best_checkpoint.pt"
)


class PreflightModelTest(unittest.TestCase):
    def test_fixed_four_class_surface_and_pretrained_match(self) -> None:
        if not OFFICIAL_REPO.is_dir() or not PRETRAINED.is_file():
            self.skipTest("local ignored official source/weight artifacts unavailable")
        set_deterministic_seed(20260801)
        model = build_pidnet_s(official_repo=OFFICIAL_REPO, augment=False)
        report = load_imagenet_backbone(model=model, checkpoint_path=PRETRAINED)
        self.assertEqual(report["matched_tensor_count"], 302)
        self.assertEqual(report["matched_parameter_count"], 6_456_274)
        output = DeploymentWrapper(model).eval()(
            torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
        )
        self.assertEqual(tuple(output.shape), (1, 4, INPUT_HEIGHT, INPUT_WIDTH))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_deterministic_initialization(self) -> None:
        if not OFFICIAL_REPO.is_dir() or not PRETRAINED.is_file():
            self.skipTest("local ignored official source/weight artifacts unavailable")
        states = []
        for _ in range(2):
            set_deterministic_seed(20260801)
            model = build_pidnet_s(official_repo=OFFICIAL_REPO, augment=False)
            load_imagenet_backbone(model=model, checkpoint_path=PRETRAINED)
            states.append({key: value.clone() for key, value in model.state_dict().items()})
        self.assertEqual(states[0].keys(), states[1].keys())
        for key in states[0]:
            self.assertTrue(torch.equal(states[0][key], states[1][key]), key)

    def test_training_checkpoint_populates_exact_deployment_subset(self) -> None:
        if not OFFICIAL_REPO.is_dir() or not TRAINED_CHECKPOINT.is_file():
            self.skipTest("local ignored official source/training artifact unavailable")
        model = build_pidnet_s(official_repo=OFFICIAL_REPO, augment=False)
        report = load_trained_deployment_checkpoint(
            model=model,
            checkpoint_path=TRAINED_CHECKPOINT,
        )
        self.assertEqual(report["seed"], 20260801)
        self.assertEqual(report["epoch"], 16)
        self.assertEqual(report["deployment_tensor_count"], 453)
        self.assertEqual(report["auxiliary_tensor_count"], 26)
        self.assertTrue(
            all(
                key.startswith(("seghead_d.", "seghead_p."))
                for key in report["auxiliary_keys"]
            )
        )
        output = DeploymentWrapper(model).eval()(
            torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
        )
        self.assertEqual(tuple(output.shape), (1, 4, INPUT_HEIGHT, INPUT_WIDTH))
        self.assertTrue(bool(torch.isfinite(output).all()))


if __name__ == "__main__":
    unittest.main()
