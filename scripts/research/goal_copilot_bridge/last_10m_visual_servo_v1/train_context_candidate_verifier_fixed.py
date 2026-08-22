#!/usr/bin/env python3
"""Train a fixed-epoch context verifier on all consumed cohorts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _require
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


EPOCHS = 5
DECISION_THRESHOLD = 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    _require(not args.output.exists(), "fixed context verifier output already exists")
    receipt = json.loads((args.dataset / "receipt.json").read_text(encoding="utf-8"))
    _require(receipt.get("future_cohort_access") is False and receipt["counts"]["train"]["door"] > 0 and receipt["counts"]["train"]["distractor"] > 0, "fixed context dataset drift")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transform = transforms.Compose([transforms.Resize((224, 448)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(args.dataset / "train", transform=transform)
    _require(dataset.class_to_idx == {"distractor": 0, "door": 1}, "fixed context classes drift")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, generator=torch.Generator().manual_seed(0))
    counts = receipt["counts"]["train"]
    weights = torch.tensor([1 / counts["distractor"], 1 / counts["door"]], dtype=torch.float32, device="cuda")
    weights = weights / weights.sum() * 2
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        row = {"epoch": epoch, "train_loss": loss_sum / len(dataset)}
        history.append(row)
        print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True)
    weights_path = args.output / "fixed.pt"
    torch.save({"state_dict": model.state_dict(), "class_to_idx": dataset.class_to_idx, "architecture": "torchvision_resnet18_dual_view", "image_size_hw": [224, 448], "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}, weights_path)
    training = {"schema_version": "blindassist_context_candidate_verifier_fixed_training_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "dataset_receipt_sha256": sha256(args.dataset / "receipt.json"), "future_cohort_access": False, "epochs": EPOCHS, "batch": args.batch, "seed": 0, "decision_threshold": DECISION_THRESHOLD, "checkpoint_selection": "fixed final epoch without validation", "weights_sha256": sha256(weights_path), "history": history}
    (args.output / "training_receipt.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
