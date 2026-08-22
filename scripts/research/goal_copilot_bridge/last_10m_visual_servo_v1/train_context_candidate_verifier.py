#!/usr/bin/env python3
"""Train the fixed dual-view context-aware candidate verifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.train_candidate_verifier import balanced_accuracy
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    _require(not args.output.exists(), "context verifier training output already exists")
    receipt = json.loads((args.dataset / "receipt.json").read_text(encoding="utf-8"))
    _require(receipt.get("future_cohort_access") is False and receipt.get("representation", "").startswith("expanded candidate crop"), "context verifier lineage drift")
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
    splits = {split: datasets.ImageFolder(args.dataset / split, transform=transform) for split in ("train", "val")}
    _require(splits["train"].class_to_idx == splits["val"].class_to_idx == {"distractor": 0, "door": 1}, "context verifier class drift")
    loaders = {split: DataLoader(dataset, batch_size=args.batch, shuffle=split == "train", num_workers=4, generator=torch.Generator().manual_seed(0)) for split, dataset in splits.items()}
    counts = receipt["counts"]["train"]
    class_weights = torch.tensor([1 / counts["distractor"], 1 / counts["door"]], dtype=torch.float32, device="cuda")
    class_weights = class_weights / class_weights.sum() * 2
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    args.output.mkdir(parents=True)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, labels in loaders["train"]:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        model.eval()
        correct, total = [0, 0], [0, 0]
        with torch.inference_mode():
            for images, labels in loaders["val"]:
                predictions = model(images.cuda(non_blocking=True)).argmax(dim=1).cpu()
                for class_id in (0, 1):
                    mask = labels == class_id
                    total[class_id] += int(mask.sum())
                    correct[class_id] += int((predictions[mask] == class_id).sum())
        score = balanced_accuracy(correct, total)
        row = {"epoch": epoch, "train_loss": loss_sum / len(splits["train"]), "val_balanced_accuracy": score, "val_correct_by_class": correct, "val_total_by_class": total}
        history.append(row)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "class_to_idx": splits["train"].class_to_idx, "architecture": "torchvision_resnet18_dual_view", "image_size_hw": [224, 448], "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}, args.output / "best.pt")
        print(json.dumps(row), flush=True)
    training = {"schema_version": "blindassist_context_candidate_verifier_training_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "dataset_receipt_sha256": sha256(args.dataset / "receipt.json"), "epochs": args.epochs, "batch": args.batch, "seed": 0, "decision_threshold": 0.5, "checkpoint_selection": "maximum held-out cohort balanced accuracy; first on tie", "best_val_balanced_accuracy": best, "best_weights_sha256": sha256(args.output / "best.pt"), "history": history}
    (args.output / "training_receipt.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
