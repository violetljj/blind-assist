#!/usr/bin/env python3
"""Train a deterministic proposal-level door-vs-distractor verifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _require
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


def balanced_accuracy(correct_by_class: list[int], total_by_class: list[int]) -> float:
    return sum(correct / total for correct, total in zip(correct_by_class, total_by_class, strict=True)) / len(total_by_class)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    _require(not args.output.exists(), "candidate verifier training output already exists")
    receipt = json.loads((args.dataset / "receipt.json").read_text(encoding="utf-8"))
    _require(receipt.get("future_cohort_access") is False and receipt.get("source_cohorts_consumed_before_materialization") is True, "candidate verifier lineage drift")

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
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    datasets_by_split = {split: datasets.ImageFolder(args.dataset / split, transform=transform) for split in ("train", "val")}
    _require(datasets_by_split["train"].class_to_idx == datasets_by_split["val"].class_to_idx == {"distractor": 0, "door": 1}, "candidate verifier class contract drift")
    loaders = {split: DataLoader(dataset, batch_size=args.batch, shuffle=(split == "train"), num_workers=4, generator=torch.Generator().manual_seed(0)) for split, dataset in datasets_by_split.items()}
    train_counts = receipt["counts"]["train"]
    weights = torch.tensor([1.0 / train_counts["distractor"], 1.0 / train_counts["door"]], dtype=torch.float32, device="cuda")
    weights = weights / weights.sum() * 2.0
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    args.output.mkdir(parents=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in loaders["train"]:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(labels)
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
        row = {"epoch": epoch, "train_loss": train_loss / len(datasets_by_split["train"]), "val_balanced_accuracy": score, "val_correct_by_class": correct, "val_total_by_class": total}
        history.append(row)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "class_to_idx": datasets_by_split["train"].class_to_idx, "architecture": "torchvision_resnet18", "image_size": 224, "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}, args.output / "best.pt")
        print(json.dumps(row), flush=True)
    training = {"schema_version": "blindassist_candidate_verifier_training_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "dataset_receipt_sha256": sha256(args.dataset / "receipt.json"), "epochs": args.epochs, "batch": args.batch, "seed": 0, "decision_threshold": 0.5, "checkpoint_selection": "maximum held-out cohort balanced accuracy; first on tie", "best_val_balanced_accuracy": best, "best_weights_sha256": sha256(args.output / "best.pt"), "history": history}
    (args.output / "training_receipt.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
