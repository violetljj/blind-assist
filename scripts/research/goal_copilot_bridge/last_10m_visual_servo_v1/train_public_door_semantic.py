#!/usr/bin/env python3
"""Train a public-data-only current-frame door semantic segmenter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_public_door_semantic_train import FORMAL_ENVIRONMENTS
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


BASE_MODEL_SHA256 = "68ac71bef2868c987fff7cd7c49cb656922f6d8447ad21af3893305577435000"


def validate_receipt(receipt: dict) -> None:
    sources = set(receipt.get("source_environments", []))
    excluded = set(receipt.get("excluded_formal_environments", []))
    _require(receipt.get("private_truth_access") is False, "training receipt crossed private boundary")
    _require(not (sources & FORMAL_ENVIRONMENTS), "formal environment leaked into training")
    _require(FORMAL_ENVIRONMENTS <= excluded, "formal exclusion declaration incomplete")
    _require(receipt.get("case_count", 0) >= 500 and receipt.get("val_count", 0) >= 100, "training denominator insufficient")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    _require(not args.output.exists(), "door semantic training output already exists")
    receipt_path = args.dataset / "receipt.json"
    data_path = args.dataset / "data.yaml"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_receipt(receipt)
    _require(sha256(data_path) == receipt["data_yaml_sha256"], "training data YAML drift")
    _require(args.base_model.is_file() and sha256(args.base_model) == BASE_MODEL_SHA256, "base semantic model drift")
    args.output.mkdir(parents=True)
    training = {
        "schema_version": "blindassist_public_door_semantic_training_run_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "DEVELOPMENT_ONLY",
        "private_truth_access": False,
        "dataset_receipt_sha256": sha256(receipt_path),
        "data_yaml_sha256": sha256(data_path),
        "base_model_sha256": sha256(args.base_model),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": 640,
        "seed": 0,
        "deterministic": True,
        "mosaic": 0.0,
    }
    (args.output / "training_receipt.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")

    from ultralytics import YOLO

    YOLO(str(args.base_model.resolve())).train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        workers=4,
        device=0,
        project=str(args.output.resolve()),
        name="yolo26n_public_door_semantic",
        exist_ok=False,
        seed=0,
        deterministic=True,
        pretrained=True,
        patience=10,
        mosaic=0.0,
        plots=True,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
