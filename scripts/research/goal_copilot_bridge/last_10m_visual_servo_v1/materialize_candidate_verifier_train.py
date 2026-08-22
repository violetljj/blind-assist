#!/usr/bin/env python3
"""Materialize automatic door-vs-distractor proposal crops from consumed cohorts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


TARGET_IOU = 0.30
EXPANSION = 0.15


def expanded_crop(image: Image.Image, box: Sequence[float], expansion: float = EXPANSION) -> Image.Image:
    x1, y1, x2, y2 = validated_box(box, "candidate crop")
    pad_x, pad_y = (x2 - x1) * expansion, (y2 - y1) * expansion
    return image.crop((max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)), min(image.width, int(x2 + pad_x)), min(image.height, int(y2 + pad_y))))


def materialize(cohort_roots: list[Path], split: str, output: Path) -> list[dict]:
    rows = []
    for cohort_root in cohort_roots:
        public, run, private = (_read(cohort_root / name) for name in ("public_input.json", "run.json", "private_eval.json"))
        _require(run.get("private_truth_access") is False and run.get("public_sha256") == sha256(cohort_root / "public_input.json"), "consumed provider boundary mismatch")
        public_cases = {case["case_id"]: case for case in public["cases"]}
        run_cases = {case["case_id"]: case for case in run["cases"]}
        truth_cases = {case["case_id"]: case for case in private["cases"]}
        _require(set(public_cases) == set(run_cases) == set(truth_cases), "candidate verifier roster mismatch")
        for case_id in sorted(public_cases):
            image_path = Path(public_cases[case_id]["query"]["image_path"])
            _require(image_path.is_file() and sha256(image_path) == public_cases[case_id]["query"]["image_sha256"], "candidate verifier image drift")
            targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in truth_cases[case_id]["legal_targets"]]
            candidates = run_cases[case_id].get("yoloe_candidates", run_cases[case_id].get("candidates"))
            _require(isinstance(candidates, list), "candidate list missing")
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            for candidate in candidates:
                box = validated_box(candidate["bbox_xyxy"], f"{case_id} candidate")
                best_iou = max((iou(box, target) for target in targets), default=0.0)
                label = "door" if best_iou >= TARGET_IOU else "distractor"
                rank = int(candidate["provider_rank"])
                filename = f"{cohort_root.name}_{case_id}_r{rank:02d}.jpg"
                target_path = output / split / label / filename
                target_path.parent.mkdir(parents=True, exist_ok=True)
                expanded_crop(image, box).save(target_path, quality=95)
                rows.append({"split": split, "cohort": cohort_root.name, "case_id": case_id, "provider_rank": rank, "label": label, "best_target_iou": best_iou, "crop_path": str(target_path.resolve()), "crop_sha256": sha256(target_path)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cohorts", nargs="+", type=Path, required=True)
    parser.add_argument("--val-cohorts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "candidate verifier dataset already exists")
    rows = materialize(args.train_cohorts, "train", args.output) + materialize(args.val_cohorts, "val", args.output)
    receipt = {"schema_version": "blindassist_candidate_verifier_dataset_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "source_cohorts_consumed_before_materialization": True, "future_cohort_access": False, "target_iou": TARGET_IOU, "crop_expansion": EXPANSION, "train_cohorts": [str(path.resolve()) for path in args.train_cohorts], "val_cohorts": [str(path.resolve()) for path in args.val_cohorts], "counts": {split: {label: sum(row["split"] == split and row["label"] == label for row in rows) for label in ("door", "distractor")} for split in ("train", "val")}, "cases": rows}
    _atomic_json(args.output / "receipt.json", receipt)
    print(json.dumps(receipt["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
