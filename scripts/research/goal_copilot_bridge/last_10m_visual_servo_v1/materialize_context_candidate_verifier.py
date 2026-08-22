#!/usr/bin/env python3
"""Materialize dual-view proposal images for a context-aware verifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_candidate_verifier_train import TARGET_IOU, expanded_crop
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


VIEW_SIZE = 224


def dual_view(image: Image.Image, box: Sequence[float]) -> Image.Image:
    crop = expanded_crop(image, box).resize((VIEW_SIZE, VIEW_SIZE), Image.Resampling.BILINEAR)
    context = image.copy()
    draw = ImageDraw.Draw(context)
    x1, y1, x2, y2 = validated_box(box, "context candidate")
    line_width = max(3, round(min(image.size) / 160))
    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=line_width)
    context = context.resize((VIEW_SIZE, VIEW_SIZE), Image.Resampling.BILINEAR)
    output = Image.new("RGB", (VIEW_SIZE * 2, VIEW_SIZE))
    output.paste(crop, (0, 0))
    output.paste(context, (VIEW_SIZE, 0))
    return output


def materialize(roots: list[Path], split: str, output: Path) -> list[dict]:
    rows = []
    for root in roots:
        public, run, private = (_read(root / name) for name in ("public_input.json", "run.json", "private_eval.json"))
        _require(run.get("private_truth_access") is False and run.get("public_sha256") == sha256(root / "public_input.json"), "context verifier parent boundary mismatch")
        public_cases = {case["case_id"]: case for case in public["cases"]}
        run_cases = {case["case_id"]: case for case in run["cases"]}
        private_cases = {case["case_id"]: case for case in private["cases"]}
        for case_id in sorted(public_cases):
            image_path = Path(public_cases[case_id]["query"]["image_path"])
            _require(sha256(image_path) == public_cases[case_id]["query"]["image_sha256"], "context verifier image drift")
            targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in private_cases[case_id]["legal_targets"]]
            candidates = run_cases[case_id].get("yoloe_candidates", run_cases[case_id].get("candidates"))
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            for candidate in candidates:
                box = validated_box(candidate["bbox_xyxy"], f"{case_id} candidate")
                best_iou = max((iou(box, target) for target in targets), default=0.0)
                label = "door" if best_iou >= TARGET_IOU else "distractor"
                rank = int(candidate["provider_rank"])
                target_path = output / split / label / f"{root.name}_{case_id}_r{rank:02d}.jpg"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                dual_view(image, box).save(target_path, quality=95)
                rows.append({"split": split, "cohort": root.name, "case_id": case_id, "provider_rank": rank, "label": label, "best_target_iou": best_iou, "image_path": str(target_path.resolve()), "image_sha256": sha256(target_path)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cohorts", nargs="+", type=Path, required=True)
    parser.add_argument("--val-cohorts", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "context verifier dataset already exists")
    rows = materialize(args.train_cohorts, "train", args.output) + materialize(args.val_cohorts, "val", args.output)
    counts = {split: {label: sum(row["split"] == split and row["label"] == label for row in rows) for label in ("door", "distractor")} for split in ("train", "val")}
    receipt = {"schema_version": "blindassist_context_candidate_verifier_dataset_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "source_cohorts_consumed_before_materialization": True, "future_cohort_access": False, "target_iou": TARGET_IOU, "view_size": VIEW_SIZE, "representation": "expanded candidate crop concatenated with full frame and red candidate box", "train_cohorts": [str(path.resolve()) for path in args.train_cohorts], "val_cohorts": [str(path.resolve()) for path in args.val_cohorts], "counts": counts, "cases": rows}
    _atomic_json(args.output / "receipt.json", receipt)
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
