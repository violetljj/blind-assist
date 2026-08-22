#!/usr/bin/env python3
"""Train a fixed depth-structure verifier from consumed proposal cohorts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


FEATURE_NAMES = ["width_fraction", "height_fraction", "aspect_ratio", "center_offset", "proposal_score", "provider_rank", "dino_iou", "dino_score", "depth_p10", "depth_p20", "depth_p30", "depth_p50", "depth_p70", "depth_p80", "depth_p90", "depth_iqr", "depth_std", "fraction_le_1m", "fraction_le_2m", "fraction_gt_4m", "valid_fraction"]


def candidate_features(candidate: dict[str, Any], dino_candidates: list[dict[str, Any]], depth: np.ndarray, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = validated_box(candidate["bbox_xyxy"], "candidate")
    left, right = max(0, int(np.floor(x1))), min(width, int(np.ceil(x2)))
    top, bottom = max(0, int(np.floor(y1))), min(height, int(np.ceil(y2)))
    raw = depth[top:bottom, left:right]
    valid = raw[np.isfinite(raw) & (raw >= 0.4) & (raw <= 8.0)]
    if valid.size:
        quantiles = np.quantile(valid, [0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90])
        depth_std = float(np.std(valid))
        fraction_le_1m = float((valid <= 1.0).mean())
        fraction_le_2m = float((valid <= 2.0).mean())
        fraction_gt_4m = float((valid > 4.0).mean())
    else:
        # Missing public depth is an observable sensor state, not a reason to
        # drop the proposal. Encode it deterministically as out-of-range and
        # expose the absence through valid_fraction=0.
        quantiles = np.full(7, 8.0, dtype=np.float32)
        depth_std = 0.0
        fraction_le_1m = 0.0
        fraction_le_2m = 0.0
        fraction_gt_4m = 1.0
    overlaps = [iou([x1, y1, x2, y2], validated_box(row["bbox_xyxy"], "DINO candidate")) for row in dino_candidates]
    best_index = int(np.argmax(overlaps)) if overlaps else None
    dino_iou = overlaps[best_index] if best_index is not None else 0.0
    dino_score = float(dino_candidates[best_index]["score"]) if best_index is not None else 0.0
    box_pixels = max(1, (bottom - top) * (right - left))
    return [
        (x2 - x1) / width, (y2 - y1) / height, (x2 - x1) / (y2 - y1), abs((x1 + x2) / 2.0 - width / 2.0) / width,
        float(candidate["proposal_score"]), float(candidate["provider_rank"]), dino_iou, dino_score,
        *[float(value) for value in quantiles], float(quantiles[5] - quantiles[1]), depth_std,
        fraction_le_1m, fraction_le_2m, fraction_gt_4m, float(valid.size / box_pixels),
    ]


def load_rows(cohort_roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for root in cohort_roots:
        public, run, private = (_read(root / name) for name in ("public_input.json", "run.json", "private_eval.json"))
        _require(run.get("private_truth_access") is False and run.get("public_sha256") == sha256(root / "public_input.json"), "depth verifier parent boundary mismatch")
        public_cases = {case["case_id"]: case for case in public["cases"]}
        run_cases = {case["case_id"]: case for case in run["cases"]}
        truth_cases = {case["case_id"]: case for case in private["cases"]}
        for case_id in sorted(public_cases):
            range_sensor = public_cases[case_id].get("range_sensor")
            if range_sensor is not None:
                depth_path = Path(range_sensor["depth_path"])
                _require(depth_path.is_file() and sha256(depth_path) == range_sensor["depth_sha256"], "depth verifier public depth drift")
            else:
                image_path = Path(public_cases[case_id]["query"]["image_path"])
                depth_path = Path(str(image_path).replace("image_lcam_front", "depth_lcam_front").replace("_lcam_front.png", "_lcam_front_depth.png"))
                _require(depth_path.is_file(), "aligned public development depth missing")
            depth = decode_depth(depth_path)
            observed = run_cases[case_id]
            candidates = observed["yoloe_candidates"]
            targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in truth_cases[case_id]["legal_targets"]]
            for candidate in candidates:
                box = validated_box(candidate["bbox_xyxy"], f"{case_id} candidate")
                best_iou = max((iou(box, target) for target in targets), default=0.0)
                rows.append({"cohort": root.name, "case_id": case_id, "candidate": candidate, "features": candidate_features(candidate, observed["dino_candidates"], depth, observed["image_width"], observed["image_height"]), "label": int(best_iou >= 0.30), "best_target_iou": best_iou, "truth": truth_cases[case_id], "image_width": observed["image_width"], "image_height": observed["image_height"]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cohorts", nargs="+", type=Path, required=True)
    parser.add_argument("--val-cohorts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "depth structure verifier output already exists")
    train_rows, val_rows = load_rows(args.train_cohorts), load_rows(args.val_cohorts)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", random_state=0, n_jobs=-1)
    model.fit(np.asarray([row["features"] for row in train_rows]), np.asarray([row["label"] for row in train_rows]))
    probabilities = model.predict_proba(np.asarray([row["features"] for row in val_rows]))[:, 1]
    predictions = probabilities >= 0.5
    candidate_confusion = confusion_matrix([row["label"] for row in val_rows], predictions, labels=[0, 1]).tolist()
    candidate_balanced_accuracy = float(balanced_accuracy_score([row["label"] for row in val_rows], predictions))
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
    for row, probability in zip(val_rows, probabilities, strict=True):
        grouped.setdefault(row["case_id"], []).append((row, float(probability)))
    completion_rows = []
    for case_id, candidates in sorted(grouped.items()):
        eligible = []
        for row, probability in candidates:
            feature = dict(zip(FEATURE_NAMES, row["features"], strict=True))
            box = row["candidate"]["bbox_xyxy"]
            if probability >= 0.5 and box[0] <= row["image_width"] / 2.0 <= box[2] and feature["height_fraction"] >= 0.40 and feature["depth_p20"] <= 2.0 and feature["dino_iou"] >= 0.85:
                eligible.append((row, probability))
        selected = max(eligible, key=lambda pair: (pair[1], pair[0]["candidate"]["proposal_score"])) if eligible else None
        exemplar = candidates[0][0]
        legal = exemplar["truth"]["legal_targets"]
        targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in legal]
        opportunity = any(box[0] <= exemplar["image_width"] / 2.0 <= box[2] and float(target["target_depth_median_m"]) <= 2.0 for box, target in zip(targets, legal, strict=True))
        matched = None
        if selected is not None:
            overlaps = [iou(validated_box(selected[0]["candidate"]["bbox_xyxy"], f"{case_id} selected"), target) for target in targets]
            matched = int(np.argmax(overlaps)) if overlaps and max(overlaps) >= 0.30 else None
        correct = selected is not None and matched is not None and float(legal[matched]["target_depth_median_m"]) <= 2.0
        completion_rows.append({"case_id": case_id, "completion_opportunity": opportunity, "completion_decision": selected is not None, "selected_probability": selected[1] if selected else None, "selected_candidate": selected[0]["candidate"] if selected else None, "correct_completion": correct, "false_completion": selected is not None and not correct, "missed_completion_opportunity": opportunity and not correct})
    opportunities = sum(row["completion_opportunity"] for row in completion_rows)
    correct = sum(row["correct_completion"] for row in completion_rows)
    false = sum(row["false_completion"] for row in completion_rows)
    args.output.mkdir(parents=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "decision_threshold": 0.5}, args.output / "model.joblib")
    receipt = {"schema_version": "blindassist_depth_structure_verifier_training_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "train_cohorts": [str(path.resolve()) for path in args.train_cohorts], "val_cohorts": [str(path.resolve()) for path in args.val_cohorts], "future_cohort_access": False, "feature_names": FEATURE_NAMES, "model": {"type": "RandomForestClassifier", "n_estimators": 500, "max_depth": 10, "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced_subsample", "random_state": 0, "decision_threshold": 0.5}, "train_candidate_count": len(train_rows), "val_candidate_count": len(val_rows), "candidate_balanced_accuracy": candidate_balanced_accuracy, "candidate_confusion_tn_fp_fn_tp": candidate_confusion, "completion": {"opportunity_count": opportunities, "correct_count": correct, "false_count": false, "coverage": correct / opportunities if opportunities else None, "terminal": "DEV_DEPTH_STRUCTURE_VERIFIER_PROMISING" if false == 0 and opportunities >= 8 and correct / opportunities >= 0.50 else "DEV_DEPTH_STRUCTURE_VERIFIER_NOT_PROMISING", "rows": completion_rows}, "model_sha256": sha256(args.output / "model.joblib")}
    _atomic_json(args.output / "training_receipt.json", receipt)
    print(json.dumps({"candidate_balanced_accuracy": candidate_balanced_accuracy, "candidate_confusion": candidate_confusion, **{key: receipt["completion"][key] for key in ("opportunity_count", "correct_count", "false_count", "coverage", "terminal")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
