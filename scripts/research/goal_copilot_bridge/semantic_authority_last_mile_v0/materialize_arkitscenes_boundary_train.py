"""Materialize source-disjoint ARKitScenes opening-proxy boundary training data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from .materialize_rgb_cohort import _intrinsics, _metric_depth, _opening_truth


SCHEMA_VERSION = "sage_lm_v1c_arkitscenes_rgbd_opening_proxy_train_v1"
SELECTION_SCORE_MINIMUM = 0.58


def materialize(source_root: Path, evaluation_cohort: Path, output_dir: Path) -> dict:
    cohort = json.loads(evaluation_cohort.read_text(encoding="utf-8"))
    excluded = sorted({row["source"]["sequence"] for row in cohort["episodes"]})
    sequences = [row for row in sorted(source_root.glob("*/*")) if row.is_dir() and row.name not in excluded]
    if len(sequences) < 3:
        raise RuntimeError("fewer than three source-disjoint ARKitScenes sequences remain")
    validation_sequences = {sequences[0].name, sequences[-1].name}
    (output_dir / "masks" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks" / "val").mkdir(parents=True, exist_ok=True)
    rows = []
    scanned = 0
    for sequence in sequences:
        rgb_dir = sequence / "lowres_wide"
        depth_dir = sequence / "lowres_depth"
        intrinsics_dir = sequence / "lowres_wide_intrinsics"
        split = "val" if sequence.name in validation_sequences else "train"
        for image_path in sorted(rgb_dir.glob("*.png")):
            scanned += 1
            depth_path = depth_dir / f"{image_path.stem}.png"
            intrinsics_path = intrinsics_dir / f"{image_path.stem}.pincam"
            if not depth_path.exists() or not intrinsics_path.exists():
                continue
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            truth = _opening_truth(bgr, _metric_depth(depth_path), _intrinsics(intrinsics_path))
            if truth is None or truth["selection_score"] < SELECTION_SCORE_MINIMUM:
                continue
            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            left = int(round(truth["left_x_px"]))
            right = int(round(truth["right_x_px"]))
            y1, y2 = int(mask.shape[0] * 0.30), int(mask.shape[0] * 0.78)
            mask[y1:y2, max(0, left) : min(mask.shape[1], right + 1)] = 255
            mask_path = output_dir / "masks" / split / f"{sequence.name}_{image_path.stem}.png"
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"failed to write {mask_path}")
            rows.append(
                {
                    "environment": sequence.name,
                    "source_sequence": sequence.name,
                    "split": split,
                    "kind": "positive",
                    "image_path": str(image_path.resolve()),
                    "mask_path": str(mask_path.resolve()),
                    "selection_score": truth["selection_score"],
                    "source_depth_discontinuity": truth["source_depth_discontinuity"],
                    "boundary_x_px": [truth["left_x_px"], truth["right_x_px"]],
                    "label_provenance": "RGB_VERTICAL_LINE_PLUS_SOURCE_NATIVE_DEPTH_DISCONTINUITY_OPENING_PROXY",
                }
            )
    if not rows or not any(row["split"] == "train" for row in rows) or not any(row["split"] == "val" for row in rows):
        raise RuntimeError("source-disjoint materialization did not produce both train and validation examples")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "ARKitScenes Training",
        "source_root": str(source_root.resolve()),
        "evaluation_cohort": str(evaluation_cohort.resolve()),
        "excluded_evaluation_sequences": excluded,
        "training_sequences": sorted({row["source_sequence"] for row in rows if row["split"] == "train"}),
        "validation_sequences": sorted({row["source_sequence"] for row in rows if row["split"] == "val"}),
        "selection": {"minimum_score": SELECTION_SCORE_MINIMUM, "frame_stride": 1, "requires_active_pair": False},
        "label_provenance": "AUTOMATIC_OPENING_PROXY_NOT_OFFICIAL_ARKITSCENES_DOOR_ANNOTATION",
        "scanned_frame_count": scanned,
        "case_count": len(rows),
        "train_count": sum(row["split"] == "train" for row in rows),
        "val_count": sum(row["split"] == "val" for row in rows),
        "source_environments": sorted({row["source_sequence"] for row in rows}),
        "excluded_formal_environments": excluded,
        "cases": rows,
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluation-cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(args.source_root, args.evaluation_cohort, args.output_dir)
    print(json.dumps({key: receipt[key] for key in ("scanned_frame_count", "case_count", "train_count", "val_count", "training_sequences", "validation_sequences")}, indent=2))


if __name__ == "__main__":
    main()
