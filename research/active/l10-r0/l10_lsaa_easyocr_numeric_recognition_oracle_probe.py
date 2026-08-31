#!/usr/bin/env python3
"""Oracle diagnostic for the frozen EasyOCR English recognizer on LSAA plaques."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import easyocr

from l10_panolab import require, sha256_file


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-source", type=Path, required=True)
    parser.add_argument("--pixel-truth", type=Path, required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--easyocr-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    require(not output_path.exists(), f"refusing to overwrite: {output_path}")

    source_path = args.public_source.resolve()
    pixel_truth_path = args.pixel_truth.resolve()
    materialization_path = args.materialization_manifest.resolve()
    easy_protocol_path = args.easyocr_protocol.resolve()
    source = read_json(source_path)
    pixel_truth = read_json(pixel_truth_path)
    materialization = read_json(materialization_path)
    easy_protocol = read_json(easy_protocol_path)
    require(pixel_truth["confirmation_holdout_state"] == "PIXELS_NOT_MATERIALIZED_OR_REVIEWED", "holdout state mismatch")
    target_by_id = {row["item_id"]: str(row["mission"]["house_number"]) for row in source["rows"]}
    materialized_by_id = {row["item_id"]: row for row in materialization["rows"]}
    model_spec = easy_protocol["models"]["easyocr"]
    model_root = (Path.cwd() / model_spec["model_root"]).resolve()
    user_root = (Path.cwd() / model_spec["user_network_root"]).resolve()
    for filename, expected in model_spec["sha256"].items():
        require(sha256_file(model_root / filename) == expected, f"model hash mismatch: {filename}")
    reader = easyocr.Reader(
        ["en"],
        gpu="cuda",
        model_storage_directory=str(model_root),
        user_network_directory=str(user_root),
        download_enabled=False,
        detector=False,
        verbose=False,
    )

    rows = []
    started = time.perf_counter()
    for item_id, annotation in sorted(pixel_truth["annotations"].items()):
        if not annotation["target_house_number_visible"]:
            continue
        image_path = materialization_path.parent / f"{item_id}.jpg"
        require(sha256_file(image_path) == materialized_by_id[item_id]["member_sha256"], f"image hash mismatch: {item_id}")
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        require(image is not None, f"decode failed: {item_id}")
        x1, y1, x2, y2 = [int(value) for value in annotation["target_credential_box_xyxy"]]
        crop = image[y1:y2, x1:x2]
        require(crop.size > 0, f"empty crop: {item_id}")
        raw = reader.recognize(
            crop,
            horizontal_list=[[0, crop.shape[1], 0, crop.shape[0]]],
            free_list=[],
            decoder="greedy",
            batch_size=1,
            workers=0,
            allowlist="0123456789",
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.5,
            reformat=False,
        )
        require(len(raw) == 1, f"unexpected recognition count: {item_id}")
        _, predicted, score = raw[0]
        target = target_by_id[item_id]
        row = {
            "item_id": item_id,
            "target": target,
            "credential_box_xyxy": [x1, y1, x2, y2],
            "crop_shape_hw": list(crop.shape),
            "prediction": str(predicted),
            "score": round(float(score), 8),
            "exact": str(predicted) == target,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    exact = sum(row["exact"] for row in rows)
    result = {
        "schema": "blindassist-l10-lsaa-easyocr-numeric-recognition-oracle-probe-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "decision": "ORACLE_DIAGNOSTIC_ONLY_NOT_RUNTIME_ALGORITHM_EVIDENCE",
        "inputs": {
            "public_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "pixel_truth": {"path": str(pixel_truth_path), "sha256": sha256_file(pixel_truth_path)},
            "materialization_manifest": {"path": str(materialization_path), "sha256": sha256_file(materialization_path)},
            "easyocr_protocol": {"path": str(easy_protocol_path), "sha256": sha256_file(easy_protocol_path)},
            "model_sha256": model_spec["sha256"],
        },
        "metrics": {
            "visible_opportunities": len(rows),
            "exact_recognition": exact,
            "exact_recognition_rate": round(exact / len(rows), 6),
        },
        "rows": rows,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "claim_boundary": "Human credential boxes and a digit-only alphabet enter this oracle probe. It diagnoses recognizer capacity only and is not runtime localization, portal ownership, confirmation, arrival, handoff, deployment, or safety evidence.",
    }
    output_path.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"metrics": result["metrics"], "output_sha256": sha256_file(output_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
