#!/usr/bin/env python3
"""Oracle diagnostic: run frozen PP-OCRv6 recognition on human text boxes.

This probe uses consumed Development annotations and is not an inference
mechanism or confirmation result.  It only distinguishes localization failure
from recognizer/information failure.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from paddleocr import TextRecognition

from l10_panolab import require, sha256_file


SCHEMA = "blindassist-l10-lsaa-ppocrv6-recognition-oracle-probe-v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-source", type=Path, required=True)
    parser.add_argument("--pixel-truth", type=Path, required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--v5-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    require(not output_path.exists(), f"refusing to overwrite: {output_path}")

    public_source_path = args.public_source.resolve()
    pixel_truth_path = args.pixel_truth.resolve()
    materialization_path = args.materialization_manifest.resolve()
    v5_protocol_path = args.v5_protocol.resolve()
    public_source = read_json(public_source_path)
    pixel_truth = read_json(pixel_truth_path)
    materialization = read_json(materialization_path)
    v5_protocol = read_json(v5_protocol_path)
    require(
        pixel_truth["confirmation_holdout_state"] == "PIXELS_NOT_MATERIALIZED_OR_REVIEWED",
        "confirmation holdout state mismatch",
    )
    materialized_by_id = {row["item_id"]: row for row in materialization["rows"]}
    target_by_id = {row["item_id"]: str(row["mission"]["house_number"]) for row in public_source["rows"]}

    recognition_spec = v5_protocol["models"]["medium_recognition"]
    recognition_root = resolve(recognition_spec["path"])
    for filename, expected in recognition_spec["sha256"].items():
        path = recognition_root / filename
        require(path.is_file(), f"missing recognition model file: {path}")
        require(sha256_file(path) == expected, f"recognition model hash mismatch: {path}")
    recognizer = TextRecognition(
        model_dir=str(recognition_root),
        engine="onnxruntime",
        device="cpu",
    )

    rows = []
    started = time.perf_counter()
    for item_id, annotation in sorted(pixel_truth["annotations"].items()):
        if not annotation["target_house_number_visible"]:
            continue
        image_spec = materialized_by_id[item_id]
        image_path = materialization_path.parent / f"{item_id}.jpg"
        require(image_path.is_file(), f"missing image: {image_path}")
        require(sha256_file(image_path) == image_spec["member_sha256"], f"image hash mismatch: {item_id}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        require(image is not None, f"image decode failed: {item_id}")
        x1, y1, x2, y2 = [int(value) for value in annotation["target_credential_box_xyxy"]]
        crop = image[y1:y2, x1:x2]
        require(crop.size > 0, f"empty oracle crop: {item_id}")
        outputs = list(recognizer.predict(input=crop, batch_size=1))
        require(len(outputs) == 1, f"unexpected recognizer result count: {item_id}")
        payload = outputs[0].json["res"]
        target = target_by_id[item_id]
        predicted = str(payload["rec_text"])
        row = {
            "item_id": item_id,
            "target": target,
            "credential_box_xyxy": [x1, y1, x2, y2],
            "crop_shape_hw": list(crop.shape[:2]),
            "prediction": predicted,
            "score": round(float(payload["rec_score"]), 8),
            "exact": predicted.strip().casefold() == target.strip().casefold(),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    exact = sum(row["exact"] for row in rows)
    result = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "decision": "ORACLE_DIAGNOSTIC_ONLY_NOT_RUNTIME_ALGORITHM_EVIDENCE",
        "inputs": {
            "public_source": {"path": str(public_source_path), "sha256": sha256_file(public_source_path)},
            "pixel_truth": {"path": str(pixel_truth_path), "sha256": sha256_file(pixel_truth_path)},
            "materialization_manifest": {"path": str(materialization_path), "sha256": sha256_file(materialization_path)},
            "v5_protocol": {"path": str(v5_protocol_path), "sha256": sha256_file(v5_protocol_path)},
            "recognition_model": {
                "path": str(recognition_root),
                "sha256": recognition_spec["sha256"],
            },
        },
        "runtime": {
            "paddleocr": importlib.metadata.version("paddleocr"),
            "paddlex": importlib.metadata.version("paddlex"),
            "onnxruntime-gpu": importlib.metadata.version("onnxruntime-gpu"),
        },
        "metrics": {
            "visible_opportunities": len(rows),
            "exact_recognition": exact,
            "exact_recognition_rate": round(exact / len(rows), 6),
        },
        "rows": rows,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "claim_boundary": "Human credential boxes enter this oracle probe. Results diagnose recognizer capacity only and provide no runtime localization, portal ownership, confirmation, arrival, handoff, deployment, or safety evidence.",
    }
    output_path.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"metrics": result["metrics"], "output_sha256": sha256_file(output_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
