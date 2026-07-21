#!/usr/bin/env python3
"""Export the frozen three-class YOLOE candidate without reading R1.2 sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FROZEN_CLASSES = ["traffic cone", "delineator", "bollard"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    weights = args.weights.resolve()
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    cache_dir = args.cache_dir.resolve()
    embedding_cache_dir = args.embedding_cache_dir.resolve()
    if args.candidate_class != FROZEN_CLASSES:
        raise ValueError(f"candidate classes must equal frozen order {FROZEN_CLASSES}")
    if not weights.is_file() or sha256_file(weights) != args.expected_weights_sha256.lower():
        raise ValueError("source weights missing or SHA-256 mismatch")
    if output.exists() or receipt.exists():
        raise ValueError("refusing to overwrite export or receipt")

    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache_dir)

    old_cwd = Path.cwd()
    os.chdir(embedding_cache_dir)
    try:
        import torch
        import ultralytics
        from ultralytics import YOLOE

        model = YOLOE(str(weights))
        model.set_classes(args.candidate_class)
        if list(model.names.values()) != args.candidate_class:
            raise ValueError("model class order differs from frozen candidate")
        exported_value = model.export(
            format="tflite",
            imgsz=args.image_size,
            half=args.half,
            nms=False,
            batch=1,
        )
        exported = Path(str(exported_value)).resolve()
    finally:
        os.chdir(old_cwd)

    if not exported.is_file() or exported.suffix.lower() != ".tflite":
        candidates = sorted(
            embedding_cache_dir.rglob("*.tflite"), key=lambda item: item.stat().st_mtime
        )
        if not candidates:
            raise FileNotFoundError(f"Ultralytics export did not produce TFLite: {exported_value}")
        exported = candidates[-1]
    shutil.copy2(exported, output)

    payload = {
        "schema": "blindassist_ustrf_crosscam_yoloe_static_export_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "yoloe11s_prompted_marker_static3_r12_v1",
        "source_weights_sha256": sha256_file(weights),
        "frozen_classes": args.candidate_class,
        "image_size": args.image_size,
        "half": args.half,
        "nms": False,
        "exported_model_path": str(output),
        "exported_model_sha256": sha256_file(output),
        "exported_model_size_bytes": output.stat().st_size,
        "runtime": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
        },
        "authority": {
            "r12_sources_read": False,
            "training_performed": False,
            "android_runtime_change_authorized": False,
            "production_model_replacement_authorized": False,
        },
    }
    receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(receipt) + ".sha256").write_text(sha256_file(receipt) + "\n", encoding="ascii")
    print("USTRF_YOLOE_STATIC_EXPORT_OK", payload["exported_model_sha256"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-class", action="append", required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
