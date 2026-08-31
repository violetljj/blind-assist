#!/usr/bin/env python3
"""Consumed-Development oracle probe for Scene Text Telescope usefulness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
import torch
from paddleocr import TextRecognition

from l10_panolab import require, sha256_file


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def canonical_digits(value: str) -> str:
    return "".join(character for character in str(value) if character in "0123456789")


def pp_recognize(model: TextRecognition, image: np.ndarray) -> dict[str, Any]:
    outputs = list(model.predict(input=image, batch_size=1))
    require(len(outputs) == 1, "unexpected PP-OCR recognition result count")
    payload = outputs[0].json["res"]
    return {
        "text": str(payload["rec_text"]),
        "canonical": canonical_digits(payload["rec_text"]),
        "score": round(float(payload["rec_score"]), 8),
    }


def easy_recognize(reader: easyocr.Reader, image: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raw = reader.recognize(
        gray,
        horizontal_list=[[0, gray.shape[1], 0, gray.shape[0]]],
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
    require(len(raw) == 1, "unexpected EasyOCR recognition result count")
    _, text, score = raw[0]
    return {
        "text": str(text),
        "canonical": canonical_digits(text),
        "score": round(float(score), 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-source", type=Path, required=True)
    parser.add_argument("--pixel-truth", type=Path, required=True)
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--v5-protocol", type=Path, required=True)
    parser.add_argument("--easyocr-protocol", type=Path, required=True)
    parser.add_argument("--stt-repository", type=Path, required=True)
    parser.add_argument("--stt-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    require(not output_path.exists(), f"refusing to overwrite: {output_path}")

    source_path = args.public_source.resolve()
    truth_path = args.pixel_truth.resolve()
    materialization_path = args.materialization_manifest.resolve()
    v5_protocol_path = args.v5_protocol.resolve()
    easy_protocol_path = args.easyocr_protocol.resolve()
    stt_repository = args.stt_repository.resolve()
    stt_checkpoint = args.stt_checkpoint.resolve()
    source = read_json(source_path)
    truth = read_json(truth_path)
    materialization = read_json(materialization_path)
    v5_protocol = read_json(v5_protocol_path)
    easy_protocol = read_json(easy_protocol_path)
    require(truth["confirmation_holdout_state"] == "PIXELS_NOT_MATERIALIZED_OR_REVIEWED", "holdout state mismatch")
    require(torch.cuda.is_available(), "CUDA unavailable for STT probe")

    target_by_id = {row["item_id"]: str(row["mission"]["house_number"]) for row in source["rows"]}
    materialized_by_id = {row["item_id"]: row for row in materialization["rows"]}

    rec_spec = v5_protocol["models"]["medium_recognition"]
    rec_root = resolve(rec_spec["path"])
    for filename, expected in rec_spec["sha256"].items():
        require(sha256_file(rec_root / filename) == expected, f"recognition model hash mismatch: {filename}")
    pp_model = TextRecognition(model_dir=str(rec_root), engine="onnxruntime", device="cpu")

    easy_spec = easy_protocol["models"]["easyocr"]
    easy_root = resolve(easy_spec["model_root"])
    easy_user_root = resolve(easy_spec["user_network_root"])
    for filename, expected in easy_spec["sha256"].items():
        require(sha256_file(easy_root / filename) == expected, f"EasyOCR model hash mismatch: {filename}")
    easy_reader = easyocr.Reader(
        ["en"],
        gpu="cuda",
        model_storage_directory=str(easy_root),
        user_network_directory=str(easy_user_root),
        download_enabled=False,
        detector=False,
        verbose=False,
    )

    ipython_stub = types.ModuleType("IPython")
    ipython_stub.embed = lambda: None
    sys.modules.setdefault("IPython", ipython_stub)
    sys.path.insert(0, str(stt_repository))
    from model.tbsrn import TBSRN

    stt = TBSRN(
        scale_factor=2,
        width=128,
        height=32,
        STN=True,
        srb_nums=5,
        hidden_units=32,
    )
    checkpoint = torch.load(stt_checkpoint, map_location="cpu", weights_only=False)
    stt.load_state_dict(checkpoint["state_dict_G"], strict=True)
    stt = stt.eval().cuda()
    for parameter in stt.parameters():
        parameter.requires_grad = False

    expansion = {
        "left_box_widths": 0.5,
        "right_box_widths": 0.5,
        "up_box_heights": 0.75,
        "down_box_heights": 1.0,
    }
    rows = []
    started = time.perf_counter()
    for item_id, annotation in sorted(truth["annotations"].items()):
        if not annotation["target_house_number_visible"]:
            continue
        image_path = materialization_path.parent / f"{item_id}.jpg"
        require(sha256_file(image_path) == materialized_by_id[item_id]["member_sha256"], f"image hash mismatch: {item_id}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        require(image is not None, f"decode failed: {item_id}")
        x1, y1, x2, y2 = [int(value) for value in annotation["target_credential_box_xyxy"]]
        box_width = x2 - x1
        box_height = y2 - y1
        crop_x1 = max(0, round(x1 - expansion["left_box_widths"] * box_width))
        crop_x2 = min(image.shape[1], round(x2 + expansion["right_box_widths"] * box_width))
        crop_y1 = max(0, round(y1 - expansion["up_box_heights"] * box_height))
        crop_y2 = min(image.shape[0], round(y2 + expansion["down_box_heights"] * box_height))
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        require(crop.size > 0, f"empty expanded crop: {item_id}")
        interpolated = cv2.resize(crop, (128, 32), interpolation=cv2.INTER_CUBIC)
        low_resolution = cv2.resize(crop, (64, 16), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(low_resolution, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        with torch.inference_mode():
            enhanced = stt(tensor)[0].detach().clamp(0.0, 1.0).mul(255.0).byte().cpu().permute(1, 2, 0).numpy()
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        target = target_by_id[item_id]
        predictions = {
            "interpolated_ppocrv6": pp_recognize(pp_model, interpolated),
            "stt_ppocrv6": pp_recognize(pp_model, enhanced),
            "interpolated_easyocr_numeric": easy_recognize(easy_reader, interpolated),
            "stt_easyocr_numeric": easy_recognize(easy_reader, enhanced),
        }
        for prediction in predictions.values():
            prediction["exact"] = prediction["canonical"] == target
        row = {
            "item_id": item_id,
            "target": target,
            "original_credential_box_xyxy": [x1, y1, x2, y2],
            "expanded_crop_box_xyxy": [crop_x1, crop_y1, crop_x2, crop_y2],
            "expanded_crop_shape_hw": list(crop.shape[:2]),
            "predictions": predictions,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    metric_names = list(rows[0]["predictions"])
    metrics = {
        name: {
            "exact": sum(row["predictions"][name]["exact"] for row in rows),
            "rate": round(sum(row["predictions"][name]["exact"] for row in rows) / len(rows), 6),
        }
        for name in metric_names
    }
    stt_commit = subprocess.run(
        ["git", "-C", str(stt_repository.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "schema": "blindassist-l10-lsaa-text-telescope-oracle-probe-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "decision": "ORACLE_DIAGNOSTIC_ONLY_NOT_RUNTIME_ALGORITHM_EVIDENCE",
        "inputs": {
            "public_source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "pixel_truth": {"path": str(truth_path), "sha256": sha256_file(truth_path)},
            "materialization_manifest": {"path": str(materialization_path), "sha256": sha256_file(materialization_path)},
            "v5_protocol": {"path": str(v5_protocol_path), "sha256": sha256_file(v5_protocol_path)},
            "easyocr_protocol": {"path": str(easy_protocol_path), "sha256": sha256_file(easy_protocol_path)},
            "stt_repository": {"path": str(stt_repository), "commit": stt_commit},
            "stt_checkpoint": {"path": str(stt_checkpoint), "sha256": sha256_file(stt_checkpoint)},
        },
        "oracle_crop_expansion": expansion,
        "metrics": metrics,
        "rows": rows,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "claim_boundary": "Human credential boxes and fixed expansion enter this consumed-Development oracle. It can decide whether STT merits runtime integration but is not runtime localization, portal ownership, confirmation, arrival, handoff, deployment, or safety evidence.",
    }
    output_path.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps({"metrics": metrics, "output_sha256": sha256_file(output_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
