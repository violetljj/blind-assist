#!/usr/bin/env python3
"""Materialize frozen parent-A2 disagreement for locked train/validation identities."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

from p3_r0_1_asset_common import (
    commit_outputs,
    canonical_bytes,
    exact_fields,
    load_json,
    output_receipt,
    pretty_bytes,
    request_sha256,
    require,
    resolve_inside,
    sha256_file,
    sha256_bytes,
    validate_protocol,
    verify_bound_file,
    verify_producer_sha,
)


REQUEST_SCHEMA = "blindassist_p3_r0_1_frozen_a2_disagreement_request"
CACHE_SCHEMA = "blindassist_p3_r0_1_frozen_a2_disagreement_cache_jsonl"
MANIFEST_SCHEMA = "blindassist_p3_r0_1_frozen_a2_disagreement_manifest"
RECEIPT_SCHEMA = "blindassist_p3_r0_1_frozen_a2_disagreement_materialization_receipt"
LOCK_SCHEMA = "blindassist_p3_r0_1_role_identity_lock"
CATALOG_SCHEMA = "blindassist_p3_r0_1_role_source_catalog"


def _default_infer_factory(repo_root: Path, request: dict[str, Any]) -> Callable[[dict[str, Any]], float]:
    # ML imports are intentionally delayed until every static binding and output preflight passes.
    import cv2
    import numpy as np
    import torch
    from torch.nn import functional

    dav2_root = resolve_inside(repo_root, str(request["dav2_repo_path"]))
    sys.path.insert(0, str(dav2_root / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        max_depth=20,
    )
    checkpoint_path = verify_bound_file(repo_root, request["a2_checkpoint"], "A2 checkpoint")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(torch.device(request["device"])).eval()

    def infer(row: dict[str, Any]) -> float:
        rgb_path = resolve_inside(repo_root, row["rgb_path"])
        teacher_path = resolve_inside(repo_root, row["teacher_depth_path"])
        require(rgb_path.is_file() and teacher_path.is_file(), "frame asset missing")
        require(sha256_file(rgb_path) == row["rgb_sha256"], "RGB asset SHA mismatch")
        require(sha256_file(teacher_path) == row["teacher_depth_sha256"], "teacher asset SHA mismatch")
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        require(image is not None, "RGB decode failed")
        tensor, _ = model.image2tensor(image, int(request["input_size"]))
        tensor = tensor.to(request["device"])
        with torch.inference_mode():
            prediction = model(tensor)
        teacher = np.load(teacher_path, mmap_mode="r")
        reference = str(row["teacher_depth_ref"])
        if teacher.ndim == 3:
            require(reference.startswith("npy-index:"), "3D teacher cache needs npy-index reference")
            teacher = teacher[int(reference.split(":", 1)[1])]
        require(teacher.ndim == 2, "teacher depth must be 2D")
        prediction = functional.interpolate(
            prediction[:, None], size=teacher.shape, mode="bilinear", align_corners=True
        )[0, 0].float().cpu().numpy()
        valid = np.isfinite(teacher) & (teacher > 0.1) & (teacher <= 20.0) & np.isfinite(prediction) & (prediction > 0.1)
        require(int(valid.sum()) > 0, "no paired valid depth pixels")
        return float(np.mean(np.abs(np.log(np.clip(prediction[valid], 0.1, 20.0)) - np.log(np.clip(teacher[valid], 0.1, 20.0)))))

    return infer


def build(
    repo_root: Path,
    request: dict[str, Any],
    source_path: Path,
    *,
    infer_factory: Callable[[Path, dict[str, Any]], Callable[[dict[str, Any]], float]] = _default_infer_factory,
) -> None:
    exact_fields(request, {"schema", "protocol", "a2_checkpoint", "a2_training_receipt", "identity_lock", "source_catalog", "dav2_repo_path", "dav2_dpt_source", "input_size", "device", "producer_sha256", "outputs"}, "request")
    require(request["schema"] == REQUEST_SCHEMA, "request schema drift")
    require(request["input_size"] == 392, "A2 input size drift")
    producer_sha = verify_producer_sha(request["producer_sha256"], source_path)
    _, protocol_sha = validate_protocol(repo_root, request["protocol"])
    checkpoint_path = verify_bound_file(repo_root, request["a2_checkpoint"], "A2 checkpoint")
    training_receipt_path = verify_bound_file(repo_root, request["a2_training_receipt"], "A2 training receipt")
    lock_path = verify_bound_file(repo_root, request["identity_lock"], "identity lock")
    catalog_path = verify_bound_file(repo_root, request["source_catalog"], "source catalog")
    verify_bound_file(repo_root, request["dav2_dpt_source"], "DA V2 source")
    training_receipt = load_json(training_receipt_path)
    require(training_receipt.get("schema") == "blindassist_dav2_392_distillation_a2_r0_training_result", "A2 receipt schema drift")
    require(training_receipt.get("truth_inputs_opened") is False, "A2 receipt truth boundary violated")
    require(training_receipt.get("checkpoint", {}).get("sha256") == request["a2_checkpoint"]["sha256"].upper(), "A2 selected checkpoint mismatch")
    lock = load_json(lock_path)
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("holdout_outcomes_opened") is False, "identity lock drift")
    catalog = load_json(catalog_path)
    exact_fields(catalog, {"schema", "train_validation_frames", "public_holdout_frames"}, "source catalog")
    require(catalog.get("schema") == CATALOG_SCHEMA, "source catalog schema drift")
    rows = {row["frame_id"]: row for row in catalog["train_validation_frames"]}
    selected = []
    for clip in lock["clips"]:
        if clip["role"] in {"train", "validation"}:
            selected.extend(clip["frame_ids"])
    require(len(selected) == len(set(selected)) and selected, "locked training identities invalid")
    for frame_id in selected:
        require(frame_id in rows, "locked frame absent from source catalog")
        row = rows[frame_id]
        exact_fields(row, {
            "frame_id", "video_id", "parent_id", "timestamp_ns", "rgb_identity", "rgb_path", "rgb_sha256",
            "teacher_depth_ref", "teacher_depth_path", "teacher_depth_sha256", "teacher_timestamp_ns",
            "teacher_valid", "tof_valid", "clearance_m", "geometry_state", "geometry_target_valid",
        }, "training source row")
        rgb_path = resolve_inside(repo_root, row["rgb_path"])
        teacher_path = resolve_inside(repo_root, row["teacher_depth_path"])
        require(rgb_path.is_file() and sha256_file(rgb_path) == row["rgb_sha256"], "RGB asset SHA mismatch")
        require(teacher_path.is_file() and sha256_file(teacher_path) == row["teacher_depth_sha256"], "teacher asset SHA mismatch")
    exact_fields(request["outputs"], {"cache", "manifest", "receipt"}, "outputs")
    # Output absence is checked before any model import/load.
    from p3_r0_1_asset_common import assert_outputs_absent
    assert_outputs_absent(repo_root, [str(request["outputs"][key]) for key in ("cache", "manifest", "receipt")])
    infer = infer_factory(repo_root, request)
    cache_rows = []
    for frame_id in selected:
        require(frame_id in rows, "locked frame absent from source catalog")
        value = float(infer(rows[frame_id]))
        require(math.isfinite(value) and value >= 0.0, "disagreement must be finite and non-negative")
        cache_rows.append({"schema": CACHE_SCHEMA, "frame_id": frame_id, "mean_abs_log_depth_disagreement": value})
    cache_bytes = b"".join(canonical_bytes(row) + b"\n" for row in cache_rows)
    cache_sha = sha256_bytes(cache_bytes)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "FROZEN_PARENT_A2_ONLY",
        "protocol_sha256": protocol_sha,
        "a2_checkpoint_sha256": request["a2_checkpoint"]["sha256"].upper(),
        "a2_training_receipt_sha256": request["a2_training_receipt"]["sha256"].upper(),
        "identity_lock_sha256": request["identity_lock"]["sha256"].upper(),
        "source_catalog_sha256": request["source_catalog"]["sha256"].upper(),
        "cache_sha256": cache_sha,
        "producer_sha256": producer_sha,
        "frame_count": len(cache_rows),
        "current_student_used": False,
        "p3_model_constructed": False,
        "optimizer_constructed": False,
        "holdout_outcomes_opened": False,
    }
    outputs = {
        "cache": (str(request["outputs"]["cache"]), cache_bytes),
        "manifest": (str(request["outputs"]["manifest"]), pretty_bytes(manifest)),
    }
    receipt = output_receipt(
        schema=RECEIPT_SCHEMA,
        producer_sha256=producer_sha,
        request_sha256=request_sha256(request),
        input_sha256={
            "protocol": protocol_sha,
            "a2_checkpoint": request["a2_checkpoint"]["sha256"].upper(),
            "a2_training_receipt": request["a2_training_receipt"]["sha256"].upper(),
            "identity_lock": request["identity_lock"]["sha256"].upper(),
            "source_catalog": request["source_catalog"]["sha256"].upper(),
            "dav2_dpt_source": request["dav2_dpt_source"]["sha256"].upper(),
        },
        outputs=outputs,
    )
    commit_outputs(repo_root, outputs=outputs, receipt_relative=str(request["outputs"]["receipt"]), receipt=receipt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo_root.resolve(), json.loads(args.request.read_text(encoding="utf-8")), Path(__file__).resolve())


if __name__ == "__main__":
    main()
