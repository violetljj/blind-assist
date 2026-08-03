"""Run the frozen scale student on ten student-unseen TartanGround parents."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import core as scale_core
from evaluate_camera_conditioned_student_r0 import (
    FEATURE_NAMES,
    RIDGE_ALPHA,
    fit_ridge,
    predict_ridge,
    runtime_features,
)
from evaluate_consumed_tartanground import (
    INTRINSICS,
    DepthAnythingV2MetricSource,
    aligned_scale_diagnostic,
    decode_depth,
    load_jsonl,
    load_metadata,
    sha256,
    strict_band_values,
    summarize_arm,
    up_optical_from_pose,
    write_json_new,
)
from evaluate_metric3d_clearance_field_a0 import clearance_field


CHECKPOINT_SHA256 = "B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0-result", required=True, type=Path)
    parser.add_argument("--cross-validation-result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--dav2-repo", required=True, type=Path)
    parser.add_argument("--dav2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.output_root.exists():
        raise FileExistsError(arguments.output_root)

    r0 = json.loads(arguments.r0_result.read_text(encoding="utf-8"))
    cross_validation = json.loads(
        arguments.cross_validation_result.read_text(encoding="utf-8")
    )
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_EXTERNAL_STUDENT_PREDICTION_OR_EFFECT_EXECUTION":
        raise ValueError("external replication protocol is not frozen")
    if protocol["cross_validation_result_sha256"] != sha256(arguments.cross_validation_result):
        raise ValueError("cross-validation result hash mismatch")
    if not all(cross_validation["gates"].values()):
        raise ValueError("cross-validation did not authorize replication")
    if sha256(arguments.dav2_checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("unexpected DA V2 checkpoint")

    training = []
    for row in r0["records"]:
        with np.load(row["prediction_path"]) as payload:
            relative_depth = payload["da_depth"].astype(np.float64)
        recovery = scale_core.recover_metric_scale(
            relative_depth,
            INTRINSICS,
            scale_core.CameraHeightReceipt(
                row["parent_id"], row["parent_id"], row["height_m"], 0.0
            ),
            row["parent_id"],
            row["parent_id"],
        )
        features = runtime_features(relative_depth, row["height_m"], recovery)
        if features is not None and row["aligned_scale_diagnostic"] is not None:
            training.append((features, float(row["aligned_scale_diagnostic"])))
    if len(training) != int(protocol["training_record_count_expected"]):
        raise ValueError("unexpected final training count")
    model = fit_ridge(
        np.stack([row[0] for row in training]),
        np.log(np.asarray([row[1] for row in training], dtype=np.float64)),
        RIDGE_ALPHA,
    )

    selected_samples = []
    height_by_parent = {}
    role_by_parent = {}
    corpus_receipts = []
    for corpus in protocol["external_corpora"]:
        corpus_root = Path(corpus["root"])
        samples_path = corpus_root / "samples.jsonl"
        if sha256(samples_path) != corpus["samples_sha256"]:
            raise ValueError("external corpus samples hash mismatch")
        eligible = {row["parent_id"]: float(row["robot_height_m"]) for row in corpus["eligible_parents"]}
        rows = [row for row in load_jsonl(samples_path) if row["parent_id"] in eligible]
        if len(rows) != len(eligible) * int(protocol["frames_per_parent"]):
            raise ValueError("unexpected external sample count")
        for parent_id, height_m in eligible.items():
            if parent_id in height_by_parent:
                raise ValueError("external parent repeated across corpora")
            height_by_parent[parent_id] = height_m
            role_by_parent[parent_id] = corpus["role"]
        selected_samples.extend(rows)
        corpus_receipts.append(
            {
                "root": str(corpus_root.resolve()),
                "role": corpus["role"],
                "samples_sha256": sha256(samples_path),
                "selected_record_count": len(rows),
            }
        )
    if len(selected_samples) != int(protocol["external_frame_count"]):
        raise ValueError("unexpected total external frame count")

    arguments.output_root.mkdir(parents=True)
    prediction_root = arguments.output_root / "predictions"
    source = DepthAnythingV2MetricSource(
        arguments.dav2_repo,
        arguments.dav2_checkpoint,
        arguments.device,
        input_size=518,
        precision="fp16" if arguments.device.startswith("cuda") else "fp32",
    )
    metadata_cache = {
        parent_id: load_metadata(arguments.metadata_root, parent_id)
        for parent_id in height_by_parent
    }
    records = []
    for index, sample in enumerate(selected_samples, 1):
        parent_id = sample["parent_id"]
        anchor = int(sample["anchor_frame_id"])
        current = sample["history_rgb"][-1]
        rgb_path = Path(current["image_path"])
        if int(current["frame_id"]) != anchor or float(current["relative_time_s"]) != 0.0:
            raise ValueError("external sample current-frame mismatch")
        if sha256(rgb_path).lower() != str(current["image_sha256"]).lower():
            raise ValueError("external RGB hash mismatch")
        depth_path = rgb_path.parent.parent / "depth" / f"{anchor:06d}.png"
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (640, 640):
            raise ValueError("unexpected external RGB")
        sensor_depth = decode_depth(depth_path.read_bytes())
        metadata, poses = metadata_cache[parent_id]
        height_m = float(metadata["robot_height"])
        if height_m != height_by_parent[parent_id] or anchor >= len(poses):
            raise ValueError("external height or pose mismatch")
        up_optical = up_optical_from_pose(poses[anchor])
        truth = strict_band_values(
            clearance_field(
                sensor_depth,
                INTRINSICS,
                plane_override=(up_optical, height_m, 0.0),
            )
        )

        started = time.perf_counter()
        relative_depth, _ = source.infer(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), {})
        latency_ms = (time.perf_counter() - started) * 1000.0
        prediction_path = prediction_root / parent_id.replace("/", "__") / f"{anchor:06d}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prediction_path, da_depth=relative_depth.astype(np.float32))

        raw = strict_band_values(clearance_field(relative_depth, INTRINSICS))
        recovery = scale_core.recover_metric_scale(
            relative_depth,
            INTRINSICS,
            scale_core.CameraHeightReceipt(parent_id, parent_id, height_m, 0.0),
            parent_id,
            parent_id,
        )
        features = runtime_features(relative_depth, height_m, recovery)
        candidate = None
        reason = None
        predicted_scale = None
        if features is None:
            reason = str(recovery.get("reason", "INVALID_RUNTIME_FEATURES"))
        else:
            predicted_scale = float(np.exp(predict_ridge(model, features)))
            if not scale_core.SCALE_RANGE[0] <= predicted_scale <= scale_core.SCALE_RANGE[1]:
                reason = "STUDENT_SCALE_OUT_OF_RANGE"
            else:
                plane = recovery["ground"]
                candidate = strict_band_values(
                    clearance_field(
                        relative_depth * predicted_scale,
                        INTRINSICS,
                        plane_override=(
                            plane.normal,
                            height_m,
                            plane.normalized_median_residual * height_m,
                        ),
                    )
                )
                if candidate is None:
                    reason = "STRICT_CLEARANCE_BAND_UNKNOWN"
        records.append(
            {
                "parent_id": parent_id,
                "environment": sample["environment"],
                "external_role": role_by_parent[parent_id],
                "anchor_frame_id": anchor,
                "height_m": height_m,
                "rgb_sha256": current["image_sha256"],
                "depth_sha256": sha256(depth_path),
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256(prediction_path),
                "latency_ms": latency_ms,
                "truth": truth,
                "raw": raw,
                "student_candidate": candidate,
                "student_unknown_reason": reason,
                "student_predicted_scale": predicted_scale,
                "aligned_scale_diagnostic": aligned_scale_diagnostic(sensor_depth, relative_depth),
            }
        )
        if index % 20 == 0:
            print(json.dumps({"processed": index, "total": len(selected_samples)}), flush=True)

    raw_summary = summarize_arm(records, "raw")
    student_summary = summarize_arm(records, "student_candidate")
    raw_by_parent = {row["parent_id"]: row for row in raw_summary["parents"]}
    jointly_better = []
    for row in student_summary["parents"]:
        raw = raw_by_parent[row["parent_id"]]
        better = (
            row["clearance_mae_m"] is not None
            and raw["clearance_mae_m"] is not None
            and row["clearance_mae_m"] < raw["clearance_mae_m"]
            and row["false_clear_rate"] is not None
            and raw["false_clear_rate"] is not None
            and row["false_clear_rate"] <= raw["false_clear_rate"]
        )
        jointly_better.append({"parent_id": row["parent_id"], "jointly_better": better})
    macro = student_summary["parent_macro"]
    gates = {
        "known_coverage": macro["known_coverage"] is not None and macro["known_coverage"] >= 0.60,
        "clearance_mae": macro["clearance_mae_m"] is not None and macro["clearance_mae_m"] <= 0.25,
        "envelope_agreement": macro["envelope_agreement"] is not None and macro["envelope_agreement"] >= 0.90,
        "false_clear": macro["false_clear_rate"] is not None and macro["false_clear_rate"] <= 0.05,
        "temporal_delta_mae": macro["temporal_delta_mae_m"] is not None and macro["temporal_delta_mae_m"] <= 0.15,
        "jointly_better_parents": sum(row["jointly_better"] for row in jointly_better) >= 6,
    }
    result = {
        "schema": "blindassist_camera_conditioned_scale_student_external_replication_r0_result_v1",
        "data_role": protocol["data_role"],
        "claim_ceiling": protocol["claim_ceiling"],
        "protocol_sha256": sha256(arguments.protocol),
        "cross_validation_result_sha256": sha256(arguments.cross_validation_result),
        "dav2_checkpoint_sha256": sha256(arguments.dav2_checkpoint),
        "feature_names": list(FEATURE_NAMES),
        "final_training_record_count": len(training),
        "final_model": {
            "feature_mean": model["mean"].tolist(),
            "feature_standard_deviation": model["standard_deviation"].tolist(),
            "weights_intercept_then_features": model["weights"].tolist(),
        },
        "corpus_receipts": corpus_receipts,
        "record_count": len(records),
        "parent_count": len(height_by_parent),
        "records": records,
        "raw_da": raw_summary,
        "student_candidate": student_summary,
        "jointly_better_parents": jointly_better,
        "latency_ms": {
            "median": float(np.median([row["latency_ms"] for row in records])),
            "p95": float(np.quantile([row["latency_ms"] for row in records], 0.95)),
        },
        "gates": gates,
        "terminal": (
            "CAMERA_CONDITIONED_SCALE_STUDENT_EXTERNAL_REPLICATION_ALL_GATES_PASS"
            if all(gates.values())
            else protocol["failure_terminal"]
        ),
    }
    write_json_new(arguments.output_root / "result.json", result)
    print(json.dumps({key: result[key] for key in ("raw_da", "student_candidate", "latency_ms", "gates", "terminal")}, indent=2))


if __name__ == "__main__":
    main()
