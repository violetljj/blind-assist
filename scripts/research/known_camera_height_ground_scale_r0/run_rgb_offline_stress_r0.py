#!/usr/bin/env python3
"""Run the frozen parent-balanced RGB stress layer through exact DA and student."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from run_offline_stress_r0 import (
    DEFAULT_PROTOCOL,
    add_clean_delta,
    append_jsonl,
    evaluate_one,
    rotation_homography,
    sha256,
    summarize_scenario,
    write_json_new,
)
from sealed_student import MODEL_ID, SealedScaleStudent

REPO_ROOT = Path(__file__).resolve().parents[3]
HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"

import sys

sys.path.insert(0, str(HFTF_DIR))
from produce_external_rgb_metric_depth_observations import (
    DepthAnythingV2MetricSource,
)


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes()).hexdigest().upper()


def select_parent_balanced(
    records: list[dict[str, Any]], records_per_parent: int
) -> list[dict[str, Any]]:
    """Select only from parent/anchor identity, before source RGB or outcomes."""
    selected = []
    parents = sorted({str(row["parent_id"]) for row in records})
    for parent in parents:
        identity_rows = sorted(
            (
                {
                    "parent_id": str(row["parent_id"]),
                    "anchor_frame_id": int(row["anchor_frame_id"]),
                    "source_index": index,
                }
                for index, row in enumerate(records)
                if str(row["parent_id"]) == parent
            ),
            key=lambda row: row["anchor_frame_id"],
        )
        if len(identity_rows) < records_per_parent:
            raise ValueError(f"too few records for parent-balanced selection: {parent}")
        indices = np.rint(
            np.linspace(0, len(identity_rows) - 1, records_per_parent)
        ).astype(int)
        if len(set(indices.tolist())) != records_per_parent:
            raise ValueError(f"selection indices repeat for parent: {parent}")
        selected.extend(records[identity_rows[index]["source_index"]] for index in indices)
    return selected


def build_rgb_scenarios(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    config = protocol["rgb_second_layer"]["scenarios"]
    scenarios: list[dict[str, Any]] = [
        {"id": "rgb_clean", "family": "rgb_clean", "truth_comparable": True}
    ]
    for retained in config["center_crop"]["retained_fractions"]:
        for mode in config["center_crop"]["coordinate_modes"]:
            scenarios.append(
                {
                    "id": f"rgb_crop_{round(retained * 100):02d}_{mode}",
                    "family": "rgb_center_crop",
                    "retained_fraction": float(retained),
                    "coordinate_mode": mode,
                    "truth_comparable": False,
                }
            )
    for degrees in config["roll"]["degrees"]:
        for mode in config["roll"]["coordinate_modes"]:
            sign = "p" if degrees >= 0 else "m"
            scenarios.append(
                {
                    "id": f"rgb_roll_{sign}{abs(int(degrees))}_{mode}",
                    "family": "rgb_roll",
                    "degrees": float(degrees),
                    "coordinate_mode": mode,
                    "truth_comparable": False,
                }
            )
    for degrees in config["pitch_homography_canary"]["degrees"]:
        sign = "p" if degrees >= 0 else "m"
        scenarios.append(
            {
                "id": f"rgb_pitch_homography_canary_{sign}{abs(int(degrees))}",
                "family": "rgb_pitch_homography_canary",
                "degrees": float(degrees),
                "truth_comparable": False,
            }
        )
    for sigma in config["gaussian_blur_sigma"]:
        scenarios.append(
            {
                "id": f"rgb_gaussian_sigma_{str(sigma).replace('.', '_')}",
                "family": "rgb_gaussian_blur",
                "sigma": float(sigma),
                "truth_comparable": True,
            }
        )
    for length in config["horizontal_motion_blur_length"]:
        scenarios.append(
            {
                "id": f"rgb_motion_blur_{int(length)}",
                "family": "rgb_motion_blur",
                "length": int(length),
                "truth_comparable": True,
            }
        )
    for gamma in config["gamma_darkening"]:
        scenarios.append(
            {
                "id": f"rgb_gamma_{str(gamma).replace('.', '_')}",
                "family": "rgb_gamma_darkening",
                "gamma": float(gamma),
                "truth_comparable": True,
            }
        )
    for multiplier in config["exposure_multiplier"]:
        scenarios.append(
            {
                "id": f"rgb_exposure_{str(multiplier).replace('.', '_')}",
                "family": "rgb_exposure",
                "multiplier": float(multiplier),
                "truth_comparable": True,
            }
        )
    ids = [row["id"] for row in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("RGB scenario ids repeat")
    return scenarios


def crop_with_intrinsics(
    bgr: np.ndarray, intrinsics: np.ndarray, retained_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    height, width = bgr.shape[:2]
    crop_height = max(2, round(height * retained_fraction))
    crop_width = max(2, round(width * retained_fraction))
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    crop = bgr[top : top + crop_height, left : left + crop_width]
    transformed = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    matrix = np.asarray(intrinsics, dtype=np.float64).copy()
    sx, sy = width / crop_width, height / crop_height
    matrix[0, 0] *= sx
    matrix[1, 1] *= sy
    matrix[0, 2] = (matrix[0, 2] - left) * sx
    matrix[1, 2] = (matrix[1, 2] - top) * sy
    return transformed, matrix


def roll_image(bgr: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = bgr.shape[:2]
    affine = cv2.getRotationMatrix2D(
        ((width - 1.0) / 2.0, (height - 1.0) / 2.0), degrees, 1.0
    )
    transformed = cv2.warpAffine(
        bgr,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return transformed, affine


def inverse_roll_depth(depth: np.ndarray, affine: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    inverse = cv2.invertAffineTransform(affine)
    return cv2.warpAffine(
        depth.astype(np.float32),
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    ).astype(np.float64)


def transform_rgb(
    bgr: np.ndarray, intrinsics: np.ndarray, scenario: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    family = scenario["family"]
    matrix = np.asarray(intrinsics, dtype=np.float64).copy()
    context: dict[str, Any] = {}
    if family == "rgb_clean":
        transformed = bgr.copy()
    elif family == "rgb_center_crop":
        transformed, corrected = crop_with_intrinsics(
            bgr, matrix, scenario["retained_fraction"]
        )
        if scenario["coordinate_mode"] == "updated_intrinsics":
            matrix = corrected
        elif scenario["coordinate_mode"] != "stale_intrinsics_mismatch":
            raise ValueError("unsupported crop coordinate mode")
    elif family == "rgb_roll":
        transformed, context["roll_affine"] = roll_image(bgr, scenario["degrees"])
        if scenario["coordinate_mode"] not in {
            "inverse_warp_depth_to_original_coordinates",
            "uncompensated_coordinate_mismatch",
        }:
            raise ValueError("unsupported roll coordinate mode")
    elif family == "rgb_pitch_homography_canary":
        height, width = bgr.shape[:2]
        homography = rotation_homography(matrix, scenario["degrees"], "pitch")
        transformed = cv2.warpPerspective(
            bgr,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
    elif family == "rgb_gaussian_blur":
        sigma = float(scenario["sigma"])
        kernel = 2 * int(np.ceil(3.0 * sigma)) + 1
        transformed = cv2.GaussianBlur(bgr, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    elif family == "rgb_motion_blur":
        length = int(scenario["length"])
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        transformed = cv2.filter2D(bgr, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    elif family == "rgb_gamma_darkening":
        normalized = bgr.astype(np.float32) / 255.0
        transformed = np.clip(np.power(normalized, scenario["gamma"]) * 255.0, 0, 255).astype(np.uint8)
    elif family == "rgb_exposure":
        transformed = np.clip(
            bgr.astype(np.float32) * float(scenario["multiplier"]), 0, 255
        ).astype(np.uint8)
    else:
        raise ValueError(f"unsupported RGB scenario: {family}")
    return transformed, matrix, context


def postprocess_depth(
    depth: np.ndarray, scenario: dict[str, Any], context: dict[str, Any]
) -> np.ndarray:
    if (
        scenario["family"] == "rgb_roll"
        and scenario["coordinate_mode"]
        == "inverse_warp_depth_to_original_coordinates"
    ):
        return inverse_roll_depth(depth, context["roll_affine"])
    return np.asarray(depth, dtype=np.float64)


def load_source_index(
    protocol: dict[str, Any], selected_keys: set[tuple[str, int]]
) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for corpus in protocol["rgb_second_layer"]["subset"]["source_corpora"]:
        samples_path = REPO_ROOT / corpus["samples_path"]
        if sha256(samples_path) != corpus["samples_sha256"]:
            raise ValueError(f"samples hash mismatch: {samples_path}")
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                sample = json.loads(line)
                key = (str(sample["parent_id"]), int(sample["anchor_frame_id"]))
                if key not in selected_keys:
                    continue
                if key in index:
                    raise ValueError(f"duplicate selected source: {key}")
                current = sample["history_rgb"][-1]
                if int(current["frame_id"]) != key[1] or float(current["relative_time_s"]) != 0.0:
                    raise ValueError(f"invalid current-frame source: {key}")
                index[key] = {
                    "image_path": current["image_path"],
                    "image_sha256": str(current["image_sha256"]).upper(),
                }
    if set(index) != selected_keys:
        raise ValueError(f"missing selected RGB sources: {sorted(selected_keys - set(index))}")
    return index


def validate_formal_inputs(
    protocol: dict[str, Any], input_path: Path, model_path: Path, checkpoint_path: Path
) -> None:
    if protocol.get("status") != "FROZEN_BEFORE_OFFLINE_STRESS_EXECUTION":
        raise ValueError("RGB stress protocol is not frozen")
    layer = protocol["rgb_second_layer"]
    if layer.get("cached_runner_implemented") is not False or layer.get("rgb_runner_implemented") is not True:
        raise ValueError("RGB/cached runner authority mismatch")
    if sha256(input_path) != protocol["input"]["result_sha256"]:
        raise ValueError("input result hash mismatch")
    if sha256(model_path) != protocol["sealed_student"]["receipt_sha256"]:
        raise ValueError("sealed model hash mismatch")
    if sha256(checkpoint_path) != layer["dav2"]["checkpoint_sha256"]:
        raise ValueError("DA checkpoint hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input-result", type=Path)
    parser.add_argument("--sealed-model", type=Path)
    parser.add_argument("--dav2-repo", type=Path)
    parser.add_argument("--dav2-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--limit",
        type=int,
        help="Development smoke only; formal selection remains frozen before truncation.",
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    layer = protocol["rgb_second_layer"]
    input_path = args.input_result or REPO_ROOT / protocol["input"]["result_path"]
    model_path = args.sealed_model or REPO_ROOT / protocol["sealed_student"]["receipt_path"]
    repo_path = args.dav2_repo or REPO_ROOT / layer["dav2"]["repo_path"]
    checkpoint_path = args.dav2_checkpoint or REPO_ROOT / layer["dav2"]["checkpoint_path"]
    validate_formal_inputs(protocol, input_path, model_path, checkpoint_path)
    if args.limit is None and args.device != layer["dav2"]["formal_device"]:
        raise ValueError("formal RGB stress requires the frozen CUDA device class")

    input_result = json.loads(input_path.read_text(encoding="utf-8"))
    if len(input_result.get("records", [])) != int(protocol["input"]["required_record_count"]):
        raise ValueError("unexpected source record count")
    per_parent = int(layer["subset"]["records_per_parent"])
    selected = select_parent_balanced(input_result["records"], per_parent)
    if len(selected) != int(layer["subset"]["record_count"]):
        raise ValueError("unexpected parent-balanced RGB subset count")
    selected_keys = {
        (str(row["parent_id"]), int(row["anchor_frame_id"])) for row in selected
    }
    source_index = load_source_index(protocol, selected_keys)
    if args.limit is not None:
        selected = selected[: args.limit]

    student = SealedScaleStudent.load(model_path)
    scenarios = build_rgb_scenarios(protocol)
    intrinsics = np.asarray(protocol["input"]["intrinsics"], dtype=np.float64)
    precision = layer["dav2"]["formal_precision"] if args.device.startswith("cuda") else "fp32"
    source = DepthAnythingV2MetricSource(
        repo_path,
        checkpoint_path,
        args.device,
        input_size=int(layer["dav2"]["input_size"]),
        precision=precision,
    )

    args.output_root.mkdir(parents=True)
    records_path = args.output_root / "records.jsonl"
    progress_path = args.output_root / "progress.jsonl"
    summaries = []
    with records_path.open("x", encoding="utf-8", newline="\n") as records_handle, progress_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as progress_handle:
        for scenario_index, scenario in enumerate(scenarios, 1):
            scenario_rows = []
            for selected_index, row in enumerate(selected, 1):
                key = (str(row["parent_id"]), int(row["anchor_frame_id"]))
                source_row = source_index[key]
                image_path = Path(source_row["image_path"])
                if sha256(image_path) != source_row["image_sha256"]:
                    raise ValueError(f"source RGB hash mismatch: {image_path}")
                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None or bgr.shape[:2] != tuple(protocol["input"]["cached_shape"]):
                    raise ValueError(f"unexpected source RGB: {image_path}")
                transformed_bgr, stressed_intrinsics, context = transform_rgb(
                    bgr, intrinsics, scenario
                )
                started = time.perf_counter()
                da_depth, da_metadata = source.infer(
                    cv2.cvtColor(transformed_bgr, cv2.COLOR_BGR2RGB), {}
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                da_depth = postprocess_depth(da_depth, scenario, context)
                if da_depth.shape != bgr.shape[:2]:
                    raise ValueError("DA output shape mismatch")
                record = evaluate_one(
                    row,
                    da_depth,
                    stressed_intrinsics,
                    float(row["height_m"]),
                    student,
                    scenario,
                )
                record.update(
                    {
                        "source_rgb_path": str(image_path.resolve()),
                        "source_rgb_sha256": source_row["image_sha256"],
                        "transformed_rgb_sha256": array_sha256(transformed_bgr),
                        "da_depth_sha256": array_sha256(da_depth.astype(np.float32)),
                        "da_latency_ms": latency_ms,
                        "da_runtime": da_metadata,
                        "coordinate_mode": scenario.get("coordinate_mode"),
                    }
                )
                scenario_rows.append(record)
                append_jsonl(records_handle, record)
                if selected_index % 10 == 0:
                    append_jsonl(
                        progress_handle,
                        {
                            "event": "scenario_progress",
                            "scenario_id": scenario["id"],
                            "processed": selected_index,
                            "total": len(selected),
                        },
                    )
            summary = summarize_scenario(scenario_rows, scenario)
            summary["da_latency_ms"] = {
                "median": float(np.median([row["da_latency_ms"] for row in scenario_rows])),
                "p95": float(np.quantile([row["da_latency_ms"] for row in scenario_rows], 0.95)),
            }
            summaries.append(summary)
            append_jsonl(
                progress_handle,
                {
                    "event": "scenario_complete",
                    "scenario_index": scenario_index,
                    "scenario_count": len(scenarios),
                    "scenario_id": scenario["id"],
                    "unknown_reason_counts": summary["unknown_reason_counts"],
                    "parent_macro": summary["parent_macro"],
                },
            )
            print(
                json.dumps(
                    {
                        "scenario": scenario["id"],
                        "index": scenario_index,
                        "total": len(scenarios),
                    }
                ),
                flush=True,
            )

    clean = summaries[0]
    for summary in summaries:
        add_clean_delta(summary, clean)
    result = {
        "schema": "blindassist_camera_conditioned_scale_student_rgb_offline_stress_r0_result_v1",
        "protocol_sha256": sha256(args.protocol),
        "input_result_sha256": sha256(input_path),
        "sealed_model_sha256": sha256(model_path),
        "dav2_checkpoint_sha256": sha256(checkpoint_path),
        "model_id": MODEL_ID,
        "data_role": protocol["data_role"],
        "claim_ceiling": protocol["claim_ceiling"],
        "selection_rule": layer["subset"]["selection"],
        "source_record_count": len(selected),
        "source_parent_count": len({row["parent_id"] for row in selected}),
        "scenario_count": len(scenarios),
        "records_path": str(records_path.resolve()),
        "progress_path": str(progress_path.resolve()),
        "pitch_claim_boundary": layer["scenarios"]["pitch_homography_canary"]["claim_boundary"],
        "scenarios": summaries,
        "terminal": (
            "CAMERA_CONDITIONED_SCALE_STUDENT_RGB_OFFLINE_STRESS_R0_COMPLETE_CONSUMED_SYNTHETIC_ONLY"
            if args.limit is None
            else "DEVELOPMENT_SMOKE_NOT_PROTOCOL_EXECUTION"
        ),
    }
    write_json_new(args.output_root / "result.json", result)
    print(json.dumps({"terminal": result["terminal"], "scenario_count": len(scenarios)}, indent=2))


if __name__ == "__main__":
    main()
