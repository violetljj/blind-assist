#!/usr/bin/env python3
"""Evaluate a track-carried lexical ledger on active Panoramax entrance views.

The ledger deliberately keeps target-name lexical candidates and the authorized
entrance ray as separate evidence components.  A candidate seen before a
provider-linked sidestep may remain available after the move, but it is never
converted into a portal-ownership binding by this evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from paddleocr import PaddleOCR

from l10_panolab_entrance_ray import project_entrance_ray


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-track-lexical-ledger-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-track-lexical-ledger-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_file(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    require(path.is_file(), f"MISSING_FROZEN_INPUT:{path}")
    require(sha256(path) == spec["sha256"], f"HASH_MISMATCH:{path}")
    if "bytes" in spec:
        require(path.stat().st_size == int(spec["bytes"]), f"BYTE_COUNT_MISMATCH:{path}")
    return path


def verify_model(spec: dict[str, Any]) -> Path:
    root = resolve(spec["path"])
    require(root.is_dir(), f"MISSING_MODEL:{root}")
    for name, expected in spec["sha256"].items():
        require(sha256(root / name) == expected, f"MODEL_HASH_MISMATCH:{root / name}")
    return root


def ascii_tokens(value: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.findall(r"[a-z0-9]+", folded)


def significant_name_tokens(value: str, contract: dict[str, Any]) -> list[str]:
    stop = set(contract["ignored_name_tokens"])
    return [token for token in ascii_tokens(value) if token not in stop]


def contains_contiguous(candidate: list[str], target: list[str]) -> bool:
    if not target or len(candidate) < len(target):
        return False
    return any(candidate[index : index + len(target)] == target for index in range(len(candidate) - len(target) + 1))


def polygon_box(polygon: Any) -> list[float]:
    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    return [
        round(float(points[:, 0].min()), 4),
        round(float(points[:, 1].min()), 4),
        round(float(points[:, 0].max()), 4),
        round(float(points[:, 1].max()), 4),
    ]


def iou(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def strict_ray(frame: dict[str, Any], projection_protocol: dict[str, Any]) -> dict[str, Any]:
    panorama = frame["panorama"]
    entrance = frame["target"]["entrance_node"]
    lon_lat = entrance["lon_lat"]
    return project_entrance_ray(
        panorama["provider_item"],
        {"id": entrance["id"], "lon": lon_lat[0], "lat": lon_lat[1]},
        projection_protocol,
        downloaded_image_size=tuple(panorama["image_size"]),
    )


def entrance_window(
    image: np.ndarray,
    ray: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    require(width == 2 * height, "IMAGE_NOT_EQUIRECTANGULAR_2_TO_1")
    pixels_per_degree = width / 360.0
    half_width = int(round(float(contract["horizontal_half_fov_degrees"]) * pixels_per_degree))
    top = int(round(height / 2 - float(contract["above_horizon_degrees"]) * pixels_per_degree))
    bottom = int(round(height / 2 + float(contract["below_horizon_degrees"]) * pixels_per_degree))
    require(0 <= top < bottom <= height, "VERTICAL_WINDOW_OUT_OF_BOUNDS")
    center = int(round(float(ray["raw_x_pixels"])))
    columns = np.arange(center - half_width, center + half_width) % width
    crop = np.ascontiguousarray(image[top:bottom, columns, :])
    require(crop.size > 0, "EMPTY_ENTRANCE_WINDOW")
    return crop, {
        "center_raw_x_pixels": round(float(ray["raw_x_pixels"]), 6),
        "horizontal_half_fov_degrees": float(contract["horizontal_half_fov_degrees"]),
        "top_raw_y_pixels": top,
        "bottom_raw_y_pixels_exclusive": bottom,
        "native_crop_width": int(crop.shape[1]),
        "native_crop_height": int(crop.shape[0]),
        "seam_wrapped": center - half_width < 0 or center + half_width > width,
        "pixel_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
    }


def run_ocr(pipeline: PaddleOCR, crop: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    outputs = list(pipeline.predict(crop))
    require(len(outputs) == 1, f"UNEXPECTED_OCR_OUTPUT_COUNT:{len(outputs)}")
    payload = outputs[0].json["res"]
    require(
        len(payload["rec_texts"]) == len(payload["rec_scores"]) == len(payload["rec_polys"]),
        "OCR_ROW_COUNT_MISMATCH",
    )
    rows = [
        {
            "text": str(text),
            "ascii_tokens": ascii_tokens(str(text)),
            "score": round(float(score), 8),
            "box_xyxy": polygon_box(polygon),
        }
        for text, score, polygon in zip(payload["rec_texts"], payload["rec_scores"], payload["rec_polys"])
        if str(text).strip()
    ]
    return rows, {
        "row_count": len(rows),
        "seconds": round(time.perf_counter() - started, 6),
        "text_detection_parameters": payload["text_det_params"],
    }


def match_target_name(
    rows: list[dict[str, Any]],
    entity_name: str,
    contract: dict[str, Any],
) -> dict[str, Any]:
    target = significant_name_tokens(entity_name, contract)
    exact_threshold = float(contract["minimum_exact_name_row_score"])
    exact = [
        row for row in rows
        if row["score"] >= exact_threshold and contains_contiguous(row["ascii_tokens"], target)
    ]
    if exact:
        return {
            "matched": True,
            "tier": "EXACT_SIGNIFICANT_NAME_TOKENS_IN_ONE_ROW",
            "target_tokens": target,
            "witnesses": exact,
        }

    initial_contract = contract["two_initial_brand_signature"]
    if len(target) != 2 or not all(len(token) == 1 and token.isalpha() for token in target):
        return {"matched": False, "tier": "NONE", "target_tokens": target, "witnesses": []}
    candidates = []
    for row in rows:
        joined = "".join(row["ascii_tokens"])
        if (
            row["score"] >= float(initial_contract["minimum_row_score"])
            and int(initial_contract["minimum_canonical_characters"]) <= len(joined) <= int(initial_contract["maximum_canonical_characters"])
            and joined.startswith(target[0])
            and joined.endswith(target[1])
        ):
            candidates.append(row)
    distinct = []
    for candidate in sorted(candidates, key=lambda row: (-row["score"], tuple(row["box_xyxy"]))):
        if all(iou(candidate["box_xyxy"], kept["box_xyxy"]) <= float(initial_contract["maximum_pair_iou"]) for kept in distinct):
            distinct.append(candidate)
    matched = len(distinct) >= int(initial_contract["minimum_distinct_rows"])
    return {
        "matched": matched,
        "tier": "REPEATED_TWO_INITIAL_BRAND_SIGNATURE" if matched else "NONE",
        "target_tokens": target,
        "witnesses": distinct if matched else [],
        "candidate_count_before_spatial_deduplication": len(candidates),
    }


def evaluate_frame(
    pipeline: PaddleOCR,
    frame: dict[str, Any],
    targets: list[dict[str, Any]],
    own_episode_id: str,
    projection_protocol: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    panorama = frame["panorama"]
    path = resolve(panorama["local_path"])
    require(sha256(path) == panorama["image_sha256"], f"IMAGE_HASH_MISMATCH:{frame['key']}")
    require(path.stat().st_size == int(panorama["image_bytes"]), f"IMAGE_BYTES_MISMATCH:{frame['key']}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    require(image is not None, f"IMAGE_DECODE_FAILED:{path}")
    require([image.shape[1], image.shape[0]] == panorama["image_size"], f"IMAGE_SIZE_MISMATCH:{frame['key']}")
    ray = strict_ray(frame, projection_protocol)
    crop, crop_receipt = entrance_window(image, ray, protocol["observation_window"])
    rows, ocr_receipt = run_ocr(pipeline, crop)
    target_matches = {
        target["episode_id"]: match_target_name(rows, target["entity_name"], protocol["lexical_match"])
        for target in targets
    }
    own = target_matches[own_episode_id]
    controls = {
        episode_id: value
        for episode_id, value in target_matches.items()
        if episode_id != own_episode_id
    }
    return {
        "frame_key": frame["key"],
        "panorama_item_id": panorama["provider_item"]["id"],
        "strict_entrance_ray": ray,
        "window": crop_receipt,
        "ocr": ocr_receipt,
        "ocr_rows": rows,
        "own_target_match": own,
        "counterfactual_target_matches": controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = load(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    require(sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")

    frozen = protocol["frozen_inputs"]
    source = load(verify_file(frozen["active_source"]))
    active_result = load(verify_file(frozen["active_ray_result"]))
    verify_file(frozen["active_ray_protocol"])
    verify_file(frozen["active_ray_evaluator"])
    projection_protocol = load(verify_file(frozen["orientation_projection_protocol"]))
    verify_file(frozen["orientation_projection_evaluator"])
    require(active_result["gate"]["passed"], "FROZEN_ACTIVE_RAY_GATE_NOT_MET")
    require(active_result["metrics"]["post_action_ray_authorizations"] == 4, "POST_ACTION_RAY_AUTHORITY_NOT_4")
    require(len(source["episodes"]) == 4, "EPISODE_COUNT_NOT_4")

    runtime = {
        "python": sys.version.split()[0],
        "numpy": importlib.metadata.version("numpy"),
        "opencv-python": importlib.metadata.version("opencv-python"),
        "paddleocr": importlib.metadata.version("paddleocr"),
        "paddlex": importlib.metadata.version("paddlex"),
        "onnxruntime-gpu": importlib.metadata.version("onnxruntime-gpu"),
    }
    require(runtime == protocol["runtime"]["versions"], f"RUNTIME_MISMATCH:{runtime}")
    detection_root = verify_model(protocol["models"]["medium_detection"])
    recognition_root = verify_model(protocol["models"]["medium_recognition"])
    pipeline = PaddleOCR(
        text_detection_model_dir=str(detection_root),
        text_recognition_model_dir=str(recognition_root),
        engine="onnxruntime",
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    targets = [
        {"episode_id": episode["episode_id"], "entity_name": episode["target"]["entity_name"]}
        for episode in source["episodes"]
    ]
    rows = []
    started = time.perf_counter()
    for episode in source["episodes"]:
        start = evaluate_frame(
            pipeline, episode["start"], targets, episode["episode_id"],
            projection_protocol, protocol,
        )
        after = evaluate_frame(
            pipeline, episode["after"], targets, episode["episode_id"],
            projection_protocol, protocol,
        )
        start_match = bool(start["own_target_match"]["matched"])
        after_match = bool(after["own_target_match"]["matched"])
        rows.append({
            "episode_id": episode["episode_id"],
            "target_name": episode["target"]["entity_name"],
            "action": "SIDESTEP_TO_ENTRANCE_FACE",
            "components": {
                "start_lexical_candidate": start,
                "after_lexical_candidate": after,
                "after_entrance_ray_authorized": True,
            },
            "ledger": {
                "current_frame_only_lexical_candidate": after_match,
                "track_carried_lexical_candidate": start_match or after_match,
                "candidate_first_observed_phase": "start" if start_match else ("after" if after_match else None),
                "pre_action_candidate_retained_after_move": start_match and not after_match,
                "new_candidate_acquired_after_move": after_match and not start_match,
                "joint_lexical_candidate_and_authorized_after_ray": (start_match or after_match),
                "portal_ownership_binding_emitted": False,
            },
        })

    count = len(rows)
    after_only = sum(row["ledger"]["current_frame_only_lexical_candidate"] for row in rows)
    fused = sum(row["ledger"]["track_carried_lexical_candidate"] for row in rows)
    retained = sum(row["ledger"]["pre_action_candidate_retained_after_move"] for row in rows)
    acquired = sum(row["ledger"]["new_candidate_acquired_after_move"] for row in rows)
    counterfactual_trials = sum(
        len(frame["counterfactual_target_matches"])
        for row in rows
        for frame in (
            row["components"]["start_lexical_candidate"],
            row["components"]["after_lexical_candidate"],
        )
    )
    counterfactual_matches = sum(
        match["matched"]
        for row in rows
        for frame in (
            row["components"]["start_lexical_candidate"],
            row["components"]["after_lexical_candidate"],
        )
        for match in frame["counterfactual_target_matches"].values()
    )
    metrics = {
        "episodes": count,
        "strict_orientation_images": 2 * count,
        "post_action_authorized_entrance_rays": active_result["metrics"]["post_action_ray_authorizations"],
        "current_after_frame_only_lexical_candidates": after_only,
        "current_after_frame_only_lexical_candidate_rate": round(after_only / count, 6),
        "track_carried_lexical_candidates": fused,
        "track_carried_lexical_candidate_rate": round(fused / count, 6),
        "track_carried_candidate_gain_count": fused - after_only,
        "track_carried_candidate_rate_gain": round((fused - after_only) / count, 6),
        "pre_action_only_candidates_retained": retained,
        "new_candidates_acquired_after_action": acquired,
        "joint_track_candidate_plus_authorized_after_ray": fused,
        "counterfactual_target_trials": counterfactual_trials,
        "counterfactual_target_matches": counterfactual_matches,
        "portal_ownership_bindings_emitted": 0,
        "wall_seconds": round(time.perf_counter() - started, 6),
    }
    gate_spec = protocol["gate"]
    gate = {
        "all_four_consumed_development_episodes_evaluated": count == 4,
        "strict_orientation_8_of_8": metrics["strict_orientation_images"] == 8,
        "frozen_active_ray_recovery_gate_met": active_result["gate"]["passed"],
        "post_action_authorized_entrance_ray_4_of_4": metrics["post_action_authorized_entrance_rays"] == 4,
        "minimum_track_carried_lexical_candidates": fused >= int(gate_spec["minimum_track_carried_lexical_candidates"]),
        "minimum_gain_over_current_after_frame_only": fused - after_only >= int(gate_spec["minimum_gain_over_current_after_frame_only"]),
        "minimum_pre_action_only_candidate_retained": retained >= int(gate_spec["minimum_pre_action_only_candidate_retained"]),
        "minimum_new_candidate_acquired_after_action": acquired >= int(gate_spec["minimum_new_candidate_acquired_after_action"]),
        "zero_counterfactual_target_matches": counterfactual_matches == 0,
        "zero_portal_ownership_bindings": metrics["portal_ownership_bindings_emitted"] == 0,
    }
    gate["passed"] = all(gate.values())
    decision = protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"]
    result = {
        "schema": RESULT_SCHEMA,
        "decision": decision,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "evaluator_sha256": protocol["evaluator"]["sha256"],
        "runtime": runtime,
        "metrics": metrics,
        "gate": gate,
        "rows": rows,
        "evidence_semantics": protocol["evidence_semantics"],
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "metrics": metrics,
        "gate": gate,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
