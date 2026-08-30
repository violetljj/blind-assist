#!/usr/bin/env python3
"""Transport an annotated exact-portal patch across Panoramax sequences."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForKeypointMatching


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))

from l10_panolab_node_credential import (  # noqa: E402
    download_index,
    load_json,
    resolve,
    sha256,
    write_json,
)
from l10_panolab_reference_portal import (  # noqa: E402
    _select_cross_collection_pair,
    _strict_direct_rows,
    _viewport_record,
)
from research_backend import (  # noqa: E402
    BackendCandidate,
    Workload,
    runtime_capabilities,
    select_backend,
    torch_observation,
)


PROTOCOL_SCHEMA = "blindassist-l10-panolab-exact-portal-patch-protocol-v1"
SOURCE_SCHEMA = "blindassist-l10-panolab-exact-portal-patch-source-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-exact-portal-patch-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def materialize_source(protocol_path: Path, source_path: Path) -> dict[str, Any]:
    require(not source_path.exists(), f"SOURCE_ALREADY_EXISTS:{source_path}")
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    spec = protocol["source"]
    candidates_path = resolve(ROOT, spec["candidates_path"])
    orientation_path = resolve(ROOT, spec["orientation_protocol_path"])
    require(sha256(candidates_path) == spec["candidates_sha256"], "CANDIDATES_HASH_MISMATCH")
    require(sha256(orientation_path) == spec["orientation_protocol_sha256"], "ORIENTATION_HASH_MISMATCH")
    candidates_payload = load_json(candidates_path)
    candidates = {int(row["query_index"]): row for row in candidates_payload["candidates"]}
    orientation = load_json(orientation_path)
    selected = [int(value) for value in protocol["cohort"]["selected_candidate_indices"]]
    require(len(selected) == 3 and len(set(selected)) == 3, "COHORT_MUST_HAVE_THREE_UNIQUE_INDICES")
    excluded = {int(value) for value in protocol["cohort"]["excluded_way_ids"]}
    known = download_index(ROOT, spec["download_manifests"])
    asset_root = resolve(ROOT, spec["new_asset_root"])
    episodes = []
    for index in selected:
        require(index in candidates, f"CANDIDATE_NOT_FOUND:{index}")
        candidate = candidates[index]
        way_id = int(candidate["target_way"]["id"])
        require(way_id not in excluded, f"EXCLUDED_WAY_REUSED:{way_id}")
        reference, query = _select_cross_collection_pair(
            _strict_direct_rows(candidate, orientation),
            float(protocol["cohort"]["target_camera_distance_m"]),
        )
        episodes.append(
            {
                "episode_id": f"EP{len(episodes) + 1:02d}",
                "candidate_index": index,
                "target_way": candidate["target_way"],
                "target_entrance_node": candidate["main_entrance_node"],
                "reference": _viewport_record(
                    "REFERENCE",
                    reference,
                    candidate,
                    orientation,
                    asset_root / "images",
                    asset_root / "viewports",
                    known,
                    protocol["viewport"],
                ),
                "query": _viewport_record(
                    "QUERY",
                    query,
                    candidate,
                    orientation,
                    asset_root / "images",
                    asset_root / "viewports",
                    known,
                    protocol["viewport"],
                ),
            }
        )
    source = {
        "schema": SOURCE_SCHEMA,
        "status": "METADATA_PAIR_FROZEN_BEFORE_REFERENCE_OR_QUERY_ANNOTATION",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256_at_materialization": sha256(protocol_path),
        "candidates": str(candidates_path.resolve()),
        "candidates_sha256": sha256(candidates_path),
        "orientation_protocol": str(orientation_path.resolve()),
        "orientation_protocol_sha256": sha256(orientation_path),
        "episode_count": len(episodes),
        "distinct_target_way_count": len({int(row["target_way"]["id"]) for row in episodes}),
        "pixel_exposure_boundary": "Only deterministic ray-centered viewport rendering and hashing occurred before source freeze. No selected viewport was visually inspected and no matcher output was computed.",
        "episodes": episodes,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(source_path, source)
    return source


def _annotation_index(
    manifest: dict[str, Any], source_path: Path, expected_role: str
) -> dict[str, dict[str, Any]]:
    require(manifest.get("source_sha256") == sha256(source_path), f"{expected_role}_SOURCE_HASH_MISMATCH")
    require(manifest.get("role") == expected_role, f"{expected_role}_ROLE_MISMATCH")
    rows = {str(row["episode_id"]): row for row in manifest["episodes"]}
    require(len(rows) == 3, f"{expected_role}_ANNOTATION_COUNT_NOT_3")
    return rows


def _validate_box(row: dict[str, Any], width: int, height: int, ray_x: float) -> list[float]:
    require(row.get("status") == "ADMITTED", f"ANNOTATION_NOT_ADMITTED:{row.get('episode_id')}")
    box = [float(value) for value in row["portal_box_xyxy"]]
    require(len(box) == 4, "PORTAL_BOX_LENGTH_NOT_4")
    require(0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height, "PORTAL_BOX_OUT_OF_RANGE")
    require(box[0] <= ray_x <= box[2], "PORTAL_BOX_DOES_NOT_CONTAIN_RAY")
    return box


def _expanded_crop(image: Image.Image, box: list[float], expansion: float) -> tuple[Image.Image, list[float], list[float]]:
    center_x, center_y = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    half_width = 0.5 * (box[2] - box[0]) * expansion
    half_height = 0.5 * (box[3] - box[1]) * expansion
    crop_box = [
        max(0.0, center_x - half_width),
        max(0.0, center_y - half_height),
        min(float(image.width), center_x + half_width),
        min(float(image.height), center_y + half_height),
    ]
    integer = [int(math.floor(crop_box[0])), int(math.floor(crop_box[1])), int(math.ceil(crop_box[2])), int(math.ceil(crop_box[3]))]
    crop = image.crop(tuple(integer))
    portal_in_crop = [
        box[0] - integer[0],
        box[1] - integer[1],
        box[2] - integer[0],
        box[3] - integer[1],
    ]
    return crop, [float(value) for value in integer], portal_in_crop


def _load_matcher(model_path: Path, representative: dict[str, torch.Tensor], receipt_path: Path) -> tuple[dict[str, Any], Any]:
    def load(device: str) -> Any:
        return AutoModelForKeypointMatching.from_pretrained(model_path, local_files_only=True).eval().to(device)

    cpu_model = load("cpu")
    gpu_model = load("cuda") if torch.cuda.is_available() else None

    def run(model: Any, device: str) -> Any:
        values = {key: value.to(device) for key, value in representative.items()}
        with torch.inference_mode():
            return model(**values)

    cpu = BackendCandidate(
        "l10-panolab-efficientloftr-cpu",
        "cpu",
        lambda: run(cpu_model, "cpu"),
        lambda output: torch_observation(model=cpu_model, output=output),
    )
    gpu = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "l10-panolab-efficientloftr-cuda",
            "cuda",
            lambda: run(gpu_model, "cuda"),
            lambda output: torch_observation(model=gpu_model, output=output),
            torch.cuda.synchronize,
        )
    receipt = select_backend(
        Workload.MODEL_INFERENCE,
        cpu=cpu,
        gpu=gpu,
        cpu_reason="ACCELERATOR_UNAVAILABLE" if gpu is None else None,
        record_path=receipt_path,
        warmups=0,
        repeats=1,
        capabilities=runtime_capabilities(),
    )
    selected = gpu_model if receipt["selected_device_type"] == "cuda" else cpu_model
    require(selected is not None, "SELECTED_MATCHER_MISSING")
    if selected is gpu_model:
        del cpu_model
    else:
        del gpu_model
    gc.collect()
    return receipt, selected


def _match(
    processor: Any,
    model: Any,
    device: str,
    reference: Image.Image,
    query: Image.Image,
    score_threshold: float,
) -> dict[str, np.ndarray]:
    inputs = processor([reference, query], return_tensors="pt")
    values = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**values)
    processed = processor.post_process_keypoint_matching(
        outputs,
        [[(reference.height, reference.width), (query.height, query.width)]],
        threshold=score_threshold,
    )[0]
    return {
        "keypoints0": processed["keypoints0"].detach().cpu().numpy().astype(np.float32),
        "keypoints1": processed["keypoints1"].detach().cpu().numpy().astype(np.float32),
        "scores": processed["matching_scores"].detach().cpu().numpy().astype(np.float32),
    }


def _transport(
    matches: dict[str, np.ndarray],
    portal_in_crop: list[float],
    query_size: tuple[int, int],
    minimum_matches: int,
    reprojection_pixels: float,
    minimum_inliers: int,
) -> dict[str, Any]:
    count = int(len(matches["scores"]))
    row: dict[str, Any] = {
        "match_count": count,
        "mean_score": float(matches["scores"].mean()) if count else 0.0,
        "homography_valid": False,
        "homography_inliers": 0,
        "predicted_portal_quad_xy": None,
        "predicted_portal_box_xyxy": None,
    }
    if count < minimum_matches:
        return row
    homography, mask = cv2.findHomography(
        matches["keypoints0"],
        matches["keypoints1"],
        cv2.RANSAC,
        reprojection_pixels,
    )
    if homography is None or mask is None:
        return row
    inliers = int(mask.reshape(-1).sum())
    row["homography_inliers"] = inliers
    if inliers < minimum_inliers:
        return row
    x1, y1, x2, y2 = portal_in_crop
    corners = np.asarray([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(corners, homography)[0]
    if not np.isfinite(projected).all():
        return row
    width, height = query_size
    projected[:, 0] = np.clip(projected[:, 0], 0, width)
    projected[:, 1] = np.clip(projected[:, 1], 0, height)
    box = [
        float(projected[:, 0].min()),
        float(projected[:, 1].min()),
        float(projected[:, 0].max()),
        float(projected[:, 1].max()),
    ]
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return row
    row.update(
        {
            "homography_valid": True,
            "predicted_portal_quad_xy": projected.round(3).tolist(),
            "predicted_portal_box_xyxy": [round(value, 3) for value in box],
        }
    )
    return row


def _iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def evaluate(protocol_path: Path, source_path: Path, result_path: Path) -> dict[str, Any]:
    require(not result_path.exists(), f"RESULT_ALREADY_EXISTS:{result_path}")
    started = time.perf_counter()
    protocol = load_json(protocol_path)
    source = load_json(source_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    require(source.get("schema") == SOURCE_SCHEMA and len(source["episodes"]) == 3, "SOURCE_SCHEMA_OR_COUNT_MISMATCH")
    reference_path = resolve(ROOT, protocol["source"]["reference_annotations"])
    query_audit_path = resolve(ROOT, protocol["source"]["query_source_audit"])
    require(sha256(reference_path) == protocol["source"]["reference_annotations_sha256"], "REFERENCE_ANNOTATION_HASH_MISMATCH")
    require(sha256(query_audit_path) == protocol["source"]["query_source_audit_sha256"], "QUERY_SOURCE_AUDIT_HASH_MISMATCH")
    reference_annotations = _annotation_index(load_json(reference_path), source_path, "REFERENCE_RUNTIME_INPUT")
    query_source_audit = _annotation_index(load_json(query_audit_path), source_path, "QUERY_SOURCE_ADMISSION")

    episodes = source["episodes"]
    strict_orientation_images = sum(
        int(episode[role]["entrance_ray"]["projection_gate"]["eligible"])
        for episode in episodes
        for role in ("reference", "query")
    )
    cross_collection_episodes = sum(
        int(episode["reference"]["collection"] != episode["query"]["collection"])
        for episode in episodes
    )
    reference_source_admitted = sum(
        reference_annotations[str(episode["episode_id"])].get("status") == "ADMITTED"
        for episode in episodes
    )
    query_source_admitted = sum(
        query_source_audit[str(episode["episode_id"])].get("status") == "ADMITTED"
        for episode in episodes
    )
    joint_source_admission = sum(
        reference_annotations[str(episode["episode_id"])].get("status") == "ADMITTED"
        and query_source_audit[str(episode["episode_id"])].get("status") == "ADMITTED"
        for episode in episodes
    )
    if reference_source_admitted < 3 or query_source_admitted < 3:
        model_path = resolve(ROOT, protocol["matcher"]["path"])
        require(
            sha256(model_path / "model.safetensors") == protocol["matcher"]["weights_sha256"],
            "MATCHER_HASH_MISMATCH",
        )
        metrics = {
            "episode_count": len(episodes),
            "reference_source_admitted": reference_source_admitted,
            "query_source_admitted": query_source_admitted,
            "source_admission": joint_source_admission,
            "strict_orientation_images": strict_orientation_images,
            "cross_collection_episodes": cross_collection_episodes,
            "target_portal_transfers": 0,
            "wrong_reference_replays": 0,
            "wrong_reference_false_bindings": 0,
            "matcher_calls": 0,
        }
        spec = protocol["gate"]
        gate = {
            "source_admission_required": joint_source_admission == int(spec["source_admission_required"]),
            "strict_orientation_images_required": strict_orientation_images == int(spec["strict_orientation_images_required"]),
            "cross_collection_episodes_required": cross_collection_episodes == int(spec["cross_collection_episodes_required"]),
            "target_portal_transfers_required": False,
            "wrong_reference_false_bindings_maximum": True,
            "source_and_model_hashes_verified": True,
        }
        result = {
            "schema": RESULT_SCHEMA,
            "decision": protocol["decision_names"]["source_not_evaluable"],
            "stop_reason": "SOURCE_NOT_EVALUABLE_REFERENCE_OR_QUERY_PORTAL_NOT_UNIQUELY_VISIBLE_NO_MATCHER_CALL",
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": sha256(protocol_path),
            "source": str(source_path.resolve()),
            "source_sha256": sha256(source_path),
            "reference_annotations_sha256": sha256(reference_path),
            "query_source_audit_sha256": sha256(query_audit_path),
            "query_truth_status": protocol["source"]["query_truth_status"],
            "execution_backend_receipt": None,
            "metrics": metrics,
            "gate": gate,
            "wall_s": round(time.perf_counter() - started, 3),
            "claim_boundary": protocol["claim_boundary"],
        }
        write_json(result_path, result)
        return result

    reference_inputs: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        viewport_path = Path(episode["reference"]["viewport"]["path"])
        require(sha256(viewport_path) == episode["reference"]["viewport"]["sha256"], f"REFERENCE_VIEWPORT_HASH_MISMATCH:{episode_id}")
        image = Image.open(viewport_path).convert("RGB")
        ray_x = float(episode["reference"]["viewport"]["projected_entrance_x_pixels"])
        box = _validate_box(reference_annotations[episode_id], image.width, image.height, ray_x)
        crop, crop_box, portal_in_crop = _expanded_crop(
            image, box, float(protocol["reference_crop"]["context_expansion"])
        )
        reference_inputs[episode_id] = {
            "image": crop,
            "crop_box_xyxy": crop_box,
            "portal_box_in_crop_xyxy": portal_in_crop,
        }

    query_images: dict[str, Image.Image] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        path = Path(episode["query"]["viewport"]["path"])
        require(sha256(path) == episode["query"]["viewport"]["sha256"], f"QUERY_VIEWPORT_HASH_MISMATCH:{episode_id}")
        query_images[episode_id] = Image.open(path).convert("RGB")

    model_path = resolve(ROOT, protocol["matcher"]["path"])
    require(sha256(model_path / "model.safetensors") == protocol["matcher"]["weights_sha256"], "MATCHER_HASH_MISMATCH")
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    first_id = str(episodes[0]["episode_id"])
    representative = processor(
        [reference_inputs[first_id]["image"], query_images[first_id]], return_tensors="pt"
    )
    receipt_path = resolve(ROOT, protocol["source"]["new_asset_root"]) / "execution_backend.json"
    receipt, model = _load_matcher(model_path, representative, receipt_path)
    device = str(receipt["selected_device_type"])

    predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for query_episode in episodes:
        query_id = str(query_episode["episode_id"])
        predictions[query_id] = {}
        query_image = query_images[query_id]
        for reference_episode in episodes:
            reference_id = str(reference_episode["episode_id"])
            reference = reference_inputs[reference_id]
            matches = _match(
                processor,
                model,
                device,
                reference["image"],
                query_image,
                float(protocol["matcher"]["postprocess_score_threshold"]),
            )
            predictions[query_id][reference_id] = _transport(
                matches,
                reference["portal_box_in_crop_xyxy"],
                (query_image.width, query_image.height),
                int(protocol["matcher"]["minimum_matches"]),
                float(protocol["matcher"]["homography_ransac_reprojection_pixels"]),
                int(protocol["matcher"]["minimum_homography_inliers"]),
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Evaluator-private query truth is intentionally loaded only after every prediction exists.
    query_truth_value = protocol["source"].get("query_truth")
    query_truth_hash = protocol["source"].get("query_truth_sha256")
    require(isinstance(query_truth_value, str) and isinstance(query_truth_hash, str), "QUERY_TRUTH_NOT_FROZEN")
    query_truth_path = resolve(ROOT, query_truth_value)
    require(sha256(query_truth_path) == query_truth_hash, "QUERY_TRUTH_HASH_MISMATCH")
    query_truth = _annotation_index(load_json(query_truth_path), source_path, "QUERY_EVALUATOR_TRUTH")
    evaluated = []
    source_admission = 0
    target_transfers = 0
    wrong_false_bindings = 0
    target_iou_minimum = float(protocol["evaluation"]["target_iou_minimum"])
    wrong_iou = float(protocol["evaluation"]["wrong_reference_false_binding_iou"])
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        query_image = query_images[episode_id]
        ray_x = float(episode["query"]["viewport"]["projected_entrance_x_pixels"])
        truth_box = _validate_box(query_truth[episode_id], query_image.width, query_image.height, ray_x)
        source_admission += 1
        target = predictions[episode_id][episode_id]
        target_box = target["predicted_portal_box_xyxy"]
        target_iou = _iou(target_box, truth_box) if target_box else 0.0
        contains_ray = bool(target_box and target_box[0] <= ray_x <= target_box[2])
        success = bool(target["homography_valid"] and target_iou >= target_iou_minimum and contains_ray)
        target_transfers += int(success)
        wrong_rows = []
        for reference_episode in episodes:
            reference_id = str(reference_episode["episode_id"])
            if reference_id == episode_id:
                continue
            wrong = predictions[episode_id][reference_id]
            box = wrong["predicted_portal_box_xyxy"]
            overlap = _iou(box, truth_box) if box else 0.0
            false_binding = bool(wrong["homography_valid"] and overlap >= wrong_iou)
            wrong_false_bindings += int(false_binding)
            wrong_rows.append(
                {
                    "reference_episode_id": reference_id,
                    "iou_with_target_truth": round(overlap, 6),
                    "false_binding": false_binding,
                    **wrong,
                }
            )
        evaluated.append(
            {
                "episode_id": episode_id,
                "target_way_id": int(episode["target_way"]["id"]),
                "target_entrance_node_id": int(episode["target_entrance_node"]["id"]),
                "query_truth_box_xyxy": truth_box,
                "target_iou": round(target_iou, 6),
                "target_prediction_contains_ray": contains_ray,
                "target_portal_transfer": success,
                "target_prediction": target,
                "reference_crop_box_xyxy": reference_inputs[episode_id]["crop_box_xyxy"],
                "wrong_reference_replays": wrong_rows,
            }
        )

    metrics = {
        "episode_count": len(episodes),
        "source_admission": source_admission,
        "strict_orientation_images": strict_orientation_images,
        "cross_collection_episodes": cross_collection_episodes,
        "target_portal_transfers": target_transfers,
        "wrong_reference_replays": len(episodes) * (len(episodes) - 1),
        "wrong_reference_false_bindings": wrong_false_bindings,
    }
    spec = protocol["gate"]
    gate = {
        "source_admission_required": source_admission == int(spec["source_admission_required"]),
        "strict_orientation_images_required": strict_orientation_images == int(spec["strict_orientation_images_required"]),
        "cross_collection_episodes_required": cross_collection_episodes == int(spec["cross_collection_episodes_required"]),
        "target_portal_transfers_required": target_transfers == int(spec["target_portal_transfers_required"]),
        "wrong_reference_false_bindings_maximum": wrong_false_bindings <= int(spec["wrong_reference_false_bindings_maximum"]),
        "source_and_model_hashes_verified": True,
    }
    passed = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["met" if passed else "not_met"],
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "source": str(source_path.resolve()),
        "source_sha256": sha256(source_path),
        "reference_annotations_sha256": sha256(reference_path),
        "query_truth_sha256": sha256(query_truth_path),
        "execution_backend_receipt": receipt,
        "execution_backend_receipt_sha256": sha256(receipt_path),
        "metrics": metrics,
        "gate": gate,
        "episodes": evaluated,
        "wall_s": round(time.perf_counter() - started, 3),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("materialize", "evaluate"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.mode == "materialize":
        source = materialize_source(args.protocol.resolve(), args.source.resolve())
        print(json.dumps({"source": str(args.source.resolve()), "episode_count": source["episode_count"]}, indent=2))
        return
    require(args.result is not None, "--result is required for evaluation")
    result = evaluate(args.protocol.resolve(), args.source.resolve(), args.result.resolve())
    print(json.dumps({"decision": result["decision"], "metrics": result["metrics"], "gate": result["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
