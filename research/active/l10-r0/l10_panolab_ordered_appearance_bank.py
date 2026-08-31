#!/usr/bin/env python3
"""Replay an entrance-ray-aligned ordered appearance memory bank.

This is a consumed-cohort Development mechanism probe.  It reuses the frozen
CLIP+DINO+mutual-patch score from the named-POI facade fingerprint and changes
only the query representation: an after-frame descriptor becomes a max-pooled
start->after bank carried across a verified provider action edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor

import l10_panolab_track_lexical_fresh_panel as panel
import named_poi_facade_fingerprint as fingerprint
from l10_panolab_entrance_ray import project_entrance_ray


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-ordered-appearance-bank-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-ordered-appearance-bank-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    require(path.is_file(), f"MISSING_INPUT:{path}")
    require(sha256(path) == spec["sha256"], f"HASH_MISMATCH:{path}")
    if "bytes" in spec:
        require(path.stat().st_size == int(spec["bytes"]), f"BYTE_COUNT_MISMATCH:{path}")
    return path


def image_payload(path: str | Path, digest: str, byte_count: int, size: list[int]) -> tuple[Path, np.ndarray]:
    image_path = resolve(path)
    require(image_path.is_file(), f"MISSING_IMAGE:{image_path}")
    require(sha256(image_path) == digest, f"IMAGE_HASH_MISMATCH:{image_path}")
    require(image_path.stat().st_size == int(byte_count), f"IMAGE_BYTES_MISMATCH:{image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    require(image is not None, f"IMAGE_DECODE_FAILED:{image_path}")
    require([image.shape[1], image.shape[0]] == list(size), f"IMAGE_SIZE_MISMATCH:{image_path}")
    return image_path, image


def build_frame(episode: dict[str, Any], phase: str, image_receipt: dict[str, Any]) -> dict[str, Any]:
    entrance = episode["main_entrance_node"]
    return {
        "target": {
            "entrance_node": {
                "id": entrance["id"],
                "lon_lat": [entrance["lon"], entrance["lat"]],
            }
        },
        "panorama": {
            "image_size": image_receipt["image_size"],
            "provider_item": episode[f"{phase}_item"],
        },
    }


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
    image: np.ndarray, ray: dict[str, Any], contract: dict[str, Any]
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


def write_crop(
    key: str,
    target_id: str,
    role: str,
    item_id: str,
    image_spec: dict[str, Any],
    ray: dict[str, Any],
    observation_window: dict[str, Any],
    crop_root: Path,
) -> tuple[fingerprint.ImageRow, dict[str, Any]]:
    source_path, image = image_payload(
        image_spec["path"], image_spec["sha256"], image_spec["bytes"], image_spec["image_size"]
    )
    crop, window = entrance_window(image, ray, observation_window)
    crop_path = crop_root / f"{key}.jpg"
    require(cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]), f"CROP_WRITE_FAILED:{crop_path}")
    digest = sha256(crop_path)
    row = fingerprint.ImageRow(
        key=key,
        target_id=target_id,
        role=role,
        path=crop_path,
        sha256=digest,
        commons_file=item_id,
    )
    receipt = {
        "key": key,
        "target_id": target_id,
        "role": role,
        "item_id": item_id,
        "source_path": str(source_path),
        "source_sha256": image_spec["sha256"],
        "crop_path": str(crop_path),
        "crop_sha256": digest,
        "crop_bytes": crop_path.stat().st_size,
        "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
        "window": window,
    }
    return row, receipt


def portal_reference(source: dict[str, Any], way_id: int, item_id: str) -> dict[str, Any]:
    episode = next(row for row in source["episodes"] if int(row["target_way"]["id"]) == way_id)
    match = next(row for row in (episode["reference"], episode["query"]) if row["item_id"] == item_id)
    return {
        "item_id": item_id,
        "collection": match["collection"],
        "distance_m": float(match["camera_to_entrance_distance_m"]),
        "image": match["panorama"],
        "ray": match["entrance_ray"],
    }


def node_reference(source: dict[str, Any], way_id: int, item_id: str) -> dict[str, Any]:
    matches = [
        view
        for episode in source["episodes"]
        if int(episode["target_way"]["id"]) == way_id
        for view in episode["views"]
        if view["item_id"] == item_id
    ]
    require(len(matches) == 1, f"REFERENCE_ITEM_NOT_UNIQUE:{way_id}:{item_id}:{len(matches)}")
    match = matches[0]
    return {
        "item_id": item_id,
        "collection": match["sequence_id"],
        "distance_m": float(match["camera_to_entrance_distance_m"]),
        "image": match["image"],
        "ray": match["target_entrance_ray"],
    }


def appearance_score(
    query: fingerprint.ImageRow,
    references: list[fingerprint.ImageRow],
    encoded: dict[str, dict[str, np.ndarray]],
    patch_grid: int,
    weights: dict[str, float],
) -> dict[str, Any]:
    query_features = encoded[query.key]
    debug = []
    for reference in references:
        reference_features = encoded[reference.key]
        geometry = fingerprint._patch_geometry(
            query_features["patches"], reference_features["patches"], patch_grid
        )
        debug.append(
            {
                "reference": reference.key,
                "clip": float(query_features["clip"] @ reference_features["clip"]),
                "dino": float(query_features["dino"] @ reference_features["dino"]),
                **geometry,
            }
        )
    clip_score = max(row["clip"] for row in debug)
    dino_score = max(row["dino"] for row in debug)
    geometry_score = max(row["score"] for row in debug)
    score = (
        weights["clip"] * clip_score
        + weights["dino"] * dino_score
        + weights["mutual_patch_geometry"] * geometry_score
    )
    return {
        "score": float(score),
        "max_clip": float(clip_score),
        "max_dino": float(dino_score),
        "max_mutual_patch_geometry": float(geometry_score),
        "references": debug,
    }


def prediction(scores: dict[str, dict[str, Any]], truth: str) -> dict[str, Any]:
    ordered = sorted(scores, key=lambda target: (-float(scores[target]["score"]), target))
    predicted = ordered[0]
    truth_score = float(scores[truth]["score"])
    strongest_wrong = max(float(scores[target]["score"]) for target in scores if target != truth)
    return {
        "prediction": predicted,
        "correct": predicted == truth,
        "truth_score": truth_score,
        "strongest_wrong_score": strongest_wrong,
        "truth_margin": truth_score - strongest_wrong,
        "ranking": ordered,
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

    frozen = protocol["inputs"]
    selection = load(verify(frozen["fresh_selection"]))
    materialization = load(verify(frozen["fresh_materialization"]))
    portal_source = load(verify(frozen["reference_portal_source"]))
    node_source = load(verify(frozen["node_credential_source"]))
    orientation = load(verify(frozen["orientation_projection_protocol"]))
    require(materialization["selection_sha256"] == frozen["fresh_selection"]["sha256"], "SELECTION_LINK_MISMATCH")

    selected = {row["episode_id"]: row for row in selection["episodes"]}
    image_index = {(row["episode_id"], row["phase"]): row for row in materialization["images"]}
    target_specs = protocol["target_roster"]
    target_ids = [row["episode_id"] for row in target_specs]
    require(len(target_ids) == len(set(target_ids)) == 3, "TARGET_ROSTER_NOT_THREE_UNIQUE_EPISODES")
    require(all(target_id in selected for target_id in target_ids), "TARGET_NOT_IN_FRESH_SELECTION")

    output_root = resolve(protocol["output_root"])
    require(not output_root.exists(), f"OUTPUT_ROOT_ALREADY_EXISTS:{output_root}")
    crop_root = output_root / "crops"
    crop_root.mkdir(parents=True)
    all_rows: list[fingerprint.ImageRow] = []
    crop_receipts = []
    query_rows: dict[str, dict[str, fingerprint.ImageRow]] = {}
    reference_rows: dict[str, list[fingerprint.ImageRow]] = {}
    reference_meta = []
    query_item_ids = {row["item_id"] for row in materialization["images"]}

    for target_spec in target_specs:
        target_id = target_spec["episode_id"]
        episode = selected[target_id]
        require(int(episode["target_way"]["id"]) == int(target_spec["target_way_id"]), f"WAY_ID_MISMATCH:{target_id}")
        require(panel.reciprocal(episode["start_item"], episode["after_item"], episode["provider_link_relation"]), f"RECIPROCAL_ACTION_FAILURE:{target_id}")
        require(episode["start_classification"]["stratum"] in {"TARGET_SELF_OCCLUDED", "OTHER_BUILDING_OCCLUDED"}, f"START_NOT_OCCLUDED:{target_id}")
        require(episode["after_classification"]["stratum"] == "DIRECT", f"AFTER_NOT_DIRECT:{target_id}")

        query_rows[target_id] = {}
        for phase in ("start", "after"):
            receipt = image_index[(target_id, phase)]
            frame = build_frame(episode, phase, receipt)
            ray = strict_ray(frame, orientation)
            row, crop_receipt = write_crop(
                f"{target_id}_{phase}",
                target_id,
                phase,
                receipt["item_id"],
                {
                    "path": receipt["path"],
                    "sha256": receipt["sha256"],
                    "bytes": receipt["bytes"],
                    "image_size": receipt["image_size"],
                },
                ray,
                protocol["observation_window"],
                crop_root,
            )
            query_rows[target_id][phase] = row
            all_rows.append(row)
            crop_receipts.append(crop_receipt)

        reference_rows[target_id] = []
        for index, reference_spec in enumerate(target_spec["references"], start=1):
            if reference_spec["source"] == "reference_portal":
                reference = portal_reference(portal_source, int(target_spec["target_way_id"]), reference_spec["item_id"])
            elif reference_spec["source"] == "node_credential":
                reference = node_reference(node_source, int(target_spec["target_way_id"]), reference_spec["item_id"])
            else:
                raise ValueError(f"UNKNOWN_REFERENCE_SOURCE:{reference_spec['source']}")
            require(reference["item_id"] not in query_item_ids, f"REFERENCE_QUERY_ITEM_OVERLAP:{reference['item_id']}")
            row, crop_receipt = write_crop(
                f"{target_id}_ref{index}",
                target_id,
                "reference",
                reference["item_id"],
                reference["image"],
                reference["ray"],
                protocol["observation_window"],
                crop_root,
            )
            reference_rows[target_id].append(row)
            all_rows.append(row)
            crop_receipts.append(crop_receipt)
            reference_meta.append(
                {
                    "target_id": target_id,
                    "item_id": reference["item_id"],
                    "collection": reference["collection"],
                    "query_collection": episode["sequence_id"],
                    "collection_disjoint": reference["collection"] != episode["sequence_id"],
                    "camera_to_entrance_distance_m": reference["distance_m"],
                }
            )
        require(len(reference_rows[target_id]) == 2, f"REFERENCE_COUNT_NOT_TWO:{target_id}")

    model_spec = protocol["models"]
    clip_path = resolve(model_spec["clip"]["path"])
    dino_path = resolve(model_spec["dinov2"]["path"])
    require(sha256(clip_path / "pytorch_model.bin") == model_spec["clip"]["weights_sha256"], "CLIP_MODEL_HASH_MISMATCH")
    require(sha256(dino_path / "model.safetensors") == model_spec["dinov2"]["weights_sha256"], "DINO_MODEL_HASH_MISMATCH")
    clip_processor = AutoProcessor.from_pretrained(clip_path, local_files_only=True)
    dino_processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    representative_image = Image.open(all_rows[0].path).convert("RGB")
    representative = (
        clip_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
        dino_processor(images=[representative_image], return_tensors="pt")["pixel_values"],
    )
    backend_path = output_root / "backend_receipt.json"
    backend, models = fingerprint._select_backend(clip_path, dino_path, representative, backend_path)
    device = str(backend["selected_device_type"])
    patch_grid = int(model_spec["dinov2"]["patch_grid"])
    started = time.perf_counter()
    encoded = fingerprint._encode_images(
        all_rows, models, clip_processor, dino_processor, device, patch_grid, int(protocol["execution"]["batch_size"])
    )

    episode_rows = []
    phase_correct = {"start": 0, "after": 0}
    bank_correct = 0
    after_margins = []
    bank_margins = []
    for target_id in target_ids:
        phase_scores: dict[str, dict[str, dict[str, Any]]] = {}
        phase_predictions = {}
        for phase in ("start", "after"):
            phase_scores[phase] = {
                candidate_id: appearance_score(
                    query_rows[target_id][phase],
                    reference_rows[candidate_id],
                    encoded,
                    patch_grid,
                    protocol["score"]["weights"],
                )
                for candidate_id in target_ids
            }
            phase_predictions[phase] = prediction(phase_scores[phase], target_id)
            phase_correct[phase] += int(phase_predictions[phase]["correct"])
        bank_scores = {
            candidate_id: {
                "score": max(
                    float(phase_scores["start"][candidate_id]["score"]),
                    float(phase_scores["after"][candidate_id]["score"]),
                ),
                "selected_phase": max(
                    ("start", "after"),
                    key=lambda phase: (float(phase_scores[phase][candidate_id]["score"]), phase),
                ),
            }
            for candidate_id in target_ids
        }
        bank_prediction = prediction(bank_scores, target_id)
        bank_correct += int(bank_prediction["correct"])
        after_margins.append(float(phase_predictions["after"]["truth_margin"]))
        bank_margins.append(float(bank_prediction["truth_margin"]))
        episode_rows.append(
            {
                "episode_id": target_id,
                "target_name": selected[target_id]["target_name"],
                "target_way_id": selected[target_id]["target_way"]["id"],
                "query_sequence_id": selected[target_id]["sequence_id"],
                "action": "SIDESTEP_TO_ENTRANCE_FACE",
                "reciprocal_action_receipt": True,
                "geometry": {
                    "start": selected[target_id]["start_classification"]["stratum"],
                    "after": selected[target_id]["after_classification"]["stratum"],
                    "after_entrance_ray_authorized": True,
                },
                "start": {"scores": phase_scores["start"], **phase_predictions["start"]},
                "after_only_baseline": {"scores": phase_scores["after"], **phase_predictions["after"]},
                "ordered_bank_successor": {"scores": bank_scores, **bank_prediction},
                "portal_ownership_binding": None,
            }
        )

    mean_after_margin = float(np.mean(after_margins))
    mean_bank_margin = float(np.mean(bank_margins))
    metrics = {
        "source_fresh_episodes_available": len(selection["episodes"]),
        "episodes_with_prior_reference_bank": len(target_ids),
        "source_coverage_rate": len(target_ids) / len(selection["episodes"]),
        "cities": len({selected[target_id]["source_city"] for target_id in target_ids}),
        "query_frames": len(target_ids) * 2,
        "reference_views": sum(len(rows) for rows in reference_rows.values()),
        "exact_query_reference_item_overlap": 0,
        "collection_disjoint_reference_views": sum(row["collection_disjoint"] for row in reference_meta),
        "start_only_top1": phase_correct["start"],
        "after_only_top1": phase_correct["after"],
        "ordered_bank_top1": bank_correct,
        "ordered_bank_top1_gain_over_after_only": bank_correct - phase_correct["after"],
        "after_only_mean_truth_margin": mean_after_margin,
        "ordered_bank_mean_truth_margin": mean_bank_margin,
        "ordered_bank_mean_margin_gain": mean_bank_margin - mean_after_margin,
        "wrong_goal_trials": len(target_ids) * (len(target_ids) - 1),
        "after_only_wrong_top1": len(target_ids) - phase_correct["after"],
        "ordered_bank_wrong_top1": len(target_ids) - bank_correct,
        "portal_ownership_bindings_emitted": 0,
        "model_seconds": time.perf_counter() - started,
    }
    gate = {
        "minimum_two_of_three_ordered_bank_top1": bank_correct >= 2,
        "minimum_one_top1_gain_over_after_only": bank_correct - phase_correct["after"] >= 1,
        "ordered_bank_wrong_top1_not_increased": len(target_ids) - bank_correct <= len(target_ids) - phase_correct["after"],
        "zero_portal_ownership_bindings": True,
    }
    gate["passed"] = all(gate.values())
    result = {
        "schema": RESULT_SCHEMA,
        "decision": protocol["decision_names"]["gate_met" if gate["passed"] else "gate_not_met"],
        "authority": "POSTHOC_CONSUMED_DEVELOPMENT_MECHANISM_ONLY",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "execution_backend": backend,
        "backend_receipt_sha256": sha256(backend_path),
        "score_contract": protocol["score"],
        "reference_metadata": reference_meta,
        "crop_receipts": crop_receipts,
        "metrics": metrics,
        "gate": gate,
        "rows": episode_rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(output_path), "decision": result["decision"], "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
