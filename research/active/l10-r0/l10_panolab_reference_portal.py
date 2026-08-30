#!/usr/bin/env python3
"""Bind a cross-sequence Panoramax reference field to a ray-consistent portal."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))

from l10_panolab_entrance_ray import project_entrance_ray, projection_gate  # noqa: E402
from l10_panolab_node_credential import (  # noqa: E402
    download_index,
    ensure_image,
    load_json,
    resolve,
    sha256,
    write_json,
)
from named_poi_portal_lattice import PortalLatticeConfig, propose_portal_sets  # noqa: E402
from named_poi_target_support_field import (  # noqa: E402
    SupportBindingState,
    SupportFieldConfig,
    SupportProposal,
    bind_support_field_to_entrances,
    build_target_support_field,
)
from research_backend import (  # noqa: E402
    BackendCandidate,
    Workload,
    runtime_capabilities,
    select_backend,
    torch_observation,
)


PROTOCOL_SCHEMA = "blindassist-l10-panolab-reference-portal-protocol-v1"
SOURCE_SCHEMA = "blindassist-l10-panolab-reference-portal-source-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-reference-portal-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rectilinear_viewport(
    panorama: np.ndarray,
    center_x: float,
    width: int,
    height: int,
    horizontal_fov_degrees: float,
    center_pitch_degrees: float,
) -> np.ndarray:
    """Render a perspective viewport without granting any pixel authority."""
    pano_height, pano_width = panorama.shape[:2]
    focal = 0.5 * width / math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    screen_x = (np.arange(width, dtype=np.float32) + 0.5 - width / 2.0) / focal
    screen_y = -(np.arange(height, dtype=np.float32) + 0.5 - height / 2.0) / focal
    x, y = np.meshgrid(screen_x, screen_y)
    z = np.ones_like(x)
    pitch = math.radians(center_pitch_degrees)
    rotated_y = math.cos(pitch) * y + math.sin(pitch) * z
    rotated_z = -math.sin(pitch) * y + math.cos(pitch) * z
    yaw = np.arctan2(x, rotated_z)
    latitude = np.arctan2(rotated_y, np.sqrt(x * x + rotated_z * rotated_z))
    map_x = np.mod(center_x + yaw / (2.0 * math.pi) * pano_width, pano_width).astype(np.float32)
    map_y = np.clip(pano_height / 2.0 - latitude / math.pi * pano_height, 0, pano_height - 1).astype(np.float32)
    return cv2.remap(
        panorama,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _strict_direct_rows(
    candidate: dict[str, Any], orientation: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for support in candidate["supports"]["direct"]:
        item_id = str(support["item_id"])
        item = candidate["items"][item_id]
        gate = projection_gate(item, orientation)
        if not gate["eligible"]:
            continue
        rows.append(
            {
                "item_id": item_id,
                "collection": str(item["collection"]),
                "distance_m": float(support["first_intersection"]["distance_from_camera_m"]),
                "item": item,
            }
        )
    return rows


def _select_cross_collection_pair(
    rows: list[dict[str, Any]], target_distance_m: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_collection: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_collection.setdefault(row["collection"], []).append(row)
    require(len(by_collection) >= 2, "FEWER_THAN_TWO_STRICT_DIRECT_COLLECTIONS")
    representatives = [
        min(values, key=lambda row: (abs(row["distance_m"] - target_distance_m), row["item_id"]))
        for values in by_collection.values()
    ]
    pair = min(
        combinations(representatives, 2),
        key=lambda values: (
            abs(values[0]["distance_m"] - values[1]["distance_m"]),
            tuple(sorted((values[0]["collection"], values[1]["collection"]))),
            tuple(sorted((values[0]["item_id"], values[1]["item_id"]))),
        ),
    )
    ordered = sorted(pair, key=lambda row: (row["collection"], row["item_id"]))
    return ordered[0], ordered[1]


def _viewport_record(
    role: str,
    row: dict[str, Any],
    candidate: dict[str, Any],
    orientation: dict[str, Any],
    image_root: Path,
    viewport_root: Path,
    known: dict[str, dict[str, Any]],
    viewport_spec: dict[str, Any],
) -> dict[str, Any]:
    image = ensure_image(row["item"], row["item_id"], image_root, known)
    full = cv2.imread(image["path"], cv2.IMREAD_COLOR)
    require(full is not None, f"IMAGE_DECODE_FAILED:{row['item_id']}")
    ray = project_entrance_ray(
        row["item"],
        candidate["main_entrance_node"],
        orientation,
        downloaded_image_size=tuple(image["image_size"]),
    )
    viewport = _rectilinear_viewport(
        full,
        float(ray["raw_x_pixels"]),
        int(viewport_spec["width_pixels"]),
        int(viewport_spec["height_pixels"]),
        float(viewport_spec["horizontal_fov_degrees"]),
        float(viewport_spec["center_pitch_degrees"]),
    )
    viewport_root.mkdir(parents=True, exist_ok=True)
    viewport_path = viewport_root / f"{row['item_id']}.jpg"
    encoded = cv2.imwrite(
        str(viewport_path),
        viewport,
        [cv2.IMWRITE_JPEG_QUALITY, int(viewport_spec["jpeg_quality"])],
    )
    require(encoded, f"VIEWPORT_WRITE_FAILED:{row['item_id']}")
    return {
        "role": role,
        "item_id": row["item_id"],
        "collection": row["collection"],
        "camera_to_entrance_distance_m": round(float(row["distance_m"]), 3),
        "provider_item": row["item"],
        "panorama": image,
        "entrance_ray": ray,
        "viewport": {
            "path": str(viewport_path.resolve()),
            "sha256": sha256(viewport_path),
            "bytes": viewport_path.stat().st_size,
            "image_size": [int(viewport.shape[1]), int(viewport.shape[0])],
            "projected_entrance_x_pixels": float(viewport_spec["projected_entrance_x_pixels"]),
        },
    }


def materialize_source(protocol_path: Path, source_path: Path) -> dict[str, Any]:
    require(not source_path.exists(), f"SOURCE_ALREADY_EXISTS:{source_path}")
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    source_spec = protocol["source"]
    candidates_path = resolve(ROOT, source_spec["candidates_path"])
    orientation_path = resolve(ROOT, source_spec["orientation_protocol_path"])
    require(sha256(candidates_path) == source_spec["candidates_sha256"], "CANDIDATES_HASH_MISMATCH")
    require(
        sha256(orientation_path) == source_spec["orientation_protocol_sha256"],
        "ORIENTATION_PROTOCOL_HASH_MISMATCH",
    )
    candidates_payload = load_json(candidates_path)
    candidates = {int(row["query_index"]): row for row in candidates_payload["candidates"]}
    orientation = load_json(orientation_path)
    selected = [int(value) for value in protocol["cohort"]["selected_candidate_indices"]]
    require(len(selected) == 4 and len(set(selected)) == 4, "COHORT_MUST_HAVE_FOUR_UNIQUE_INDICES")
    excluded = {int(value) for value in protocol["cohort"]["excluded_way_ids"]}
    known = download_index(ROOT, source_spec["download_manifests"])
    asset_root = resolve(ROOT, source_spec["new_asset_root"])
    image_root = asset_root / "images"
    viewport_root = asset_root / "viewports"
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
                "episode_id": f"RP{len(episodes) + 1:02d}",
                "candidate_index": index,
                "target_way": candidate["target_way"],
                "target_entrance_node": candidate["main_entrance_node"],
                "reference": _viewport_record(
                    "REFERENCE",
                    reference,
                    candidate,
                    orientation,
                    image_root,
                    viewport_root,
                    known,
                    protocol["viewport"],
                ),
                "query": _viewport_record(
                    "QUERY",
                    query,
                    candidate,
                    orientation,
                    image_root,
                    viewport_root,
                    known,
                    protocol["viewport"],
                ),
            }
        )
    source = {
        "schema": SOURCE_SCHEMA,
        "status": "METADATA_PAIR_FROZEN_BEFORE_DINO_OR_PORTAL_OUTCOME_ACCESS",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "candidates": str(candidates_path.resolve()),
        "candidates_sha256": sha256(candidates_path),
        "orientation_protocol": str(orientation_path.resolve()),
        "orientation_protocol_sha256": sha256(orientation_path),
        "episode_count": len(episodes),
        "distinct_target_way_count": len({int(row["target_way"]["id"]) for row in episodes}),
        "distinct_reference_collection_count": len({row["reference"]["collection"] for row in episodes}),
        "distinct_query_collection_count": len({row["query"]["collection"] for row in episodes}),
        "pixel_exposure_boundary": "The materializer decoded selected panoramas only to render deterministic ray-centered viewports and compute hashes. No selected pixels were visually inspected, and no DINO field or portal proposal was computed before this source was frozen.",
        "episodes": episodes,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(source_path, source)
    return source


def _load_dino(
    model_path: Path,
    representative_pixels: torch.Tensor,
    receipt_path: Path,
) -> tuple[dict[str, Any], Any]:
    def load(device: str) -> Any:
        return AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)

    cpu_model = load("cpu")
    gpu_model = load("cuda") if torch.cuda.is_available() else None

    def run(model: Any, device: str) -> torch.Tensor:
        with torch.inference_mode():
            return model(pixel_values=representative_pixels.to(device)).last_hidden_state

    cpu = BackendCandidate(
        "l10-panolab-reference-dinov2-cpu",
        "cpu",
        lambda: run(cpu_model, "cpu"),
        lambda output: torch_observation(model=cpu_model, output=output),
    )
    gpu = None
    if gpu_model is not None:
        gpu = BackendCandidate(
            "l10-panolab-reference-dinov2-cuda",
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
    require(selected is not None, "SELECTED_DINO_MODEL_MISSING")
    if selected is gpu_model:
        del cpu_model
    else:
        del gpu_model
    gc.collect()
    return receipt, selected


def _encode_viewports(
    rows: list[tuple[str, Path]],
    processor: Any,
    model: Any,
    device: str,
    expected_grid: int,
) -> dict[str, np.ndarray]:
    images = [Image.open(path).convert("RGB") for _, path in rows]
    pixels = processor(images=images, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        hidden = model(pixel_values=pixels.to(device)).last_hidden_state
    patch_count = int(hidden.shape[1] - 1)
    grid = int(round(math.sqrt(patch_count)))
    require(grid == expected_grid and grid * grid == patch_count, "DINO_NATIVE_PATCH_GRID_MISMATCH")
    patches = F.normalize(hidden[:, 1:], dim=-1).detach().cpu().numpy().astype(np.float32)
    return {key: value for (key, _), value in zip(rows, patches, strict=True)}


def _active_ids(binding: Any) -> set[str]:
    if binding.state == SupportBindingState.COMMIT and binding.selected_proposal_id:
        return {str(binding.selected_proposal_id)}
    if binding.state == SupportBindingState.SET_VALUED:
        return {str(value) for value in binding.candidate_set}
    return set()


def _ray_ids(proposals: list[Any], ray_x: float) -> set[str]:
    return {
        proposal.proposal_id
        for proposal in proposals
        if float(proposal.box_xyxy[0]) <= ray_x <= float(proposal.box_xyxy[2])
    }


def evaluate(protocol_path: Path, source_path: Path, result_path: Path) -> dict[str, Any]:
    require(not result_path.exists(), f"RESULT_ALREADY_EXISTS:{result_path}")
    started = time.perf_counter()
    protocol = load_json(protocol_path)
    source = load_json(source_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    require(source.get("schema") == SOURCE_SCHEMA, "SOURCE_SCHEMA_MISMATCH")
    require(source["protocol_sha256"] == sha256(protocol_path), "SOURCE_PROTOCOL_HASH_MISMATCH")
    episodes = source["episodes"]
    require(len(episodes) == 4, "EPISODE_COUNT_NOT_4")
    require(len({int(row["target_way"]["id"]) for row in episodes}) == 4, "TARGET_WAYS_NOT_UNIQUE")

    model_path = resolve(ROOT, protocol["model"]["dinov2_path"])
    require(sha256(model_path / "model.safetensors") == protocol["model"]["weights_sha256"], "DINO_HASH_MISMATCH")
    viewport_rows: list[tuple[str, Path]] = []
    strict_orientation_images = 0
    cross_collection_episodes = 0
    for episode in episodes:
        cross_collection_episodes += int(episode["reference"]["collection"] != episode["query"]["collection"])
        for role in ("reference", "query"):
            row = episode[role]
            panorama_path = Path(row["panorama"]["path"])
            viewport_path = Path(row["viewport"]["path"])
            require(sha256(panorama_path) == row["panorama"]["sha256"], f"PANORAMA_HASH_MISMATCH:{row['item_id']}")
            require(sha256(viewport_path) == row["viewport"]["sha256"], f"VIEWPORT_HASH_MISMATCH:{row['item_id']}")
            require(row["entrance_ray"]["projection_gate"]["eligible"], f"ORIENTATION_NOT_STRICT:{row['item_id']}")
            strict_orientation_images += 1
            viewport_rows.append((f"{episode['episode_id']}:{role}", viewport_path))

    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    representative = processor(
        images=[Image.open(viewport_rows[0][1]).convert("RGB")], return_tensors="pt"
    )["pixel_values"]
    receipt_path = resolve(ROOT, protocol["source"]["new_asset_root"]) / "execution_backend.json"
    receipt, model = _load_dino(model_path, representative, receipt_path)
    device = str(receipt["selected_device_type"])
    encoded = _encode_viewports(
        viewport_rows,
        processor,
        model,
        device,
        int(protocol["model"]["native_patch_grid"]),
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    target_ids = [str(row["target_entrance_node"]["id"]) for row in episodes]
    references = {
        target_id: [encoded[f"{episode['episode_id']}:reference"]]
        for target_id, episode in zip(target_ids, episodes, strict=True)
    }
    field_config = SupportFieldConfig()
    portal_config = PortalLatticeConfig()
    evaluated = []
    wrong_reference_ray_bindings = 0
    portal_top3_ray_retention = 0
    unique_tokens = 0
    for target_id, episode in zip(target_ids, episodes, strict=True):
        query_path = Path(episode["query"]["viewport"]["path"])
        query_image = cv2.imread(str(query_path), cv2.IMREAD_COLOR)
        require(query_image is not None, f"QUERY_VIEWPORT_DECODE_FAILED:{episode['episode_id']}")
        height, width = query_image.shape[:2]
        ray_x = float(episode["query"]["viewport"]["projected_entrance_x_pixels"])
        portal_rows, portal_diagnostics = propose_portal_sets(query_image, portal_config)
        proposals = [
            SupportProposal(
                row.proposal_id,
                float(row.proposal_score),
                tuple(float(value) for value in row.box_xyxy),
            )
            for row in portal_rows
        ]
        ray_candidates = _ray_ids(portal_rows, ray_x)
        top3_ids = {row.proposal_id for row in portal_rows[:3]}
        top3_retained = bool(ray_candidates & top3_ids)
        portal_top3_ray_retention += int(top3_retained)
        query_patches = encoded[f"{episode['episode_id']}:query"]
        field, field_diagnostics = build_target_support_field(
            target_id,
            query_patches,
            references,
            int(protocol["model"]["native_patch_grid"]),
            field_config,
        )
        binding = bind_support_field_to_entrances(
            field, proposals, width, height, field_config
        )
        active = _active_ids(binding)
        unique = (
            binding.state == SupportBindingState.COMMIT
            and binding.selected_proposal_id in ray_candidates
            and binding.selected_proposal_id in top3_ids
        )
        token = None
        if unique:
            token_payload = {
                "entrance_node_id": int(episode["target_entrance_node"]["id"]),
                "entrance_node_version": int(episode["target_entrance_node"]["version"]),
                "target_way_id": int(episode["target_way"]["id"]),
                "reference_viewport_sha256": episode["reference"]["viewport"]["sha256"],
                "query_viewport_sha256": episode["query"]["viewport"]["sha256"],
                "proposal_id": binding.selected_proposal_id,
                "dinov2_weights_sha256": protocol["model"]["weights_sha256"],
            }
            token = hashlib.sha256(
                json.dumps(token_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            unique_tokens += 1

        wrong_rows = []
        for wrong_id in target_ids:
            if wrong_id == target_id:
                continue
            wrong_field, wrong_diagnostics = build_target_support_field(
                wrong_id,
                query_patches,
                references,
                int(protocol["model"]["native_patch_grid"]),
                field_config,
            )
            wrong_binding = bind_support_field_to_entrances(
                wrong_field, proposals, width, height, field_config
            )
            wrong_active = _active_ids(wrong_binding)
            false_binding_ids = sorted(wrong_active & ray_candidates)
            wrong_reference_ray_bindings += len(false_binding_ids)
            wrong_rows.append(
                {
                    "wrong_entrance_node_id": int(wrong_id),
                    "binding_state": wrong_binding.state.value,
                    "active_ids": sorted(wrong_active),
                    "ray_false_binding_ids": false_binding_ids,
                    "field_diagnostics": wrong_diagnostics,
                }
            )

        evaluated.append(
            {
                "episode_id": episode["episode_id"],
                "target_way_id": int(episode["target_way"]["id"]),
                "target_entrance_node_id": int(target_id),
                "reference_item_id": episode["reference"]["item_id"],
                "query_item_id": episode["query"]["item_id"],
                "cross_collection": episode["reference"]["collection"] != episode["query"]["collection"],
                "portal_top3_ray_retained": top3_retained,
                "ray_candidate_ids": sorted(ray_candidates),
                "target_binding_state": binding.state.value,
                "target_active_ids": sorted(active),
                "unique_reference_bound_ray_portal": unique,
                "reference_portal_token_sha256": token,
                "portal_diagnostics": portal_diagnostics,
                "portal_proposals": [asdict(row) for row in portal_rows[:10]],
                "target_binding": asdict(binding),
                "target_field_diagnostics": field_diagnostics,
                "wrong_reference_replays": wrong_rows,
            }
        )

    metrics = {
        "episode_count": len(episodes),
        "strict_orientation_images": strict_orientation_images,
        "cross_collection_episodes": cross_collection_episodes,
        "portal_top3_ray_retention": portal_top3_ray_retention,
        "unique_target_reference_portal_tokens": unique_tokens,
        "wrong_reference_replays": len(episodes) * (len(episodes) - 1),
        "wrong_reference_ray_bindings": wrong_reference_ray_bindings,
    }
    gate_spec = protocol["gate"]
    gate = {
        "strict_orientation_images_required": strict_orientation_images == int(gate_spec["strict_orientation_images_required"]),
        "cross_collection_episodes_required": cross_collection_episodes == int(gate_spec["cross_collection_episodes_required"]),
        "portal_top3_ray_retention_required": portal_top3_ray_retention == int(gate_spec["portal_top3_ray_retention_required"]),
        "unique_target_reference_portal_tokens_required": unique_tokens == int(gate_spec["unique_target_reference_portal_tokens_required"]),
        "wrong_reference_ray_bindings_maximum": wrong_reference_ray_bindings <= int(gate_spec["wrong_reference_ray_bindings_maximum"]),
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
        "execution_backend_receipt": receipt,
        "execution_backend_receipt_sha256": sha256(receipt_path),
        "configuration": {
            "portal_lattice": asdict(portal_config),
            "support_field": asdict(field_config),
            "viewport": protocol["viewport"],
        },
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
    protocol_path = args.protocol.resolve()
    source_path = args.source.resolve()
    if args.mode == "materialize":
        source = materialize_source(protocol_path, source_path)
        print(
            json.dumps(
                {
                    "source": str(source_path),
                    "episode_count": source["episode_count"],
                    "distinct_target_way_count": source["distinct_target_way_count"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    require(args.result is not None, "--result is required for evaluation")
    result = evaluate(protocol_path, source_path, args.result.resolve())
    print(
        json.dumps(
            {"decision": result["decision"], "metrics": result["metrics"], "gate": result["gate"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
