#!/usr/bin/env python3
"""Join exact OSM entrance credentials to strict Panoramax entrance rays.

The source stage selects views from geometry and public metadata only.  OCR is
constructed only after the source JSON has been written, so pixel outcomes
cannot influence the cohort or view order.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from rapidocr import RapidOCR

from l10_panolab_entrance_ray import project_entrance_ray, projection_gate


PROTOCOL_SCHEMA = "blindassist-l10-panolab-node-credential-protocol-v1"
SOURCE_SCHEMA = "blindassist-l10-panolab-node-credential-source-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-node-credential-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", folded.lower()))


def aliases_from_node(node: dict[str, Any], fields: Iterable[str]) -> list[dict[str, str]]:
    tags = node.get("tags") or {}
    aliases: list[dict[str, str]] = []
    seen: set[str] = set()
    for field in fields:
        raw = tags.get(field)
        if raw is None:
            continue
        values = [str(raw)]
        if field == "name":
            values.extend(re.split(r"\s+-\s+|\s*[;/]\s*", str(raw)))
        for value in values:
            value = value.strip()
            key = normalize(value)
            if len(key) < 1 or key in seen or (field == "name" and len(key) < 4):
                continue
            seen.add(key)
            aliases.append({"field": field, "value": value, "normalized": key})
    return aliases


def download_index(root: Path, manifests: list[str]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for value in manifests:
        path = resolve(root, value)
        if not path.exists():
            continue
        for row in load_json(path).get("images", []):
            image_path = Path(row["path"])
            if image_path.exists():
                index[str(row["item_id"])] = {
                    "path": str(image_path.resolve()),
                    "sha256": sha256(image_path),
                    "bytes": image_path.stat().st_size,
                    "image_size": row.get("image_size"),
                    "provenance": str(path.resolve()),
                }
    return index


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    require(image is not None, f"IMAGE_DECODE_FAILED:{path}")
    height, width = image.shape[:2]
    return width, height


def ensure_image(item: dict[str, Any], item_id: str, asset_root: Path, known: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if item_id in known:
        row = dict(known[item_id])
        width, height = image_size(Path(row["path"]))
        row["image_size"] = [width, height]
        return row

    href = ((item.get("assets") or {}).get("hd") or {}).get("href")
    require(isinstance(href, str) and href.startswith("https://"), f"HD_ASSET_MISSING:{item_id}")
    asset_root.mkdir(parents=True, exist_ok=True)
    target = asset_root / f"{item_id}.jpg"
    temporary: Path | None = None
    try:
        if not target.exists():
            descriptor, name = tempfile.mkstemp(prefix=f"{item_id}-", suffix=".partial", dir=asset_root)
            os.close(descriptor)
            temporary = Path(name)
            request = urllib.request.Request(href, headers={"User-Agent": "BlindAssist-L10-Research/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            os.replace(temporary, target)
            temporary = None
        width, height = image_size(target)
        return {
            "path": str(target.resolve()),
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
            "image_size": [width, height],
            "provenance": href,
        }
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def representative_views(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-row["distance_m"], row["item_id"]))
    if len(ordered) <= count:
        return ordered
    positions = [0, (len(ordered) - 1) // 2, len(ordered) - 1]
    chosen = [ordered[index] for index in positions[:count]]
    return list({row["item_id"]: row for row in chosen}.values())


def materialize_source(protocol_path: Path, source_path: Path) -> dict[str, Any]:
    root = repo_root()
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    source_contract = protocol["source"]
    candidates_path = resolve(root, source_contract["candidates_path"])
    orientation_path = resolve(root, source_contract["orientation_protocol_path"])
    require(sha256(candidates_path) == source_contract["candidates_sha256"], "CANDIDATES_HASH_MISMATCH")
    require(sha256(orientation_path) == source_contract["orientation_protocol_sha256"], "ORIENTATION_PROTOCOL_HASH_MISMATCH")
    candidates_payload = load_json(candidates_path)
    orientation = load_json(orientation_path)
    candidates = {int(row["query_index"]): row for row in candidates_payload["candidates"]}
    selected_indices = [int(value) for value in protocol["cohort"]["selected_candidate_indices"]]
    require(len(set(selected_indices)) == len(selected_indices), "DUPLICATE_CANDIDATE_INDEX")
    known = download_index(root, source_contract["download_manifests"])
    asset_root = resolve(root, source_contract["new_asset_root"])
    excluded_ways = {int(value) for value in protocol["cohort"]["excluded_prior_active_recovery_target_way_ids"]}
    credential_fields = protocol["cohort"]["credential_fields"]
    episodes: list[dict[str, Any]] = []
    candidate_by_way: dict[int, list[dict[str, Any]]] = {}
    for index in selected_indices:
        require(index in candidates, f"CANDIDATE_NOT_FOUND:{index}")
        candidate = candidates[index]
        way_id = int(candidate["target_way"]["id"])
        require(way_id not in excluded_ways, f"PRIOR_ACTIVE_RECOVERY_WAY_REUSED:{way_id}")
        aliases = aliases_from_node(candidate["main_entrance_node"], credential_fields)
        require(aliases, f"ENTRANCE_CREDENTIAL_MISSING:{index}")
        candidate_by_way.setdefault(way_id, []).append(candidate)

        direct_rows: list[dict[str, Any]] = []
        for support in candidate["supports"]["direct"]:
            item_id = str(support["item_id"])
            item = candidate["items"][item_id]
            gate = projection_gate(item, orientation)
            if not gate["eligible"]:
                continue
            distance = float(support["first_intersection"]["distance_from_camera_m"])
            direct_rows.append({"item_id": item_id, "item": item, "distance_m": distance})
        require(direct_rows, f"NO_STRICT_DIRECT_VIEW:{index}")
        picked = representative_views(direct_rows, int(protocol["cohort"]["views_per_entrance"]))
        views: list[dict[str, Any]] = []
        for order, row in enumerate(picked, start=1):
            image = ensure_image(row["item"], row["item_id"], asset_root, known)
            projection = project_entrance_ray(
                row["item"],
                candidate["main_entrance_node"],
                orientation,
                downloaded_image_size=tuple(image["image_size"]),
            )
            views.append(
                {
                    "view_order": order,
                    "item_id": row["item_id"],
                    "sequence_id": row["item"]["collection"],
                    "camera_to_entrance_distance_m": row["distance_m"],
                    "provider_item": row["item"],
                    "image": image,
                    "target_entrance_ray": projection,
                }
            )
        episodes.append(
            {
                "episode_id": f"NC{len(episodes) + 1:02d}",
                "candidate_index": index,
                "target_way": candidate["target_way"],
                "target_entrance_node": candidate["main_entrance_node"],
                "credential_aliases": aliases,
                "views": views,
            }
        )

    for episode in episodes:
        way_id = int(episode["target_way"]["id"])
        competitors = []
        for candidate in candidate_by_way[way_id]:
            node_id = int(candidate["main_entrance_node"]["id"])
            if node_id == int(episode["target_entrance_node"]["id"]):
                continue
            competitors.append(
                {
                    "entrance_node": candidate["main_entrance_node"],
                    "credential_aliases": aliases_from_node(candidate["main_entrance_node"], credential_fields),
                }
            )
        episode["same_way_competing_entrances"] = competitors

    source = {
        "schema": SOURCE_SCHEMA,
        "status": "METADATA_FROZEN_BEFORE_EVALUATION_OUTCOME_ACCESS",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "candidates": str(candidates_path.resolve()),
        "candidates_sha256": sha256(candidates_path),
        "orientation_protocol": str(orientation_path.resolve()),
        "orientation_protocol_sha256": sha256(orientation_path),
        "episode_count": len(episodes),
        "distinct_target_way_count": len({int(row["target_way"]["id"]) for row in episodes}),
        "view_count": sum(len(row["views"]) for row in episodes),
        "episodes": episodes,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(source_path, source)
    return source


def wrap_crop(image: np.ndarray, center_x: float, protocol: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    height, width = image.shape[:2]
    observation = protocol["observation"]
    fov = float(observation["ray_crop_horizontal_fov_degrees"])
    crop_width = max(8, int(round(width * fov / 360.0)))
    start = int(round(center_x - crop_width / 2.0))
    indices = np.mod(np.arange(start, start + crop_width), width)
    top = int(round(height * float(observation["ray_crop_top_fraction"])))
    bottom = int(round(height * float(observation["ray_crop_bottom_fraction"])))
    crop = image[top:bottom, indices]
    output_width = int(observation["ray_crop_output_width_pixels"])
    scale = output_width / crop.shape[1]
    output_height = max(1, int(round(crop.shape[0] * scale)))
    crop = cv2.resize(crop, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    return crop, {"input_start_x": start, "input_width": crop_width, "output_width": output_width, "fov_degrees": fov}


def ocr_rows(engine: RapidOCR, image: np.ndarray) -> list[dict[str, Any]]:
    result = engine(image)
    boxes = result.boxes if result.boxes is not None else []
    texts = result.txts if result.txts is not None else []
    scores = result.scores if result.scores is not None else []
    require(len(boxes) == len(texts) == len(scores), "OCR_OUTPUT_ALIGNMENT_MISMATCH")
    rows: list[dict[str, Any]] = []
    for box, text, score in zip(boxes, texts, scores, strict=True):
        array = np.asarray(box, dtype=np.float64)
        rows.append(
            {
                "text": str(text),
                "normalized": normalize(str(text)),
                "confidence": float(score),
                "center_x": float(array[:, 0].mean()),
                "center_y": float(array[:, 1].mean()),
            }
        )
    return rows


def alias_similarity(alias: dict[str, str], text: str) -> float:
    target = alias["normalized"]
    observed = normalize(text)
    if not target or not observed:
        return 0.0
    if alias["field"] == "addr:housenumber" and target.isdigit():
        return 1.0 if target in re.findall(r"\d+", observed) else 0.0
    target_tokens = set(target.split())
    observed_tokens = set(observed.split())
    overlap = len(target_tokens & observed_tokens)
    token_f1 = 2.0 * overlap / (len(target_tokens) + len(observed_tokens)) if overlap else 0.0
    containment = 1.0 if target in observed or observed in target else 0.0
    sequence = difflib.SequenceMatcher(None, target, observed).ratio()
    return max(token_f1, containment, sequence)


def best_match(aliases: list[dict[str, str]], rows: list[dict[str, Any]], *, max_offset: float | None = None, fov: float | None = None, width: float | None = None) -> dict[str, Any]:
    best = {"score": 0.0, "alias": None, "text": None, "confidence": None, "offset_degrees": None}
    for row in rows:
        offset = None
        if max_offset is not None:
            require(fov is not None and width is not None, "LOCAL_MATCH_GEOMETRY_MISSING")
            offset = (float(row["center_x"]) / width - 0.5) * fov
            if abs(offset) > max_offset:
                continue
        for alias in aliases:
            score = alias_similarity(alias, row["text"])
            key = (score, float(row["confidence"]), -abs(offset or 0.0), str(row["text"]))
            current = (float(best["score"]), float(best["confidence"] or 0.0), -abs(float(best["offset_degrees"] or 0.0)), str(best["text"] or ""))
            if key > current:
                best = {
                    "score": round(score, 6),
                    "alias": alias,
                    "text": row["text"],
                    "confidence": round(float(row["confidence"]), 6),
                    "offset_degrees": round(offset, 6) if offset is not None else None,
                }
    return best


def evaluate(protocol_path: Path, source_path: Path, result_path: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    source = load_json(source_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    require(source.get("schema") == SOURCE_SCHEMA, "SOURCE_SCHEMA_MISMATCH")
    require(source["protocol_sha256"] == sha256(protocol_path), "SOURCE_PROTOCOL_HASH_MISMATCH")
    root = repo_root()
    models = resolve(root, protocol["ocr"]["model_root"])
    for filename, expected in protocol["ocr"]["model_sha256"].items():
        require(sha256(models / filename) == expected, f"OCR_MODEL_HASH_MISMATCH:{filename}")

    engine = RapidOCR(
        params={
            "Global.model_root_dir": str(models),
            "Global.log_level": "error",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    threshold = float(protocol["credential"]["minimum_text_similarity"])
    margin = float(protocol["credential"]["minimum_exact_entrance_margin"])
    max_offset = float(protocol["observation"]["maximum_credential_offset_degrees"])
    fov = float(protocol["observation"]["ray_crop_horizontal_fov_degrees"])
    full_width = int(protocol["observation"]["full_frame_width_pixels"])
    started = time.perf_counter()
    episode_rows: list[dict[str, Any]] = []

    for episode in source["episodes"]:
        view_rows: list[dict[str, Any]] = []
        for view in episode["views"]:
            image_path = Path(view["image"]["path"])
            require(sha256(image_path) == view["image"]["sha256"], f"IMAGE_HASH_MISMATCH:{view['item_id']}")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            require(image is not None, f"IMAGE_DECODE_FAILED:{view['item_id']}")
            height, width = image.shape[:2]
            resized_height = int(round(height * full_width / width))
            full = cv2.resize(image, (full_width, resized_height), interpolation=cv2.INTER_AREA)
            full_rows = ocr_rows(engine, full)
            crop, crop_geometry = wrap_crop(image, float(view["target_entrance_ray"]["raw_x_pixels"]), protocol)
            ray_rows = ocr_rows(engine, crop)
            target_scene = best_match(episode["credential_aliases"], full_rows)
            target_ray = best_match(
                episode["credential_aliases"],
                ray_rows,
                max_offset=max_offset,
                fov=fov,
                width=float(crop.shape[1]),
            )
            competitor_aliases = [
                alias
                for competitor in episode["same_way_competing_entrances"]
                for alias in competitor["credential_aliases"]
            ]
            wrong_ray = best_match(
                competitor_aliases,
                ray_rows,
                max_offset=max_offset,
                fov=fov,
                width=float(crop.shape[1]),
            ) if competitor_aliases else {"score": 0.0, "alias": None, "text": None, "confidence": None, "offset_degrees": None}
            exact = target_ray["score"] >= threshold and target_ray["score"] - wrong_ray["score"] >= margin
            wrong = wrong_ray["score"] >= threshold and wrong_ray["score"] > target_ray["score"]
            scene = target_scene["score"] >= threshold
            view_rows.append(
                {
                    "view_order": view["view_order"],
                    "item_id": view["item_id"],
                    "camera_to_entrance_distance_m": view["camera_to_entrance_distance_m"],
                    "full_frame_ocr_count": len(full_rows),
                    "ray_crop_ocr_count": len(ray_rows),
                    "scene_match": target_scene,
                    "ray_match": target_ray,
                    "strongest_competing_entrance_match": wrong_ray,
                    "scene_credential_proof": scene,
                    "exact_entrance_credential_proof": exact,
                    "wrong_entrance_proof": wrong,
                    "ray_crop": {
                        **crop_geometry,
                        "output_height": int(crop.shape[0]),
                    },
                    "ocr_texts": [row["text"] for row in ray_rows],
                }
            )

        first = view_rows[0]
        best = max(
            view_rows,
            key=lambda row: (
                float(row["ray_match"]["score"]) - float(row["strongest_competing_entrance_match"]["score"]),
                float(row["ray_match"]["score"]),
                -int(row["view_order"]),
            ),
        )
        multi_exact = bool(best["exact_entrance_credential_proof"])
        first_proof = next((int(row["view_order"]) for row in view_rows if row["exact_entrance_credential_proof"]), None)
        episode_rows.append(
            {
                "episode_id": episode["episode_id"],
                "candidate_index": episode["candidate_index"],
                "target_way_id": int(episode["target_way"]["id"]),
                "target_name": episode["target_way"]["tags"].get("name"),
                "target_entrance_node_id": int(episode["target_entrance_node"]["id"]),
                "credential_aliases": episode["credential_aliases"],
                "same_way_competitor_count": len(episode["same_way_competing_entrances"]),
                "scene_single_view_proof": bool(first["scene_credential_proof"]),
                "ray_single_view_proof": bool(first["exact_entrance_credential_proof"]),
                "ray_multiview_proof": multi_exact,
                "ray_multiview_wrong_entrance_proof": any(row["wrong_entrance_proof"] for row in view_rows),
                "first_proof_view_order": first_proof,
                "selected_best_view_order": best["view_order"],
                "selected_best_target_score": best["ray_match"]["score"],
                "selected_best_competitor_score": best["strongest_competing_entrance_match"]["score"],
                "views": view_rows,
            }
        )

    count = len(episode_rows)
    scene_proofs = sum(row["scene_single_view_proof"] for row in episode_rows)
    ray_single = sum(row["ray_single_view_proof"] for row in episode_rows)
    ray_multi = sum(row["ray_multiview_proof"] for row in episode_rows)
    wrong = sum(row["ray_multiview_wrong_entrance_proof"] for row in episode_rows)
    same_building = [row for row in episode_rows if row["same_way_competitor_count"] > 0]
    same_building_proofs = sum(row["ray_multiview_proof"] for row in same_building)
    metrics = {
        "episode_count": count,
        "distinct_target_way_count": len({row["target_way_id"] for row in episode_rows}),
        "view_count": sum(len(row["views"]) for row in episode_rows),
        "scene_single_view_correct_proofs": scene_proofs,
        "ray_single_view_correct_exact_entrance_proofs": ray_single,
        "ray_multiview_correct_exact_entrance_proofs": ray_multi,
        "ray_multiview_coverage": round(ray_multi / count, 6),
        "ray_localization_uplift_over_scene_single": ray_single - scene_proofs,
        "multiview_uplift_over_ray_single": ray_multi - ray_single,
        "wrong_entrance_proofs": wrong,
        "same_building_distinct_credential_episode_count": len(same_building),
        "same_building_distinct_credential_proofs": same_building_proofs,
        "mean_views_to_first_proof": round(
            sum(row["first_proof_view_order"] for row in episode_rows if row["first_proof_view_order"] is not None) / ray_multi,
            6,
        ) if ray_multi else None,
    }
    gate_spec = protocol["gate"]
    gate = {
        "ray_multiview_correct_exact_entrance_proofs_minimum": ray_multi >= int(gate_spec["ray_multiview_correct_exact_entrance_proofs_minimum"]),
        "wrong_entrance_proofs_maximum": wrong <= int(gate_spec["wrong_entrance_proofs_maximum"]),
        "same_building_distinct_credential_proofs_minimum": same_building_proofs >= int(gate_spec["same_building_distinct_credential_proofs_minimum"]),
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
        "runtime": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "rapidocr": importlib.metadata.version("rapidocr"),
            "opencv": cv2.__version__,
            "onnxruntime_providers": __import__("onnxruntime").get_available_providers(),
            "selected_backend": "CPUExecutionProvider",
            "selection_reason": protocol["ocr"]["selection_reason"],
            "wall_s": round(time.perf_counter() - started, 3),
        },
        "metrics": metrics,
        "gate": gate,
        "episodes": episode_rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    source_path = args.source.resolve()
    result_path = args.result.resolve()
    source = materialize_source(protocol_path, source_path)
    print(json.dumps({"stage": "source_frozen", "episodes": source["episode_count"], "views": source["view_count"], "source": str(source_path)}, ensure_ascii=False))
    result = evaluate(protocol_path, source_path, result_path)
    print(json.dumps({"stage": "evaluated", "decision": result["decision"], "metrics": result["metrics"], "result": str(result_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
