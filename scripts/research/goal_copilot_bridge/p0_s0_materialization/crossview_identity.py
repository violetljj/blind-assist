"""Conservative P0-S1 same-physical-entrance identity gate.

This module runs after the frozen P0-S0 materializer.  It never upgrades
cross-sequence evidence to strong entrance identity in V1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import PIL
from PIL import Image

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


VERDICTS = {
    "P0_S1_SAME_SEQUENCE_IDENTITY_ESTABLISHED",
    "P0_S1_IDENTITY_RULE_TOO_WEAK",
    "P0_S1_SCHEMA_INADEQUACY",
}
FORBIDDEN_INPUT_KEYS = {
    "visual_audit_disposition",
    "manual_truth",
    "evaluator_truth",
    "silver_quality_class",
    "accepted_identity_pairs",
}


class IdentityError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "rule_id", "same_sequence_required_for_strong_identity",
        "cross_sequence_authority", "min_camera_baseline_m", "max_camera_baseline_m",
        "max_capture_gap_ms", "max_local_wall_position_delta_m", "min_ray_angle_deg",
        "max_ray_angle_deg", "max_aspect_ratio_ratio", "max_physical_height_proxy_ratio",
        "min_appearance_similarity", "appearance_inconsistency_threshold", "appearance_method",
        "required_strong_evidence",
    }
    if set(config) != required or config.get("schema_version") != 1:
        raise IdentityError("frozen config field set or schema mismatch")
    if config.get("same_sequence_required_for_strong_identity") is not True:
        raise IdentityError("V1 strong identity must require same sequence")
    if config.get("cross_sequence_authority") != "SUPPORT_ONLY_NEVER_PRIMARY":
        raise IdentityError("cross-sequence authority escalation")
    return config


def audit_forbidden_keys(value: Any, path: str = "") -> list[str]:
    leaks: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_INPUT_KEYS:
                leaks.append(child_path)
            leaks.extend(audit_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            leaks.extend(audit_forbidden_keys(child, f"{path}[{index}]"))
    return leaks


def _ratio(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return math.inf
    return max(left, right) / min(left, right)


def _bbox_shape(candidate: Mapping[str, Any]) -> tuple[float, float, float]:
    bbox = candidate.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise IdentityError("candidate bbox missing")
    x0, y0, x1, y1 = map(float, bbox)
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise IdentityError("candidate bbox invalid")
    return width, height, width / height


def _crop_signature(path: Path, bbox: Sequence[float]) -> dict[str, Any]:
    if not path.is_file():
        raise IdentityError(f"image missing: {path}")
    with Image.open(path) as source:
        image = source.convert("RGB")
        x0 = max(0, min(image.width - 1, int(math.floor(float(bbox[0])))))
        y0 = max(0, min(image.height - 1, int(math.floor(float(bbox[1])))))
        x1 = max(x0 + 1, min(image.width, int(math.ceil(float(bbox[2])))))
        y1 = max(y0 + 1, min(image.height, int(math.ceil(float(bbox[3])))))
        crop = image.crop((x0, y0, x1, y1))
        histogram_image = crop.resize((64, 64), Image.Resampling.BILINEAR)
        pixels = list(histogram_image.get_flattened_data())
        histograms = [[0] * 8 for _ in range(3)]
        for pixel in pixels:
            for channel in range(3):
                histograms[channel][min(7, pixel[channel] // 32)] += 1
        total = float(len(pixels))
        normalized_histogram = [value / total for channel in histograms for value in channel]
        gray = crop.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        values = list(gray.get_flattened_data())
        dhash_bits = []
        for y in range(8):
            for x in range(8):
                dhash_bits.append(1 if values[y * 9 + x] > values[y * 9 + x + 1] else 0)
    return {"histogram": normalized_histogram, "dhash_bits": dhash_bits}


def appearance_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    left_hist = left["histogram"]
    right_hist = right["histogram"]
    histogram_intersection = sum(min(a, b) for a, b in zip(left_hist, right_hist)) / 3.0
    bits_left = left["dhash_bits"]
    bits_right = right["dhash_bits"]
    dhash_similarity = 1.0 - sum(a != b for a, b in zip(bits_left, bits_right)) / len(bits_left)
    combined = 0.5 * histogram_intersection + 0.5 * dhash_similarity
    return {
        "histogram_intersection": round(histogram_intersection, 9),
        "dhash_similarity": round(dhash_similarity, 9),
        "combined": round(combined, 9),
    }


def _metadata_index(metadata_document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    images = metadata_document.get("images")
    if not isinstance(images, list):
        raise IdentityError("metadata images missing")
    result: dict[str, Mapping[str, Any]] = {}
    for item in images:
        if not isinstance(item, Mapping) or not item.get("id"):
            raise IdentityError("metadata image identity missing")
        result[str(item["id"])] = item
    return result


def _physical_height_proxy(candidate: Mapping[str, Any], metadata: Mapping[str, Any]) -> float:
    _, bbox_height, _ = _bbox_shape(candidate)
    focal = metadata.get("camera_parameters")
    width = metadata.get("width")
    ray_range = candidate.get("ray_range_m")
    if not isinstance(focal, list) or not focal or not isinstance(width, (int, float)) or not isinstance(ray_range, (int, float)):
        raise IdentityError("physical-height proxy inputs missing")
    return bbox_height / float(width) * float(ray_range) / float(focal[0])


def assess_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    left_meta = metadata_by_id.get(str(left.get("frame_id")))
    right_meta = metadata_by_id.get(str(right.get("frame_id")))
    if left_meta is None or right_meta is None:
        raise IdentityError("candidate frame metadata missing")
    if left.get("frame_id") == right.get("frame_id"):
        raise IdentityError("pair must use distinct images")
    same_sequence = left_meta.get("sequence_id") == right_meta.get("sequence_id")
    baseline = materializer.metric_distance_m(
        {"lon": left_meta["coordinates"][0], "lat": left_meta["coordinates"][1]},
        {"lon": right_meta["coordinates"][0], "lat": right_meta["coordinates"][1]},
    )
    capture_gap = abs(int(left_meta["captured_at"]) - int(right_meta["captured_at"]))
    local_delta = materializer.metric_distance_m(left["predicted_entrance_geo"], right["predicted_entrance_geo"])
    ray_angle = materializer.angular_separation_deg(float(left["ray_heading_deg"]), float(right["ray_heading_deg"]))
    _, _, left_aspect = _bbox_shape(left)
    _, _, right_aspect = _bbox_shape(right)
    aspect_ratio_ratio = _ratio(left_aspect, right_aspect)
    physical_height_ratio = _ratio(
        _physical_height_proxy(left, left_meta),
        _physical_height_proxy(right, right_meta),
    )
    left_signature = _crop_signature(Path(str(left_meta["path"])), left["bbox_xyxy"])
    right_signature = _crop_signature(Path(str(right_meta["path"])), right["bbox_xyxy"])
    appearance = appearance_similarity(left_signature, right_signature)
    gates = {
        "same_sequence": same_sequence,
        "same_building_and_anchor": left.get("building_id") == right.get("building_id") and left.get("anchor_id") == right.get("anchor_id"),
        "temporal_adjacency": capture_gap <= int(config["max_capture_gap_ms"]),
        "camera_parallax": float(config["min_camera_baseline_m"]) <= baseline <= float(config["max_camera_baseline_m"]),
        "local_wall_position_consistency": local_delta <= float(config["max_local_wall_position_delta_m"]),
        "ray_angle_diversity": float(config["min_ray_angle_deg"]) <= ray_angle <= float(config["max_ray_angle_deg"]),
        "aspect_ratio_compatibility": aspect_ratio_ratio <= float(config["max_aspect_ratio_ratio"]),
        "physical_height_proxy_compatibility": physical_height_ratio <= float(config["max_physical_height_proxy_ratio"]),
        "appearance_compatibility": appearance["combined"] >= float(config["min_appearance_similarity"]),
    }
    if not same_sequence:
        disposition = "CROSS_SEQUENCE_SUPPORT_ONLY"
        if appearance["combined"] < float(config["appearance_inconsistency_threshold"]):
            disposition = "CROSS_SEQUENCE_APPEARANCE_INCONSISTENT_SUPPORT"
    elif all(gates.values()):
        disposition = "ENTRANCE_IDENTITY_ESTABLISHED"
    else:
        disposition = "SAME_SEQUENCE_IDENTITY_NOT_ESTABLISHED"
    return {
        "candidate_ids": [left["candidate_id"], right["candidate_id"]],
        "frame_ids": [left["frame_id"], right["frame_id"]],
        "sequence_ids": [left_meta["sequence_id"], right_meta["sequence_id"]],
        "disposition": disposition,
        "gates": gates,
        "measurements": {
            "camera_baseline_m": round(baseline, 9),
            "capture_gap_ms": capture_gap,
            "local_wall_position_delta_m": round(local_delta, 9),
            "ray_angle_deg": round(ray_angle, 9),
            "aspect_ratio_ratio": round(aspect_ratio_ratio, 9),
            "physical_height_proxy_ratio": round(physical_height_ratio, 9),
            "appearance": appearance,
        },
    }


def assess_identity(
    bundle: Mapping[str, Any],
    metadata_document: Mapping[str, Any],
    materialization_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    leaks = audit_forbidden_keys({"bundle": bundle, "metadata": metadata_document})
    if leaks:
        return {
            "schema_version": 1,
            "rule_id": config.get("rule_id"),
            "verdict": "P0_S1_SCHEMA_INADEQUACY",
            "failure_codes": ["FORBIDDEN_INPUT_LEAK"],
            "forbidden_input_leaks": leaks,
            "records": [],
            "claim_ceiling": "NO_SCIENTIFIC_VERDICT",
        }
    try:
        metadata_by_id = _metadata_index(metadata_document)
        nominal_primary_ids = {
            item["record_id"] for item in materialization_result.get("results", [])
            if item.get("quality_class") == "SILVER_A_PRIMARY"
        }
        records = []
        established_record_count = 0
        for record in bundle.get("records", []):
            candidates = record.get("candidates")
            if not isinstance(candidates, list):
                raise IdentityError("record candidates missing")
            pairs = []
            for index, left in enumerate(candidates):
                for right in candidates[index + 1:]:
                    pairs.append(assess_pair(left, right, metadata_by_id, config))
            established_pairs = [item for item in pairs if item["disposition"] == "ENTRANCE_IDENTITY_ESTABLISHED"]
            eligible_nominal_primary = record.get("record_id") in nominal_primary_ids
            identity_established = bool(established_pairs) and eligible_nominal_primary
            if identity_established:
                established_record_count += 1
            records.append({
                "record_id": record.get("record_id"),
                "nominal_silver_a_primary": eligible_nominal_primary,
                "wall_associated_candidate_count": sum(
                    item.get("geometry_verified") is True and item.get("map_anchored") is True for item in candidates
                ),
                "pair_count": len(pairs),
                "strong_identity_pair_count": len(established_pairs),
                "entrance_identity_established": identity_established,
                "disposition": "IDENTITY_ESTABLISHED" if identity_established else "IDENTITY_NOT_ESTABLISHED",
                "pairs": pairs,
            })
        verdict = (
            "P0_S1_SAME_SEQUENCE_IDENTITY_ESTABLISHED"
            if established_record_count
            else "P0_S1_IDENTITY_RULE_TOO_WEAK"
        )
        report = {
            "schema_version": 1,
            "rule_id": config["rule_id"],
            "verdict": verdict,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config_sha256": content_sha256(config),
            "bundle_sha256": content_sha256(bundle),
            "materialization_report_sha256": materialization_result.get("report_sha256"),
            "nominal_primary_count": len(nominal_primary_ids),
            "identity_established_primary_count": established_record_count,
            "records": records,
            "runtime_versions": {"python": platform.python_version(), "pillow": PIL.__version__},
            "forbidden_input_leaks": [],
            "claim_ceiling": "CONSUMED_DEVELOPMENT_IDENTITY_MECHANICS_ONLY_NO_FRESH_SILVER_CLAIM",
        }
        report["report_sha256"] = content_sha256(report)
        return report
    except (IdentityError, KeyError, TypeError, ValueError) as error:
        return {
            "schema_version": 1,
            "rule_id": config.get("rule_id"),
            "verdict": "P0_S1_SCHEMA_INADEQUACY",
            "failure_codes": ["IDENTITY_INPUT_SCHEMA_INADEQUATE"],
            "error": str(error),
            "records": [],
            "claim_ceiling": "NO_SCIENTIFIC_VERDICT",
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--materialization-result", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    result = json.loads(args.materialization_result.read_text(encoding="utf-8"))
    config = load_config(args.config)
    first = assess_identity(bundle, metadata, result, config)
    second = assess_identity(bundle, metadata, result, config)
    if canonical_bytes(first) != canonical_bytes(second):
        raise IdentityError("deterministic replay mismatch")
    write_json(args.output, first)
    print(json.dumps({
        "verdict": first["verdict"],
        "nominal_primary_count": first.get("nominal_primary_count", 0),
        "identity_established_primary_count": first.get("identity_established_primary_count", 0),
        "report_sha256": first.get("report_sha256"),
    }, indent=2))
    return 0 if first["verdict"] != "P0_S1_SCHEMA_INADEQUACY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
