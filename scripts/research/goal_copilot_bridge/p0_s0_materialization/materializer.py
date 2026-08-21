"""Fail-closed P0-S0 GoalGrounding-Silver materialization mechanics.

The module consumes source-normalized records.  It never calls a detector or
teacher and it never treats a cluster as multiview evidence.  Network/source
acquisition is deliberately kept outside the admission engine so the same
normalized bundle can be replayed byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "BA-P0-GOAL-GROUNDING-SILVER-V1"
UPSTREAM_COMMIT = "3d3b85244b1a1ec2ba05a997d56d000936cc554a"
VERDICTS = {
    "P0_S0_MATERIALIZATION_CANARY_PASS",
    "P0_S0_PASS_WITH_COVERAGE_LIMITATION",
    "P0_S0_PROTOCOL_OR_SCHEMA_INADEQUACY",
    "P0_S0_SOURCE_OR_LICENSE_BLOCKED",
}
FAILURE_CODES = {
    "NO_VALID_MAP_ANCHOR",
    "TARGET_BUILDING_CROSSWALK_AMBIGUOUS",
    "INSUFFICIENT_GEOMETRY_SUPPORT",
    "INSUFFICIENT_MULTIVIEW_SUPPORT",
    "EVIDENCE_CONFLICT",
    "PROVENANCE_INCOMPLETE",
    "LICENSE_METADATA_INCOMPLETE",
    "EVALUATOR_LINEAGE_OVERLAP",
    "TARGET_VISIBILITY_NOT_IDENTIFIABLE",
    "ENTRANCE_RELATION_NOT_IDENTIFIABLE",
    "SCHEMA_INADEQUACY",
    "ANCESTRY_OVERLAP",
    "MAPILLARY_ACCESS_TOKEN_MISSING",
    "MANDATORY_CANDIDATE_GENERATOR_NOT_AUTHORIZED",
}
PROTECTED_PROVIDER_KEYS = {
    "exact_osm_entrance_coordinate",
    "osm_entrance_id",
    "osm_entrance_ids",
    "target_building_crosswalk",
    "target_building_id",
    "positive_candidate_ids",
    "quality_class",
    "quality_record",
    "evaluator_only",
}


class MaterializationError(ValueError):
    """A normalized record cannot be interpreted under the frozen contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MaterializationError(f"{path} must be a finite number")
    return float(value)


def _point(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, Mapping) or set(value) != {"lon", "lat"}:
        raise MaterializationError(f"{path} must contain exactly lon/lat")
    lon, lat = _number(value["lon"], f"{path}.lon"), _number(value["lat"], f"{path}.lat")
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise MaterializationError(f"{path} is outside WGS84 bounds")
    return lon, lat


def metric_distance_m(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    """Small-area equirectangular metric distance; never compares raw degrees."""
    lon1, lat1 = _point(left, "left")
    lon2, lat2 = _point(right, "right")
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    x = math.radians(lon2 - lon1) * math.cos(mean_lat)
    y = math.radians(lat2 - lat1)
    return 6_371_008.8 * math.hypot(x, y)


def angular_separation_deg(left: float, right: float) -> float:
    delta = abs((left - right) % 360.0)
    return min(delta, 360.0 - delta)


def _missing_text(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def provenance_complete(record: Mapping[str, Any]) -> tuple[bool, bool]:
    sources = record.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"mapillary", "overture", "osm"}:
        return False, False
    provenance_ok = True
    license_ok = True
    for source in sources.values():
        if not isinstance(source, Mapping):
            return False, False
        for key in ("source_name", "snapshot_or_release", "record_ids", "retrieved_at", "content_sha256"):
            if key not in source or source[key] in (None, "", []):
                provenance_ok = False
        if _missing_text(source.get("license")) or _missing_text(source.get("attribution")):
            license_ok = False
    return provenance_ok, license_ok


def _candidate_by_id(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    candidates = record.get("candidates", [])
    if not isinstance(candidates, list):
        raise MaterializationError("candidates must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(candidates):
        if not isinstance(item, Mapping) or _missing_text(item.get("candidate_id")):
            raise MaterializationError(f"candidates[{index}] lacks candidate_id")
        candidate_id = str(item["candidate_id"])
        if candidate_id in result:
            raise MaterializationError(f"duplicate candidate_id: {candidate_id}")
        result[candidate_id] = item
    return result


def candidate_mechanics_complete(item: Mapping[str, Any]) -> bool:
    bbox = item.get("bbox_xyxy")
    provenance = item.get("generator_provenance")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x0, y0, x1, y1 = (_number(value, "candidate.bbox_xyxy") for value in bbox)
    except MaterializationError:
        return False
    if not (x0 < x1 and y0 < y1):
        return False
    if not isinstance(provenance, Mapping):
        return False
    return all(not _missing_text(provenance.get(key)) for key in (
        "provider_id", "model_id", "model_version", "config_sha256", "input_sha256", "candidate_source", "lineage_group"
    ))


def candidate_shape_valid(item: Mapping[str, Any]) -> bool:
    required = {
        "candidate_id", "frame_id", "building_id", "anchor_id", "bbox_xyxy", "predicted_entrance_geo",
        "ray_heading_deg", "ray_range_m", "candidate_anchor_distance_m", "second_anchor_margin_m",
        "geometry_verified", "map_anchored",
    }
    return required <= set(item)


def candidate_passes_geometry(item: Mapping[str, Any]) -> bool:
    if not candidate_mechanics_complete(item):
        return False
    try:
        return (
            item.get("geometry_verified") is True
            and item.get("map_anchored") is True
            and _number(item.get("ray_range_m"), "candidate.ray_range_m") <= 60.0
            and _number(item.get("candidate_anchor_distance_m"), "candidate.candidate_anchor_distance_m") <= 3.0
            and _number(item.get("second_anchor_margin_m"), "candidate.second_anchor_margin_m") >= 2.0
        )
    except MaterializationError:
        return False


def verified_multiview_pairs(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only pairs satisfying every frozen V1 multiview gate."""
    frames_raw = record.get("frames", [])
    if not isinstance(frames_raw, list):
        raise MaterializationError("frames must be an array")
    frames = {str(item.get("frame_id")): item for item in frames_raw if isinstance(item, Mapping)}
    candidates = list(_candidate_by_id(record).values())
    accepted: list[dict[str, Any]] = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            if not candidate_passes_geometry(left) or not candidate_passes_geometry(right):
                continue
            if left.get("anchor_id") != right.get("anchor_id") or left.get("building_id") != right.get("building_id"):
                continue
            lf, rf = frames.get(str(left.get("frame_id"))), frames.get(str(right.get("frame_id")))
            if not lf or not rf or left.get("frame_id") == right.get("frame_id") or lf.get("is_panorama_slice") or rf.get("is_panorama_slice"):
                continue
            baseline = metric_distance_m(lf.get("camera_position", {}), rf.get("camera_position", {}))
            anchor_residual = metric_distance_m(left.get("predicted_entrance_geo", {}), right.get("predicted_entrance_geo", {}))
            angle = angular_separation_deg(
                _number(left.get("ray_heading_deg"), "candidate.ray_heading_deg"),
                _number(right.get("ray_heading_deg"), "candidate.ray_heading_deg"),
            )
            if 3.0 <= baseline <= 30.0 and anchor_residual <= 3.0 and 10.0 <= angle <= 120.0:
                accepted.append({
                    "candidate_ids": [left["candidate_id"], right["candidate_id"]],
                    "frame_ids": [left["frame_id"], right["frame_id"]],
                    "sequence_ids": [lf.get("sequence_id"), rf.get("sequence_id")],
                    "camera_baseline_m": round(baseline, 6),
                    "anchor_residual_m": round(anchor_residual, 6),
                    "ray_intersection_angle_deg": round(angle, 6),
                })
    return accepted


def audit_provider_input(value: Any) -> list[str]:
    leaks: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key in PROTECTED_PROVIDER_KEYS:
                    leaks.append(child_path)
                visit(child, child_path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return leaks


def prerequisite_report(*, token_present: bool | None = None, candidate_generator_authorized: bool = False) -> dict[str, Any]:
    if token_present is None:
        token_present = bool(os.environ.get("MAPILLARY_ACCESS_TOKEN"))
    failures: list[str] = []
    if not token_present:
        failures.append("MAPILLARY_ACCESS_TOKEN_MISSING")
    if not candidate_generator_authorized:
        failures.append("MANDATORY_CANDIDATE_GENERATOR_NOT_AUTHORIZED")
    return {
        "mapillary_access_token_present": token_present,
        "candidate_generator_authorized": candidate_generator_authorized,
        "failure_codes": failures,
        "ready_for_real_episode_materialization": not failures,
    }


def blocked_closeout(source_report: Mapping[str, Any], *, token_present: bool | None = None) -> dict[str, Any]:
    prerequisites = prerequisite_report(token_present=token_present, candidate_generator_authorized=False)
    source_files = source_report.get("source_files", {})
    implementation_path = Path(__file__).resolve()
    source_slice_path = implementation_path.with_name("source_slice.py")
    protocol_path = implementation_path.parents[4] / "docs" / "research" / "goal-copilot" / "P0_GOAL_GROUNDING_SILVER_PROTOCOL_V1.md"
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "verdict": "P0_S0_SOURCE_OR_LICENSE_BLOCKED",
        "failure_codes": prerequisites["failure_codes"],
        "source_slice_report_sha256": source_report.get("report_sha256"),
        "frozen_identities": {
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "materializer_sha256": hashlib.sha256(implementation_path.read_bytes()).hexdigest(),
            "source_slice_sha256": hashlib.sha256(source_slice_path.read_bytes()).hexdigest(),
        },
        "source_counts": {
            "overture_buildings": source_files.get("overture_buildings", {}).get("feature_count", 0),
            "overture_places": source_files.get("overture_places", {}).get("feature_count", 0),
            "osm_entrances": source_files.get("osm", {}).get("entrance_count", 0),
        },
        "materialized_episode_count": 0,
        "silver_a_primary_count": 0,
        "evaluator_dry_run_episode_count": 0,
        "provider_input_leaks": [],
        "deterministic_source_replay": True,
        "claim_ceiling": "NO_GROUNDING_MODEL_PERFORMANCE_CLAIM",
        "next_action": "SUPPLY_COMPLIANT_MAPILLARY_ACCESS_AND_SEPARATELY_AUTHORIZE_A_FROZEN_CANDIDATE_GENERATOR_BEFORE_NEW_REAL_RUN",
    }
    report["report_sha256"] = content_sha256(report)
    return report


def admit_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Apply frozen hard gates to one normalized candidate record."""
    required = {
        "record_id", "sources", "crosswalk", "anchor", "frames", "candidates", "conflicts",
        "evaluated_system_overlap", "target_visibility_identifiable", "entrance_semantics_match_goal",
        "ancestry_deduplicated",
    }
    missing = sorted(required - set(record))
    if missing:
        return _rejection(record.get("record_id", "unknown"), ["SCHEMA_INADEQUACY"], {"missing_fields": missing})
    provenance_ok, license_ok = provenance_complete(record)
    crosswalk = record["crosswalk"]
    anchor = record["anchor"]
    candidates = _candidate_by_id(record)
    if any(not candidate_shape_valid(item) for item in candidates.values()):
        return _rejection(record["record_id"], ["SCHEMA_INADEQUACY"], {"error": "candidate mechanics fields missing"})
    failures: list[str] = []
    if not provenance_ok:
        failures.append("PROVENANCE_INCOMPLETE")
    if candidates and not all(candidate_mechanics_complete(item) for item in candidates.values()):
        failures.append("PROVENANCE_INCOMPLETE")
    if not license_ok:
        failures.append("LICENSE_METADATA_INCOMPLETE")
    if not isinstance(crosswalk, Mapping) or crosswalk.get("status") != "ADMITTED" or not crosswalk.get("unique"):
        failures.append("TARGET_BUILDING_CROSSWALK_AMBIGUOUS")
    if not isinstance(anchor, Mapping) or anchor.get("status") != "ADMITTED" or not anchor.get("unique"):
        failures.append("NO_VALID_MAP_ANCHOR")
    geometry_candidates = [
        item for item in candidates.values()
        if candidate_passes_geometry(item)
        and item.get("building_id") == crosswalk.get("building_id")
        and item.get("anchor_id") == anchor.get("anchor_id")
    ]
    if not geometry_candidates:
        failures.append("INSUFFICIENT_GEOMETRY_SUPPORT")
    pairs = verified_multiview_pairs(record)
    if not pairs:
        failures.append("INSUFFICIENT_MULTIVIEW_SUPPORT")
    conflicts = record.get("conflicts")
    if not isinstance(conflicts, list) or conflicts:
        failures.append("EVIDENCE_CONFLICT")
    if record.get("evaluated_system_overlap") != "NO_KNOWN_OVERLAP":
        failures.append("EVALUATOR_LINEAGE_OVERLAP")
    if record.get("target_visibility_identifiable") is not True:
        failures.append("TARGET_VISIBILITY_NOT_IDENTIFIABLE")
    if record.get("entrance_semantics_match_goal") is not True:
        failures.append("ENTRANCE_RELATION_NOT_IDENTIFIABLE")
    if record.get("ancestry_deduplicated") is not True:
        failures.append("ANCESTRY_OVERLAP")
    hard_failures = [item for item in failures if item != "INSUFFICIENT_MULTIVIEW_SUPPORT"]
    if hard_failures:
        return _rejection(record["record_id"], failures, {"multiview_pairs": pairs})
    admitted_ids = sorted(item["candidate_id"] for item in geometry_candidates)
    if not pairs:
        return {
            "record_id": record["record_id"],
            "quality_class": "SILVER_B_MAP_GEOMETRY",
            "admission_status": "SECONDARY_ADMITTED",
            "evidence_flags": ["MAP_ANCHORED", "GEOMETRY_VERIFIED"],
            "positive_candidate_ids": admitted_ids,
            "failure_codes": ["INSUFFICIENT_MULTIVIEW_SUPPORT"],
            "multiview_pairs": [],
            "content_sha256": content_sha256(record),
        }
    return {
        "record_id": record["record_id"],
        "quality_class": "SILVER_A_PRIMARY",
        "admission_status": "PRIMARY_ADMITTED",
        "evidence_flags": ["MAP_ANCHORED", "GEOMETRY_VERIFIED", "MULTIVIEW_VERIFIED"],
        "positive_candidate_ids": admitted_ids,
        "failure_codes": [],
        "multiview_pairs": pairs,
        "content_sha256": content_sha256(record),
    }


def _rejection(record_id: Any, failures: Sequence[str], details: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(failures) - FAILURE_CODES)
    if unknown:
        raise MaterializationError(f"unknown failure codes: {unknown}")
    return {
        "record_id": str(record_id),
        "quality_class": "REJECT_AMBIGUOUS",
        "admission_status": "REJECTED",
        "evidence_flags": ["ABSTAIN"],
        "positive_candidate_ids": [],
        "failure_codes": list(dict.fromkeys(failures)),
        "details": dict(details),
    }


def materialize_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    if bundle.get("protocol_id") != PROTOCOL_ID:
        raise MaterializationError("protocol_id mismatch")
    if bundle.get("upstream_commit") != UPSTREAM_COMMIT:
        raise MaterializationError("upstream commit mismatch")
    records = bundle.get("records")
    if not isinstance(records, list) or len(records) > 20:
        raise MaterializationError("records must be an array of at most 20 canary items")
    provider_input = bundle.get("provider_input", {})
    leaks = audit_provider_input(provider_input)
    results: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        try:
            results.append(admit_record(item))
        except (MaterializationError, KeyError, TypeError, ValueError) as error:
            record_id = item.get("record_id", f"record-{index}") if isinstance(item, Mapping) else f"record-{index}"
            results.append(_rejection(record_id, ["SCHEMA_INADEQUACY"], {"error": str(error)}))
    primary = sum(item["quality_class"] == "SILVER_A_PRIMARY" for item in results)
    secondary = sum(item["quality_class"] == "SILVER_B_MAP_GEOMETRY" for item in results)
    rejected = sum(item["admission_status"] == "REJECTED" for item in results)
    verdict = "P0_S0_MATERIALIZATION_CANARY_PASS" if primary else "P0_S0_PASS_WITH_COVERAGE_LIMITATION"
    if leaks:
        verdict = "P0_S0_PROTOCOL_OR_SCHEMA_INADEQUACY"
    if any("SCHEMA_INADEQUACY" in item["failure_codes"] for item in results):
        verdict = "P0_S0_PROTOCOL_OR_SCHEMA_INADEQUACY"
    report = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "verdict": verdict,
        "input_sha256": content_sha256(bundle),
        "record_count": len(results),
        "primary_admitted_count": primary,
        "secondary_admitted_count": secondary,
        "rejected_count": rejected,
        "provider_input_leaks": leaks,
        "results": results,
        "claim_ceiling": "MECHANICS_ONLY_NO_GROUNDING_MODEL_PERFORMANCE_CLAIM",
    }
    report["report_sha256"] = content_sha256(report)
    return report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--candidate-generator-authorized", action="store_true")
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--input", required=True, type=Path)
    materialize.add_argument("--output", required=True, type=Path)
    closeout = subparsers.add_parser("closeout-blocked")
    closeout.add_argument("--source-report", required=True, type=Path)
    closeout.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(prerequisite_report(candidate_generator_authorized=args.candidate_generator_authorized), indent=2))
        return 0
    if args.command == "closeout-blocked":
        source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
        report = blocked_closeout(source_report)
        write_json(args.output, report)
        print(json.dumps({"verdict": report["verdict"], "report_sha256": report["report_sha256"]}))
        return 0
    bundle = json.loads(args.input.read_text(encoding="utf-8"))
    first = materialize_bundle(bundle)
    replay = materialize_bundle(json.loads(args.input.read_text(encoding="utf-8")))
    if canonical_bytes(first) != canonical_bytes(replay):
        raise MaterializationError("deterministic replay mismatch")
    write_json(args.output, first)
    print(json.dumps({"verdict": first["verdict"], "report_sha256": first["report_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
