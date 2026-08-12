#!/usr/bin/env python3
"""Offline independent replay of a D3R1 Phase-B PASS or FAIL terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from scripts.research.assistive_geometry.arkitscenes_truth_reader import (
    WORLD_UP,
    TruthReaderPolicy,
    canonicalize_frame,
    depth_mm_to_metres,
    derive_assistive_truth,
    interpolate_camera_to_world,
    parse_trajectory,
)
from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d3r1_phase_b import (
    ASSETS,
    ATTEMPT_SCHEMA,
    BANDS,
    CHECKPOINT_SCHEMA,
    FAIL_TERMINAL,
    HORIZONS,
    MANIFEST_SCHEMA,
    PASS_NEXT_GATE,
    PASS_TERMINAL,
    checkpoint_path,
    expected_attempt,
    head_lookup,
    json_bytes,
    load_json,
    phase_a_selection,
    read_sealed_json,
    require,
    require_under,
    same_path,
    sha256_file,
    validate_bindings,
    write_json_exclusive,
)


RESULT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_terminal_validation_v1"


def independently_qualifies(counts: dict[str, Any], thresholds: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    require(counts["known_cells"] == counts["clear_cells"] + counts["occupied_cells"], "known total identity drift")
    grids = [f"{band}@{horizon:.1f}m" for band in BANDS for horizon in HORIZONS]
    require(set(counts["known_by_grid"]) == set(grids), "known grid schema drift")
    require(set(counts["clear_by_grid"]) == set(grids), "clear grid schema drift")
    require(set(counts["occupied_by_grid"]) == set(grids), "occupied grid schema drift")
    require(sum(counts["known_by_grid"].values()) == counts["known_cells"], "known grid total drift")
    require(sum(counts["clear_by_grid"].values()) == counts["clear_cells"], "clear grid total drift")
    require(sum(counts["occupied_by_grid"].values()) == counts["occupied_cells"], "occupied grid total drift")
    for key in grids:
        require(counts["known_by_grid"][key] == counts["clear_by_grid"][key] + counts["occupied_by_grid"][key], f"grid identity drift: {key}")
    for count_key, threshold_key in (
        ("known_cells", "minimum_truth_known_cells"),
        ("clear_cells", "minimum_truth_clear_cells"),
        ("occupied_cells", "minimum_truth_occupied_cells"),
        ("valid_band_clearances", "minimum_valid_band_clearances"),
    ):
        if int(counts[count_key]) < int(thresholds[threshold_key]):
            failures.append(f"{count_key}={counts[count_key]}<{thresholds[threshold_key]}")
    for key in grids:
        if int(counts["clear_by_grid"][key]) < int(thresholds["minimum_truth_clear_cells_per_band_horizon"]):
            failures.append(
                f"{key}_clear={counts['clear_by_grid'][key]}<"
                f"{thresholds['minimum_truth_clear_cells_per_band_horizon']}"
            )
        if int(counts["occupied_by_grid"][key]) < int(thresholds["minimum_truth_occupied_cells_per_band_horizon"]):
            failures.append(
                f"{key}_occupied={counts['occupied_by_grid'][key]}<"
                f"{thresholds['minimum_truth_occupied_cells_per_band_horizon']}"
            )
    return not failures, failures


def independent_empty_counts() -> dict[str, Any]:
    grid = {f"{band}@{horizon:.1f}m": 0 for band in BANDS for horizon in HORIZONS}
    return {
        "known_cells": 0,
        "clear_cells": 0,
        "occupied_cells": 0,
        "valid_band_clearances": 0,
        "known_by_grid": dict(grid),
        "clear_by_grid": dict(grid),
        "occupied_by_grid": dict(grid),
    }


def independent_summarize_truth(truth: dict[str, Any]) -> dict[str, Any]:
    counts = independent_empty_counts()
    for band in BANDS:
        band_row = truth.get("bands", {}).get(band)
        if not band_row:
            continue
        if band_row.get("clearance_m") is not None:
            counts["valid_band_clearances"] += 1
        states = band_row.get("occupied_by_horizon", {})
        for horizon in HORIZONS:
            state = states.get(str(horizon))
            if state is None:
                continue
            key = f"{band}@{horizon:.1f}m"
            counts["known_cells"] += 1
            counts["known_by_grid"][key] += 1
            class_name = "occupied" if bool(state) else "clear"
            counts[f"{class_name}_cells"] += 1
            counts[f"{class_name}_by_grid"][key] += 1
    return counts


def independent_add_counts(total: dict[str, Any], frame: dict[str, Any]) -> None:
    for key in ("known_cells", "clear_cells", "occupied_cells", "valid_band_clearances"):
        total[key] += int(frame[key])
    for group in ("known_by_grid", "clear_by_grid", "occupied_by_grid"):
        for key in total[group]:
            total[group][key] += int(frame[group][key])


def independent_timestamp(stem: str) -> float:
    try:
        value = Decimal(stem.rsplit("_", 1)[-1])
    except (InvalidOperation, IndexError) as error:
        raise ValueError(f"invalid independent frame timestamp: {stem}") from error
    require(value.is_finite(), f"non-finite independent frame timestamp: {stem}")
    result = float(value)
    require(math.isfinite(result), f"non-finite independent frame timestamp: {stem}")
    return result


def independent_pincam(payload: bytes, label: str) -> tuple[np.ndarray, tuple[int, int]]:
    try:
        values = [float(value) for value in payload.decode("utf-8").split()]
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"independent invalid pincam: {label}") from error
    require(len(values) == 6 and all(math.isfinite(value) for value in values), f"independent invalid pincam fields: {label}")
    width, height = int(values[0]), int(values[1])
    require(values[0] == width and values[1] == height and width > 0 and height > 0, f"independent invalid pincam dimensions: {label}")
    fx, fy, cx, cy = values[2:]
    require(fx > 0 and fy > 0 and 0 <= cx < width and 0 <= cy < height, f"independent invalid pincam values: {label}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), (width, height)


def independent_archive_map(archive: zipfile.ZipFile, suffix: str) -> tuple[dict[str, str], int]:
    require(archive.testzip() is None, f"independent ZIP CRC failure: {archive.filename}")
    mapping: dict[str, str] = {}
    names: set[str] = set()
    file_count = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename.replace("\\", "/"))
        require(not path.is_absolute() and ".." not in path.parts, f"independent unsafe ZIP member: {info.filename}")
        require(info.filename not in names, f"independent duplicate ZIP member: {info.filename}")
        names.add(info.filename)
        if info.is_dir():
            continue
        file_count += 1
        if path.suffix.lower() == suffix:
            require(path.stem not in mapping, f"independent duplicate frame stem: {path.stem}")
            mapping[path.stem] = info.filename
    require(mapping, f"independent ZIP has no {suffix}: {archive.filename}")
    return mapping, file_count


def independently_audit_identity(
    *,
    identity: dict[str, Any],
    depth_path: Path,
    confidence_path: Path,
    phase_a_root: Path,
    policy: TruthReaderPolicy,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    video_id = str(identity["video_id"])
    phase_a_dir = phase_a_root / "source" / "Training" / video_id
    intrinsics_path = phase_a_dir / "lowres_wide_intrinsics.zip"
    trajectory_path = phase_a_dir / "lowres_wide.traj"
    recorded = {str(row["asset"]): row for row in identity["phase_a_checkpoint"]["source_assets"]}
    require(set(recorded) == {"lowres_wide_intrinsics.zip", "lowres_wide.traj"}, "independent Phase-A source inventory drift")
    phase_a_sources: list[dict[str, Any]] = []
    for asset, path in (("lowres_wide_intrinsics.zip", intrinsics_path), ("lowres_wide.traj", trajectory_path)):
        entry = recorded[asset]
        require(path.is_file() and path.stat().st_size == int(entry["bytes"]), f"independent Phase-A source byte drift: {asset}")
        digest = sha256_file(path)
        require(digest == entry["sha256"], f"independent Phase-A source SHA drift: {asset}")
        phase_a_sources.append({"asset": asset, "path": str(path.resolve()), "bytes": int(entry["bytes"]), "sha256": digest})
    trajectory = parse_trajectory(trajectory_path)
    stems = list(identity["selected_frame_stems"])
    counts = independent_empty_counts()
    orientation_counts = {str(index): 0 for index in range(4)}
    depth_sizes: set[tuple[int, int]] = set()
    confidence_values: set[int] = set()
    maximum_pose_gap = 0.0
    with (
        zipfile.ZipFile(intrinsics_path) as intrinsics_zip,
        zipfile.ZipFile(depth_path) as depth_zip,
        zipfile.ZipFile(confidence_path) as confidence_zip,
    ):
        intrinsics_map, intrinsics_files = independent_archive_map(intrinsics_zip, ".pincam")
        depth_map, depth_files = independent_archive_map(depth_zip, ".png")
        confidence_map, confidence_files = independent_archive_map(confidence_zip, ".png")
        for label, mapping in (("intrinsics", intrinsics_map), ("depth", depth_map), ("confidence", confidence_map)):
            require(all(stem in mapping for stem in stems), f"independent selected {label} coverage drift")
        for stem in stems:
            intrinsics, source_size = independent_pincam(intrinsics_zip.read(intrinsics_map[stem]), intrinsics_map[stem])
            with depth_zip.open(depth_map[stem]) as stream, Image.open(stream) as image:
                depth_raw = np.asarray(image).copy()
            with confidence_zip.open(confidence_map[stem]) as stream, Image.open(stream) as image:
                confidence = np.asarray(image).copy()
            require(depth_raw.ndim == 2 and np.issubdtype(depth_raw.dtype, np.integer), "independent invalid depth raster")
            require(confidence.ndim == 2 and np.issubdtype(confidence.dtype, np.integer), "independent invalid confidence raster")
            require(depth_raw.shape == confidence.shape and source_size == (depth_raw.shape[1], depth_raw.shape[0]), "independent registered shape drift")
            depth_sizes.add((int(depth_raw.shape[1]), int(depth_raw.shape[0])))
            confidence_values.update(int(value) for value in np.unique(confidence))
            pose, metadata = interpolate_camera_to_world(
                trajectory, independent_timestamp(stem), policy.maximum_pose_bracketing_gap_seconds
            )
            maximum_pose_gap = max(maximum_pose_gap, float(metadata["bracketing_gap_seconds"]))
            dummy_rgb = np.zeros((*depth_raw.shape, 3), dtype=np.uint8)
            canonical = canonicalize_frame(dummy_rgb, depth_raw, confidence, intrinsics, pose)
            orientation_counts[str(canonical["rotation_index"])] += 1
            require(canonical["rotation_index"] in (1, 3), "independent portrait orientation drift")
            up_camera = canonical["camera_to_world"][:3, :3].T @ WORLD_UP
            truth = derive_assistive_truth(
                depth_mm_to_metres(canonical["depth_raw_mm"]),
                canonical["confidence"],
                canonical["intrinsics"],
                up_camera,
                policy,
            )
            independent_add_counts(counts, independent_summarize_truth(truth))
    require(confidence_values.issubset({0, 1, 2}), f"independent confidence values drift: {sorted(confidence_values)}")
    qualified, failures = independently_qualifies(counts, thresholds)
    stem_digest = hashlib.sha256(("\n".join(stems) + "\n").encode("ascii")).hexdigest().upper()
    return {
        "frame_count": len(stems),
        "selected_frame_plan_sha256": stem_digest,
        "archive_validation": {
            "intrinsics_total_file_members": intrinsics_files,
            "intrinsics_frame_members": len(intrinsics_map),
            "depth_total_file_members": depth_files,
            "depth_frame_members": len(depth_map),
            "confidence_total_file_members": confidence_files,
            "confidence_frame_members": len(confidence_map),
            "selected_intrinsics_coverage": len(stems),
            "selected_depth_coverage": len(stems),
            "selected_confidence_coverage": len(stems),
            "zip_crc_and_member_safety_checked": True,
        },
        "trajectory_row_count": int(trajectory.shape[0]),
        "maximum_pose_bracketing_gap_seconds": maximum_pose_gap,
        "depth_sizes_wh": [list(value) for value in sorted(depth_sizes)],
        "confidence_values": sorted(confidence_values),
        "orientation_counts": orientation_counts,
        "truth_support": counts,
        "source_truth_support_qualified": qualified,
        "qualification_failures": failures,
        "phase_a_sources": phase_a_sources,
        "rgb_read": False,
        "model_output_read": False,
        "per_frame_truth_retained": False,
    }


def independently_finalize(processed: list[dict[str, Any]]) -> tuple[bool, list[dict[str, Any]]]:
    require(len(processed) == 32, "independent selection requires exact32")
    qualified = [row for row in processed if row["source_truth_support_qualified"]]
    if len(qualified) < 16:
        return False, []
    selected: list[dict[str, Any]] = []
    for order, row in enumerate(qualified[:16], start=1):
        selected.append({
            "phase_b_selection_order": order,
            "phase_a_selection_order": row["selection_order"],
            "pool_order": row["pool_order"],
            "visit_id": row["visit_id"],
            "video_id": row["video_id"],
            "fold": "Training",
            "selected_frame_count": 300,
            "selected_frame_plan_sha256": row["selected_frame_plan_sha256"],
            "role_assigned": False,
        })
    return True, selected


def comparable_checkpoint(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value[key]
        for key in (
            "frame_count",
            "selected_frame_plan_sha256",
            "archive_validation",
            "trajectory_row_count",
            "maximum_pose_bracketing_gap_seconds",
            "depth_sizes_wh",
            "confidence_values",
            "orientation_counts",
            "truth_support",
            "source_truth_support_qualified",
            "phase_a_sources",
            "rgb_read",
            "model_output_read",
            "per_frame_truth_retained",
        )
    }


def validate_inventory(
    output_root: Path,
    selected: list[dict[str, Any]],
    attempt: dict[str, Any],
    heads: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    require(not (output_root / "_temporary_downloads").exists(), "temporary inventory remains")
    source_root = output_root / "source" / "Training"
    receipts_root = output_root / "receipts"
    expected_receipts = [checkpoint_path(receipts_root, row) for row in selected]
    actual_receipts = sorted(
        path for path in receipts_root.glob("[0-9][0-9][0-9]-*.json") if not path.name.endswith(".sha256.json")
    )
    require([path.name for path in actual_receipts] == [path.name for path in expected_receipts], "checkpoint inventory drift")
    require(len(list(receipts_root.glob("*.sha256.json"))) == 32, "checkpoint sidecar inventory drift")
    expected_receipt_names = {path.name for path in expected_receipts} | {
        path.with_suffix(".sha256.json").name for path in expected_receipts
    }
    require({path.name for path in receipts_root.iterdir()} == expected_receipt_names, "extra receipt inventory")
    require(
        {path.name for path in output_root.iterdir()}
        == {"attempt.json", "manifest.json", "manifest.sha256.json", "receipts", "source"},
        "attempt-root inventory drift",
    )
    actual_dirs = sorted(path.name for path in source_root.iterdir() if path.is_dir())
    expected_dirs = sorted(str(row["video_id"]) for row in selected)
    require(actual_dirs == expected_dirs, "source directory inventory drift")
    require(all(path.is_dir() for path in source_root.iterdir()), "extra source-root file")
    attempt_sha = hashlib.sha256(json_bytes(attempt)).hexdigest().upper()
    checkpoints: list[dict[str, Any]] = []
    for identity, path in zip(selected, expected_receipts, strict=True):
        checkpoint = read_sealed_json(path, CHECKPOINT_SCHEMA)
        require(checkpoint["attempt_sha256"] == attempt_sha, "checkpoint attempt drift")
        for key in ("selection_order", "pool_order", "visit_id", "video_id", "fold"):
            require(checkpoint[key] == identity[key], f"checkpoint identity drift: {key}")
        identity_root = source_root / str(identity["video_id"])
        require(sorted(path.name for path in identity_root.iterdir()) == sorted(ASSETS), "source asset inventory drift")
        source_assets = checkpoint["source_assets"]
        require([entry["asset"] for entry in source_assets] == list(ASSETS), "checkpoint source asset order/family drift")
        for entry in source_assets:
            head = heads[(str(identity["video_id"]), str(entry["asset"]))]
            expected = identity_root / str(entry["asset"])
            actual = Path(str(entry["path"]))
            require(same_path(actual, expected), "checkpoint source path drift")
            require_under(actual, output_root, "checkpoint source")
            require(actual.stat().st_size == int(entry["bytes"]), "checkpoint source byte drift")
            require(sha256_file(actual) == entry["sha256"], "checkpoint source SHA drift")
            require(entry["response_http_status"] == 200, "GET status drift")
            require(entry["url"] == entry["response_final_url"] == head["url"], "GET/HEAD URL binding drift")
            require(entry["response_content_length_bytes"] == entry["frozen_head_content_length_bytes"] == entry["bytes"] == int(head["content_length_bytes"]), "GET/HEAD length binding drift")
            require(entry["response_etag"] == entry["frozen_head_etag"] == head["etag"], "GET/HEAD ETag binding drift")
            require(entry["response_last_modified"] == entry["frozen_head_last_modified"] == head["last_modified"], "GET/HEAD Last-Modified binding drift")
            require(entry["range_request_used"] is False and entry["redirect_followed"] is False, "GET boundary drift")
            history = entry["attempt_history"]
            require(len(history) == entry["attempts"] and 1 <= len(history) <= 3, "GET attempt history drift")
            for index, attempt_row in enumerate(history, start=1):
                require(attempt_row["attempt"] == index and attempt_row["method"] == "GET", "GET attempt order/method drift")
                if index < len(history):
                    status = attempt_row["http_status"]
                    require(attempt_row["error"] is not None, "successful attempt cannot be retried")
                    require(attempt_row["error_type"] is not None, "retry error type missing")
                    if status is None:
                        require(attempt_row["retry_class"] == "TRANSIENT_TRANSPORT", "unclassified transport retry")
                    else:
                        require(
                            attempt_row["retry_class"] == "TRANSIENT_HTTP"
                            and (status in {408, 429} or 500 <= status <= 599),
                            "non-transient HTTP retry",
                        )
            require(
                history[-1]["http_status"] == 200
                and history[-1]["error"] is None
                and history[-1]["error_type"] is None
                and history[-1]["retry_class"] is None,
                "GET final attempt drift",
            )
        require(checkpoint["range_get_used"] is False and checkpoint["redirect_followed"] is False, "checkpoint transport boundary drift")
        require(checkpoint["role_assigned"] is False and checkpoint["training"] is False, "checkpoint role/training drift")
        require(checkpoint["development_outcome_read"] is False and checkpoint["r2_access"] == "NONE", "checkpoint outcome boundary drift")
        checkpoints.append(checkpoint)
    return checkpoints


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--phase-a-result", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--head-governed-result", type=Path, required=True)
    parser.add_argument("--head-machine-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol, _, source_scope, phase_a_result, phase_a, head_machine = validate_bindings(
        protocol_path=args.protocol,
        activation_path=args.activation,
        source_scope_path=args.source_scope,
        phase_a_result_path=args.phase_a_result,
        phase_a_manifest_path=args.phase_a_manifest,
        head_governed_path=args.head_governed_result,
        head_machine_path=args.head_machine_result,
    )
    output_root = args.manifest.parent
    require(same_path(output_root, args.output.parent), "validation output must share manifest root")
    require(same_path(output_root, Path(__file__).resolve().parents[5] / protocol["output_root"]), "manifest root drift")
    manifest = read_sealed_json(args.manifest, MANIFEST_SCHEMA)
    selected = phase_a_selection(source_scope, phase_a_result, phase_a)
    heads = head_lookup(head_machine, selected)
    declared_bytes = sum(int(row["content_length_bytes"]) for row in heads.values())
    expected_attempt_value = expected_attempt(
        protocol_path=args.protocol,
        activation_path=args.activation,
        source_scope_path=args.source_scope,
        phase_a_result_path=args.phase_a_result,
        phase_a_manifest_path=args.phase_a_manifest,
        head_governed_path=args.head_governed_result,
        head_machine_path=args.head_machine_result,
        output_root=output_root,
        selected=selected,
        protocol=protocol,
        declared_bytes=declared_bytes,
    )
    attempt = load_json(output_root / "attempt.json")
    require(attempt.get("schema") == ATTEMPT_SCHEMA and attempt == expected_attempt_value, "attempt receipt drift")
    checkpoints = validate_inventory(output_root, selected, attempt, heads)
    policy = TruthReaderPolicy()
    policy.validate()
    replayed: list[dict[str, Any]] = []
    for identity, checkpoint in zip(selected, checkpoints, strict=True):
        identity_root = output_root / "source" / "Training" / str(identity["video_id"])
        audit = independently_audit_identity(
            identity=identity,
            depth_path=identity_root / "lowres_depth.zip",
            confidence_path=identity_root / "confidence.zip",
            phase_a_root=args.phase_a_manifest.parent,
            policy=policy,
            thresholds=protocol["truth_support_thresholds"],
        )
        require(comparable_checkpoint(checkpoint) == comparable_checkpoint(audit), f"offline replay mismatch: {identity['video_id']}")
        require(
            sorted(checkpoint["qualification_failures"])
            == sorted(audit["qualification_failures"]),
            f"offline qualification failure-set mismatch: {identity['video_id']}",
        )
        independently_qualified, _ = independently_qualifies(audit["truth_support"], protocol["truth_support_thresholds"])
        require(independently_qualified is checkpoint["source_truth_support_qualified"], "independent gate mismatch")
        replayed.append(checkpoint)
        print(
            json.dumps(
                {
                    "validated": len(replayed),
                    "total": 32,
                    "video_id": identity["video_id"],
                    "qualified": checkpoint["source_truth_support_qualified"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    passed, selected_phase_b = independently_finalize(replayed)
    expected_terminal = PASS_TERMINAL if passed else FAIL_TERMINAL
    require(manifest["terminal"] == expected_terminal, "manifest terminal drift")
    require(manifest["bindings"] == attempt["bindings"], "manifest binding drift")
    require(manifest["attempt_sha256"] == hashlib.sha256(json_bytes(attempt)).hexdigest().upper(), "manifest attempt SHA drift")
    require(manifest["processed"] == checkpoints, "manifest processed payload drift")
    require(manifest["identity_count"] == manifest["processed_identity_count"] == 32 and manifest["selected_frame_count"] == 9600, "manifest count drift")
    qualified_count = sum(bool(row["source_truth_support_qualified"]) for row in checkpoints)
    require(manifest["source_truth_support_qualified_identity_count"] == qualified_count, "qualified count drift")
    expected_qualified = [
        {key: row[key] for key in ("selection_order", "pool_order", "visit_id", "video_id")}
        for row in checkpoints if row["source_truth_support_qualified"]
    ]
    require(manifest["source_truth_support_qualified_identities"] == expected_qualified, "qualified identity payload drift")
    require(manifest["phase_b_selection_locked"] is passed, "selection lock drift")
    require(manifest["selected_phase_b"] == selected_phase_b, "selected Phase-B payload drift")
    require(manifest["selected_identity_count"] == len(selected_phase_b), "selected count drift")
    require(manifest["next_gate"] == (PASS_NEXT_GATE if passed else None), "manifest successor drift")
    require(manifest["declared_download_bytes"] == manifest["downloaded_body_bytes"] == declared_bytes, "manifest body byte drift")
    require(manifest["truth_reader_policy"] == protocol["truth_reader_policy"], "manifest truth policy drift")
    require(manifest["truth_support_thresholds"] == protocol["truth_support_thresholds"], "manifest threshold drift")
    require(manifest["source_archives_retained_for_offline_validation"] is True, "manifest source retention drift")
    for key, expected in (
        ("per_frame_truth_retained", False),
        ("rgb_read", False),
        ("model_output_read", False),
        ("role_assignment_made", False),
        ("training", False),
        ("development_outcome_read", False),
        ("r2_access", "NONE"),
        ("performance_claim", False),
        ("android_default_authority", False),
        ("production_authority", False),
        ("safety_authority", False),
    ):
        require(manifest[key] == expected, f"manifest authority drift: {key}")
    result = {
        "schema": RESULT_SCHEMA,
        "status": "D3R1_PHASE_B_TERMINAL_OFFLINE_VALIDATION_PASS",
        "scientific_terminal": expected_terminal,
        "manifest": {"path": str(args.manifest.resolve()), "bytes": args.manifest.stat().st_size, "sha256": sha256_file(args.manifest)},
        "processed_identity_count": 32,
        "source_asset_count": 64,
        "replayed_selected_frame_count": 9600,
        "source_truth_support_qualified_identity_count": qualified_count,
        "selected_identity_count": len(selected_phase_b),
        "archive_crc_and_member_safety_replayed": True,
        "source_sha_and_header_bindings_replayed": True,
        "truth_support_recomputed": True,
        "per_frame_truth_retained": False,
        "rgb_read": False,
        "model_output_read": False,
        "role_assignment_made": False,
        "training": False,
        "development_outcome_read": False,
        "r2_access": "NONE",
        "next_gate": PASS_NEXT_GATE if passed else None,
    }
    write_json_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
