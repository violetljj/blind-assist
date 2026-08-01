#!/usr/bin/env python3
"""Audit transport for the frozen HFTF Stage C E0 fresh cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_stage_c_c0_1_egowalk_timebase_repair import (  # noqa: E402
    _parquet_timeline_metrics,
)
from audit_stage_c_c0_egowalk_transport import (  # noqa: E402
    _decode_stream,
)


SCHEMA = "blindassist_hftf_stage_c_e0_fresh_media_transport"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_fresh_foot_ground_student_canary_e0"
)
PROTOCOL_STATUS = (
    "FROZEN_BEFORE_FRESH_RGB_DEPTH_OR_GEOMETRY_LABEL_OUTCOME"
)
ACQUISITION_SCHEMA = (
    "blindassist_hftf_stage_c_e0_fresh_media_acquisition"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_acquisition(
    protocol: dict[str, Any],
    protocol_path: Path,
    acquisition: dict[str, Any],
    media_root: Path,
) -> None:
    if acquisition.get("schema") != ACQUISITION_SCHEMA:
        raise ValueError("Unexpected E0 acquisition schema")
    if (
        acquisition.get("terminal")
        != "E0_FRESH_MEDIA_BYTES_ACQUIRED_AND_HASH_BOUND"
    ):
        raise ValueError("E0 media bytes are not acquired and bound")
    if acquisition.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("E0 acquisition protocol binding mismatch")
    if Path(str(acquisition["output_root"])).resolve() != media_root:
        raise ValueError("E0 acquisition media-root mismatch")
    if not acquisition.get("selected_sources_burned"):
        raise ValueError("E0 fresh-source burn was not recorded")
    if acquisition.get("fresh_geometry_label_outcome_read"):
        raise ValueError("Acquisition unexpectedly read geometry outcome")
    if not acquisition.get("transport_decode_audit_authorized"):
        raise ValueError("E0 transport audit is not authorized")

    expected = {
        (item["role"], item["trajectory"], kind): item["files"][kind]
        for item in protocol["frozen_sources"]
        for kind in ("pose", "rgb", "depth")
    }
    actual = {
        (item["role"], item["trajectory"], item["kind"]): {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in acquisition["downloaded_files"]
        if item["kind"] in {"pose", "rgb", "depth"}
    }
    if actual != expected:
        raise ValueError("E0 acquisition file ledger mismatch")


def _audit_source(
    source: dict[str, Any],
    media_root: Path,
    meta_fps: float,
) -> dict[str, Any]:
    pose_path = media_root / source["files"]["pose"]["path"]
    rgb_path = media_root / source["files"]["rgb"]["path"]
    depth_path = media_root / source["files"]["depth"]["path"]
    timeline = _parquet_timeline_metrics(pose_path)
    rgb = _decode_stream(rgb_path, "rgb", [])
    depth = _decode_stream(depth_path, "depth", [])
    expected_rows = int(source["rows"])
    failures: list[str] = []
    if (
        timeline["rows"] != expected_rows
        or not timeline["frame_zero_contiguous"]
        or not timeline["timestamps_strictly_increasing"]
    ):
        failures.append("pose_frame_or_timestamp_alignment")
    if (
        timeline["effective_rate_hz"] is None
        or abs(float(timeline["effective_rate_hz"]) - meta_fps) > 0.01
    ):
        failures.append("physical_timeline_rate_mismatch")
    if rgb["decoded_frame_count"] != expected_rows:
        failures.append("rgb_frame_count_mismatch")
    if depth["decoded_frame_count"] != expected_rows:
        failures.append("depth_frame_count_mismatch")
    if not rgb["pts_strictly_increasing"]:
        failures.append("rgb_pts_not_strictly_increasing")
    if not depth["pts_strictly_increasing"]:
        failures.append("depth_pts_not_strictly_increasing")
    return {
        "role": source["role"],
        "trajectory": source["trajectory"],
        "expected_rows": expected_rows,
        "parquet_timeline": timeline,
        "frame_counts": {
            "pose": timeline["rows"],
            "rgb": rgb["decoded_frame_count"],
            "depth": depth["decoded_frame_count"],
        },
        "rgb_stream": {
            key: value
            for key, value in rgb.items()
            if key not in {"_samples", "path"}
        },
        "depth_stream": {
            key: value
            for key, value in depth.items()
            if key not in {"_samples", "path"}
        },
        "container_nominal_rate_used_as_physical_timeline": False,
        "gate_failures": failures,
        "transport_pass": not failures,
    }


def audit(
    protocol_path: Path,
    acquisition_path: Path,
    media_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("Stage C E0 protocol is not frozen")
    acquisition = _load_json(acquisition_path)
    _validate_acquisition(
        protocol,
        protocol_path,
        acquisition,
        media_root,
    )
    meta = _load_json(media_root / "meta/info.json")
    meta_fps = float(meta["fps"])
    reports = [
        _audit_source(source, media_root, meta_fps)
        for source in protocol["frozen_sources"]
    ]
    passed = meta_fps == 5.0 and all(
        item["transport_pass"] for item in reports
    )
    terminal = (
        "E0_FRESH_MEDIA_TRANSPORT_SUPPORTED"
        if passed
        else "E0_FRESH_MEDIA_TRANSPORT_NOT_EVALUABLE"
    )
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "acquisition_manifest_path": str(acquisition_path.resolve()),
        "acquisition_manifest_sha256": _sha256(acquisition_path),
        "media_root": str(media_root),
        "meta_info_fps": meta_fps,
        "trajectory_reports": reports,
        "all_transport_gates_pass": passed,
        "fresh_rgb_or_depth_media_content_read": True,
        "fresh_geometry_label_outcome_read": False,
        "student_output_read": False,
        "teacher_mechanics_and_label_opportunity_audit_authorized": passed,
        "teacher_corpus_generation_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _require_new_artifacts_output(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    artifacts_root = (repo_root / "artifacts.local").resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError("E0 audit output must stay under artifacts.local") from error
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite report: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    protocol = (repo_root / args.protocol).resolve()
    acquisition = (repo_root / args.acquisition_manifest).resolve()
    media_root = (repo_root / args.media_root).resolve()
    output = _require_new_artifacts_output(
        repo_root / args.output, repo_root
    )
    first = audit(protocol, acquisition, media_root)
    second = audit(protocol, acquisition, media_root)
    deterministic = (
        json.dumps(first, sort_keys=True, separators=(",", ":"))
        == json.dumps(second, sort_keys=True, separators=(",", ":"))
    )
    if not deterministic:
        raise ValueError("E0 transport result is not byte deterministic")
    first["determinism_check"] = {
        "second_run_payload_byte_exact": True
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(first, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "terminal": first["terminal"],
                "all_transport_gates_pass": first[
                    "all_transport_gates_pass"
                ],
                "sources": [
                    {
                        "trajectory": item["trajectory"],
                        "frame_counts": item["frame_counts"],
                        "effective_rate_hz": item[
                            "parquet_timeline"
                        ]["effective_rate_hz"],
                    }
                    for item in first["trajectory_reports"]
                ],
                "deterministic": True,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if first["all_transport_gates_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
