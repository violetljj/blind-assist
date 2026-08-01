#!/usr/bin/env python3
"""Audit HFTF Stage C E0.1 fresh dev/heldout transport."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_stage_c_e0_fresh_media_transport as e0_transport  # noqa: E402


SCHEMA = "blindassist_hftf_stage_c_e0_1_fresh_transport"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_foot_ground_student_canary_e0_1"
)
PROTOCOL_STATUS = (
    "FROZEN_BEFORE_FRESH_EVALUATION_RGB_DEPTH_OR_LABEL_OUTCOME"
)
ACQUISITION_SCHEMA = (
    "blindassist_hftf_stage_c_e0_1_fresh_evaluation_media_acquisition"
)


def _load_json(path: Path) -> dict[str, Any]:
    return e0_transport._load_json(path)


def _sha256(path: Path) -> str:
    return e0_transport._sha256(path)


def _validate_acquisition(
    protocol: dict[str, Any],
    protocol_path: Path,
    acquisition: dict[str, Any],
    media_root: Path,
) -> None:
    if acquisition.get("schema") != ACQUISITION_SCHEMA:
        raise ValueError("Unexpected E0.1 acquisition schema")
    if (
        acquisition.get("terminal")
        != "E0_1_FRESH_EVALUATION_MEDIA_BYTES_ACQUIRED_AND_HASH_BOUND"
    ):
        raise ValueError("E0.1 evaluation media is not acquired")
    if acquisition.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("E0.1 acquisition protocol binding mismatch")
    if Path(str(acquisition["output_root"])).resolve() != media_root:
        raise ValueError("E0.1 acquisition media-root mismatch")
    if not acquisition.get("new_dev_and_heldout_burned"):
        raise ValueError("E0.1 fresh evaluation burn is not recorded")
    if acquisition.get("fresh_evaluation_geometry_label_outcome_read"):
        raise ValueError("E0.1 acquisition unexpectedly read labels")
    if not acquisition.get("fresh_transport_audit_authorized"):
        raise ValueError("E0.1 transport is not authorized")
    expected = {
        (item["role"], item["trajectory"], kind): item["files"][kind]
        for item in protocol["fresh_evaluation_sources"]
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
        raise ValueError("E0.1 acquisition file ledger mismatch")


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
        raise ValueError("Stage C E0.1 protocol is not frozen")
    acquisition = _load_json(acquisition_path)
    _validate_acquisition(
        protocol, protocol_path, acquisition, media_root
    )
    meta = _load_json(media_root / "meta/info.json")
    meta_fps = float(meta["fps"])
    reports = [
        e0_transport._audit_source(source, media_root, meta_fps)
        for source in protocol["fresh_evaluation_sources"]
    ]
    passed = meta_fps == 5.0 and all(
        item["transport_pass"] for item in reports
    )
    terminal = (
        "E0_1_FRESH_EVALUATION_MEDIA_TRANSPORT_SUPPORTED"
        if passed
        else "E0_1_FRESH_EVALUATION_MEDIA_TRANSPORT_NOT_EVALUABLE"
    )
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "acquisition_manifest_path": str(acquisition_path),
        "acquisition_manifest_sha256": _sha256(acquisition_path),
        "media_root": str(media_root),
        "meta_info_fps": meta_fps,
        "trajectory_reports": reports,
        "all_transport_gates_pass": passed,
        "fresh_evaluation_rgb_or_depth_read": True,
        "fresh_evaluation_geometry_label_outcome_read": False,
        "student_output_read": False,
        "fresh_0_4_s_teacher_opportunity_audit_authorized": passed,
        "teacher_corpus_generation_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _new_output(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((repo_root / "artifacts.local").resolve())
    except ValueError as error:
        raise ValueError("E0.1 transport output must stay under artifacts.local") from error
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
    output = _new_output(repo_root / args.output, repo_root)
    first = audit(protocol, acquisition, media_root)
    second = audit(protocol, acquisition, media_root)
    if json.dumps(first, sort_keys=True) != json.dumps(
        second, sort_keys=True
    ):
        raise ValueError("E0.1 transport report is not deterministic")
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
