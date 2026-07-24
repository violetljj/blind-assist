#!/usr/bin/env python3
"""Audit public JRDB access for a same-frame RGB/time/transform canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "blindassist_ustrf_jrdb_rgb_time_frame_transform_access_canary_r0"
CONFIG_SCHEMA = "blindassist_ustrf_jrdb_rgb_time_frame_transform_access_canary_r0_config"
STAGE = "JRDB_RGB_TIME_FRAME_TRANSFORM_ACCESS_CANARY_R0"
TERMINALS = (
    "FAIL_CLOSED_AUDIT_INCOMPLETE",
    "ACCESS_BLOCKED_LOGIN_REQUIRED",
    "FRAME_IDENTITY_OR_TIME_AUTHORITY_INSUFFICIENT",
    "RGB_TIME_TRANSFORM_CANARY_PRESENT",
)
RESEARCH_IMPLEMENTATIONS = {
    "producer": "scripts/research/ustrf_route_target_evidence_closure/audit_jrdb_rgb_time_frame_transform_access_canary_r0.py",
    "validator": "scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_rgb_time_frame_transform_access_canary_r0.py",
}


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_root_not_object:{path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def load_config(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    require(config["schema"] == CONFIG_SCHEMA, "config_schema_drift")
    require(config["stage"] == STAGE, "stage_drift")
    require(config["status"] == "frozen_before_execution", "config_not_frozen")
    require(tuple(config["terminal_states"]) == TERMINALS, "terminal_order_drift")
    for label, binding in config["bindings"].items():
        path = repo / binding["path"]
        require(path.is_file(), f"{label}_missing")
        require(sha256_file(path) == binding["sha256"], f"{label}_sha256_drift")
    digests = config["research_implementation_digests"]
    require(set(digests) == set(RESEARCH_IMPLEMENTATIONS), "implementation_digest_keys_drift")
    for label, relative_path in RESEARCH_IMPLEMENTATIONS.items():
        require(sha256_file(repo / relative_path) == digests[label], f"{label}_implementation_drift")
    return config


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and not re.match(r"^[A-Za-z]:", name)


def audit(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(repo, config_path)
    bindings = config["bindings"]
    page = (repo / bindings["public_download_page"]["path"]).read_text(encoding="utf-8")
    readme = (repo / bindings["toolkit_visualisation_readme"]["path"]).read_text(encoding="utf-8")
    visualise = (repo / bindings["toolkit_visualise"]["path"]).read_text(encoding="utf-8")
    sample_path = repo / bindings["sample_structure"]["path"]
    with zipfile.ZipFile(sample_path) as archive:
        members = archive.namelist()
    require(members and all(safe_member(name) for name in members), "sample_structure_unsafe")

    login_required = "required to log in" in page and "not logged in" in page
    public_test_labels = "/static/downloads/test_labels.zip" in page
    normalized_readme = " ".join(readme.split())
    dataset_download_required = (
        "download the dataset set on https://jrdb.erc.monash.edu/" in normalized_readme
    )
    sample_has_timestamps_dir = any(name.endswith("/timestamps/") for name in members)
    sample_has_real_files = any(not name.endswith("/") for name in members)
    image_path_contract = "images/image_stitched/{location}/{file_index}.jpg" in visualise
    label_keys_used_as_timestamps = (
        "timestamps = sorted(list(visualizer.labels_3d.keys()))" in visualise
        and "timestamps = [ts[:-4] for ts in self.labels_3d.keys()]" in visualise
    )
    calibration_present = all(
        (repo / bindings[key]["path"]).is_file()
        for key in ("toolkit_calibration_defaults", "toolkit_calibration_cameras")
    )
    rgb_present = False
    capture_timestamp_present = False
    canary_complete = rgb_present and capture_timestamp_present and image_path_contract and calibration_present
    if canary_complete:
        terminal = "RGB_TIME_TRANSFORM_CANARY_PRESENT"
    elif login_required and dataset_download_required:
        terminal = "ACCESS_BLOCKED_LOGIN_REQUIRED"
    else:
        terminal = "FRAME_IDENTITY_OR_TIME_AUTHORITY_INSUFFICIENT"

    return {
        "schema": SCHEMA,
        "stage": STAGE,
        "status": "AUDIT_COMPLETE",
        "terminal_state": terminal,
        "process_id": os.getpid(),
        "config_sha256": sha256_file(config_path),
        "official_toolkit_commit": config["official_toolkit_commit"],
        "public_access": {
            "login_required": login_required,
            "public_test_labels_link_present": public_test_labels,
            "dataset_download_required_by_toolkit": dataset_download_required,
        },
        "sample_structure": {
            "member_count": len(members),
            "unsafe_member_count": 0,
            "timestamps_directory_declared": sample_has_timestamps_dir,
            "real_payload_file_count": sum(not name.endswith("/") for name in members),
            "contains_real_files": sample_has_real_files,
        },
        "authority": {
            "stitched_image_path_contract": image_path_contract,
            "calibration_files_present": calibration_present,
            "real_rgb_present": rgb_present,
            "independent_capture_timestamp_present": capture_timestamp_present,
            "toolkit_label_keys_called_timestamps": label_keys_used_as_timestamps,
            "label_key_is_capture_time_authority": False,
            "same_frame_rgb_time_transform_canary_complete": canary_complete,
        },
        "claim_boundary": {
            "access_audit_only": True,
            "g1_authorized": False,
            "signal_authorized": False,
            "route_truth_authorized": False,
            "android_authorized": False,
            "human_authorized": False,
            "production_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve()
    result = audit(repo, config_path)
    output = repo / load_json(config_path)["outputs"]["receipt"]
    atomic_write(output, result)
    print(json.dumps({"terminal_state": result["terminal_state"], "process_id": result["process_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
