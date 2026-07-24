#!/usr/bin/env python3
"""Audit the JRDB stitched 2D test labels as a source-authority canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "blindassist_ustrf_jrdb_source_authority_canary_r0"
CONFIG_SCHEMA = "blindassist_ustrf_canonical_observation_source_authority_data_pack_r0"
STAGE = "CANONICAL_OBSERVATION_SOURCE_AUTHORITY_DATA_PACK_R0"
RESEARCH_IMPLEMENTATIONS = {
    "producer": "scripts/research/ustrf_route_target_evidence_closure/audit_jrdb_source_authority_canary_r0.py",
    "validator": "scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_source_authority_canary_r0.py",
}
TERMINALS = (
    "FAIL_CLOSED_ACCESS_OR_AUDIT_INCOMPLETE",
    "SOURCE_SCHEMA_AUTHORITY_INSUFFICIENT",
    "AUTHORITY_CANARY_PRESENT_ROUTE_ROLE_PENDING",
    "SOURCE_DATA_PACK_ADMISSIBLE_FOR_NEW_DISCOVERY",
)
OCCLUSION_STATES = {
    "Fully_visible",
    "Mostly_visible",
    "Severely_occluded",
    "Fully_occluded",
}
FRAME_RE = re.compile(r"^\d{6}\.jpg$")


class CanaryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


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
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("wb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    require(load_json(path) == value, "atomic_output_verify_failed")


def verify_binding(repo: Path, binding: dict[str, Any], label: str) -> Path:
    require(set(binding) == {"path", "sha256"}, f"{label}_binding_keys_drift")
    path = repo / str(binding["path"])
    require(path.is_file(), f"{label}_missing")
    require(sha256_file(path) == binding["sha256"], f"{label}_sha256_drift")
    return path


def load_config(repo: Path, path: Path) -> dict[str, Any]:
    config = load_json(path)
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_drift")
    require(config.get("stage") == STAGE, "config_stage_drift")
    require(config.get("status") == "frozen_before_canary", "config_not_frozen")
    require(tuple(config["terminal_states"]) == TERMINALS, "terminal_states_drift")
    for label, binding in config["bindings"].items():
        verify_binding(repo, binding, label)
    digests = config["research_implementation_digests"]
    require(set(digests) == set(RESEARCH_IMPLEMENTATIONS), "research_implementation_digest_keys_drift")
    for label, relative_path in RESEARCH_IMPLEMENTATIONS.items():
        require(
            sha256_file(repo / relative_path) == digests[label],
            f"research_implementation_{label}_sha256_drift",
        )
    require(
        config["known_missing"]
        == {
            "rgb_frame_identity": "not_in_test_labels_archive",
            "capture_timestamp": "not_in_test_labels_archive",
            "route_role_truth": "not_materialized",
        },
        "known_missing_drift",
    )
    return config


def safe_member(name: str) -> bool:
    pure = PurePosixPath(name)
    return (
        not pure.is_absolute()
        and ".." not in pure.parts
        and not re.match(r"^[A-Za-z]:", name)
        and "\\" not in name
    )


def _boolish(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise CanaryError(f"{label}_not_boolish:{value!r}")


def audit(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(repo, config_path)
    archive = repo / config["bindings"]["jrdb_test_labels"]["path"]
    archive_size = archive.stat().st_size
    require(archive_size <= int(config["resource_gate"]["maximum_download_bytes"]), "archive_over_budget")

    sequence_count = 0
    frame_count = 0
    object_count = 0
    truncated = Counter()
    occlusion = Counter()
    interpolated = Counter()
    no_eval = Counter()
    unique_tracks: set[tuple[str, str]] = set()
    member_count = 0
    uncompressed_bytes = 0

    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        member_count = len(infos)
        require(member_count > 0, "archive_empty")
        require(all(safe_member(info.filename) for info in infos), "archive_unsafe_member")
        uncompressed_bytes = sum(info.file_size for info in infos)
        stitched = sorted(
            (
                info
                for info in infos
                if info.filename.startswith("labels_2d_stitched/")
                and info.filename.endswith(".json")
                and info.file_size > 0
            ),
            key=lambda info: info.filename,
        )
        require(stitched, "stitched_labels_absent")
        for info in stitched:
            sequence_id = PurePosixPath(info.filename).stem
            with bundle.open(info) as handle:
                payload = json.load(handle)
            require(isinstance(payload, dict) and isinstance(payload.get("labels"), dict), "label_schema_drift")
            sequence_count += 1
            for frame_id, rows in payload["labels"].items():
                require(bool(FRAME_RE.fullmatch(frame_id)), f"frame_id_drift:{sequence_id}:{frame_id}")
                require(isinstance(rows, list), f"frame_rows_not_list:{sequence_id}:{frame_id}")
                frame_count += 1
                for row in rows:
                    require(isinstance(row, dict), "object_row_not_object")
                    require(row.get("file_id") == frame_id, "object_frame_identity_drift")
                    box = row.get("box")
                    require(
                        isinstance(box, list)
                        and len(box) == 4
                        and all(isinstance(value, (int, float)) for value in box)
                        and box[2] >= 0
                        and box[3] >= 0,
                        "bbox_schema_drift",
                    )
                    attrs = row.get("attributes")
                    require(isinstance(attrs, dict), "attributes_missing")
                    truncated[_boolish(attrs.get("truncated"), "truncated")] += 1
                    state = attrs.get("occlusion")
                    require(state is None or state in OCCLUSION_STATES, f"occlusion_state_drift:{state!r}")
                    occlusion[state] += 1
                    interpolated[_boolish(attrs.get("interpolated"), "interpolated")] += 1
                    no_eval[_boolish(attrs.get("no_eval"), "no_eval")] += 1
                    label_id = str(row.get("label_id", ""))
                    require(label_id.startswith("pedestrian:"), "label_id_drift")
                    unique_tracks.add((sequence_id, label_id))
                    object_count += 1

    authority_present = truncated[True] > 0 and truncated[False] > 0
    terminal = (
        "AUTHORITY_CANARY_PRESENT_ROUTE_ROLE_PENDING"
        if authority_present
        else "SOURCE_SCHEMA_AUTHORITY_INSUFFICIENT"
    )
    return {
        "schema": SCHEMA,
        "stage": STAGE,
        "status": "AUDIT_COMPLETE",
        "terminal_state": terminal,
        "process_id": os.getpid(),
        "config_sha256": sha256_file(config_path),
        "archive": {
            "path": archive.resolve().relative_to(repo.resolve()).as_posix(),
            "bytes": archive_size,
            "sha256": sha256_file(archive),
            "member_count": member_count,
            "unsafe_member_count": 0,
            "uncompressed_bytes": uncompressed_bytes,
        },
        "stitched_label_canary": {
            "sequence_count": sequence_count,
            "frame_count": frame_count,
            "object_count": object_count,
            "unique_track_count": len(unique_tracks),
            "truncated": {
                "false": truncated[False],
                "true": truncated[True],
                "missing": truncated[None],
            },
            "truncation_coverage": (truncated[False] + truncated[True]) / object_count,
            "occlusion": {
                **dict(sorted((str(key), value) for key, value in occlusion.items() if key is not None)),
                "missing": occlusion[None],
            },
            "interpolated": {
                "false": interpolated[False],
                "true": interpolated[True],
                "missing": interpolated[None],
            },
            "no_eval": {
                "false": no_eval[False],
                "true": no_eval[True],
                "missing": no_eval[None],
            },
            "source_native_truncation_is_nonconstant": authority_present,
        },
        "authority": {
            "truncation_annotation": "source_native_canary_present" if authority_present else "insufficient",
            "occlusion_annotation": "source_native_canary_present",
            "static_sensor_and_projection_documentation": "hash_bound_official_pdf_present",
            "rgb_frame_identity": "pending",
            "capture_timestamp": "pending",
            "route_role_truth": "pending",
        },
        "claim_boundary": {
            "source_authority_canary_only": True,
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
    result = audit(repo, (repo / args.config).resolve() if not args.config.is_absolute() else args.config)
    output = repo / load_json((repo / args.config).resolve())["outputs"]["receipt"]
    atomic_write(output, result)
    print(json.dumps({"receipt": output.relative_to(repo).as_posix(), "terminal_state": result["terminal_state"], "process_id": result["process_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
