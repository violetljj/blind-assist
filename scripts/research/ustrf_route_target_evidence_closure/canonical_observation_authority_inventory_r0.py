#!/usr/bin/env python3
"""Candidate-blind source/transport authority inventory for USTRF G0-A."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CONFIG_SCHEMA = "blindassist_ustrf_canonical_observation_authority_inventory_r0"
INVENTORY_SCHEMA = "blindassist_ustrf_canonical_observation_authority_inventory_r0"
LEDGER_SCHEMA = "blindassist_ustrf_canonical_observation_authority_frame_ledger_r0"
STAGE = "CANONICAL-OBSERVATION-AUTHORITY-AND-REPAIRABILITY-AUDIT-R0-A"
RESEARCH_IMPLEMENTATIONS = {
    "inventory_core": "scripts/research/ustrf_route_target_evidence_closure/canonical_observation_authority_inventory_r0.py",
    "inventory_runner": "scripts/research/ustrf_route_target_evidence_closure/run_canonical_observation_authority_inventory_r0.py",
    "bound_decoder": "scripts/research/ustrf_route_target_evidence_closure/exploratory_profiles_r2_l1.py",
}
LEGAL_STATES = {
    "authoritative",
    "verifiable_transform",
    "inferred",
    "unknown",
    "absent",
}
FORBIDDEN_FRAGMENTS = {
    "alert",
    "candidate",
    "cell",
    "critical",
    "event",
    "negative",
    "oracle",
    "outcome",
    "score",
    "signal",
    "truth",
}


class AuthorityInventoryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorityInventoryError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"json_root_not_object:{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))
    require(load_json(path) == value, f"atomic_json_verify_failed:{path}")


def _verify_binding(repo: Path, binding: dict[str, Any], label: str) -> Path:
    require(set(binding) == {"path", "sha256"}, f"{label}_binding_keys_drift")
    path = repo / str(binding["path"])
    require(path.is_file(), f"{label}_missing")
    require(sha256_file(path) == binding["sha256"], f"{label}_sha256_drift")
    return path


def _forbidden_key_scan(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            require(
                not any(fragment in lowered for fragment in FORBIDDEN_FRAGMENTS),
                f"forbidden_key_in_a_config:{path}.{key}",
            )
            _forbidden_key_scan(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_key_scan(child, path=f"{path}[{index}]")


def load_and_verify_config(repo: Path, path: Path) -> dict[str, Any]:
    config = load_json(path)
    require(config.get("schema") == CONFIG_SCHEMA, "config_schema_drift")
    require(config.get("stage") == STAGE, "config_stage_drift")
    require(config.get("status") == "frozen_before_audit", "config_not_frozen")
    _forbidden_key_scan(
        {
            "source_inputs": config.get("source_inputs"),
            "expected_scope": config.get("expected_scope"),
            "field_contract": config.get("field_contract"),
            "outputs": config.get("outputs"),
        }
    )
    for label, binding in config["parent_bindings"].items():
        _verify_binding(repo, binding, label)
    for label, binding in config["implementation_bindings"].items():
        _verify_binding(repo, binding, f"implementation_{label}")
    digests = config["research_implementation_digests"]
    require(set(digests) == set(RESEARCH_IMPLEMENTATIONS), "research_implementation_digest_keys_drift")
    for label, relative_path in RESEARCH_IMPLEMENTATIONS.items():
        require(
            sha256_file(repo / relative_path) == digests[label],
            f"research_implementation_{label}_sha256_drift",
        )
    expected = config["expected_scope"]
    require(
        expected
        == {
            "crowdbot_sequence_count": 39,
            "lilocbench_sequence_count": 2,
            "sequence_count": 41,
            "frame_count": 62229,
            "observed_track_record_count_from_parent_gap": 263680,
        },
        "expected_scope_drift",
    )
    return config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            require(isinstance(row, dict), f"jsonl_row_not_object:{path}:{line_number}")
            rows.append(row)
    return rows


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    require(
        len(header) == 24
        and header[:8] == b"\x89PNG\r\n\x1a\n"
        and header[12:16] == b"IHDR",
        f"rgb_not_readable_png:{path}",
    )
    width, height = struct.unpack(">II", header[16:24])
    require(width > 0 and height > 0, f"png_size_invalid:{path}")
    return width, height


def _field(
    *,
    origin_authority: str,
    transform_status: str,
    value_state: str,
    scope: str,
    state: str,
    reason_code: str,
    parent_path: str,
    parent_sha256: str,
    join_key: str,
) -> dict[str, Any]:
    require(state in LEGAL_STATES, f"illegal_authority_state:{state}")
    return {
        "origin_authority": origin_authority,
        "transform_status": transform_status,
        "value_state": value_state,
        "scope": scope,
        "state": state,
        "reason_code": reason_code,
        "parent_path": parent_path,
        "parent_sha256": parent_sha256,
        "join_key": join_key,
    }


def _rel(repo: Path, path: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _read_manifest_row(repo: Path, path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": _rel(repo, path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _index_source_rows(
    rows: Iterable[dict[str, Any]], *, source_id: str, sequence_id: str
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        frame_id = int(row["frame_id"])
        require(frame_id not in indexed, f"duplicate_source_frame:{source_id}:{sequence_id}:{frame_id}")
        indexed[frame_id] = row
    return indexed


def _crowdbot_sources(
    repo: Path, config: dict[str, Any], read_manifest: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for root_text in config["source_inputs"]["crowdbot_dataset_roots"]:
        root = repo / root_text
        require(root.is_dir(), f"crowdbot_root_missing:{root_text}")
        for bundle_path in sorted(root.rglob("bundle.json")):
            bundle = load_json(bundle_path)
            if bundle.get("schema") != "blindassist_crowdbot_rgbd_sequence_bundle_r1":
                continue
            key = (str(bundle["source_id"]), str(bundle["sequence_id"]))
            require(key not in sources, f"duplicate_crowdbot_bundle:{key!r}")
            frames_path = Path(str(bundle["frames_path"]))
            require(frames_path.is_file(), f"crowdbot_frames_missing:{key!r}")
            require(
                sha256_file(frames_path) == bundle["frames_sha256"],
                f"crowdbot_frames_sha_drift:{key!r}",
            )
            read_manifest.extend(
                (
                    _read_manifest_row(repo, bundle_path, "crowdbot_source_bundle"),
                    _read_manifest_row(repo, frames_path, "crowdbot_source_frame_ledger"),
                )
            )
            sources[key] = {
                "bundle_path": bundle_path,
                "bundle_sha256": sha256_file(bundle_path),
                "bundle": bundle,
                "frames_path": frames_path,
                "frames": _index_source_rows(
                    _read_jsonl(frames_path), source_id=key[0], sequence_id=key[1]
                ),
            }
    require(
        len(sources) == config["expected_scope"]["crowdbot_sequence_count"],
        "crowdbot_bundle_count_drift",
    )
    return sources


def _crowdbot_observation_ledgers(
    repo: Path, config: dict[str, Any], read_manifest: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    root = repo / config["source_inputs"]["crowdbot_observation_ledger_root"]
    ledgers: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".successor-receipt.json"):
            continue
        ledger = load_json(path)
        if ledger.get("schema") != "blindassist_ustrf_route_target_l1e_compact_detector_ledger_r1":
            continue
        key = (str(ledger["source_id"]), str(ledger["sequence_id"]))
        require(key not in ledgers, f"duplicate_crowdbot_observation_ledger:{key!r}")
        read_manifest.append(_read_manifest_row(repo, path, "crowdbot_observation_ledger"))
        ledgers[key] = {"path": path, "sha256": sha256_file(path), "value": ledger}
    require(
        len(ledgers) == config["expected_scope"]["crowdbot_sequence_count"],
        "crowdbot_observation_ledger_count_drift",
    )
    return ledgers


def _lilocbench_sources(
    repo: Path, config: dict[str, Any], read_manifest: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for source_id, binding in config["source_inputs"]["lilocbench_source_bundles"].items():
        bundle_path = _verify_binding(repo, binding, f"lilocbench_bundle_{source_id}")
        bundle = load_json(bundle_path)
        require(bundle["source"]["source_id"] == source_id, f"lilocbench_source_id_drift:{source_id}")
        frames_path = bundle_path.with_name("frames.jsonl")
        require(frames_path.is_file(), f"lilocbench_frames_missing:{source_id}")
        require(sha256_file(frames_path) == bundle["frames_sha256"], f"lilocbench_frames_sha_drift:{source_id}")
        read_manifest.extend(
            (
                _read_manifest_row(repo, bundle_path, "lilocbench_source_bundle"),
                _read_manifest_row(repo, frames_path, "lilocbench_source_frame_ledger"),
            )
        )
        result[source_id] = {
            "bundle_path": bundle_path,
            "bundle_sha256": sha256_file(bundle_path),
            "bundle": bundle,
            "frames_path": frames_path,
            "frames": _index_source_rows(
                _read_jsonl(frames_path), source_id=source_id, sequence_id=source_id
            ),
        }
    require(
        len(result) == config["expected_scope"]["lilocbench_sequence_count"],
        "lilocbench_source_count_drift",
    )
    return result


def _lilocbench_observations(
    repo: Path, config: dict[str, Any], read_manifest: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    root = repo / config["source_inputs"]["lilocbench_observation_ledger_root"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".successor-receipt.json"):
            continue
        ledger = load_json(path)
        if ledger.get("schema") != "blindassist_ustrf_route_target_l1e_compact_detector_ledger_r1":
            continue
        read_manifest.append(_read_manifest_row(repo, path, "lilocbench_observation_ledger"))
        for row in ledger["frames"]:
            grouped[str(row["source_id"])].append(row)
    require(
        set(grouped) == set(config["source_inputs"]["lilocbench_source_bundles"]),
        "lilocbench_observation_sources_drift",
    )
    return dict(grouped)


def _timestamp_ns(source_id: str, row: dict[str, Any]) -> int:
    if source_id.startswith("lilocbench_"):
        return round(float(row["rgb_timestamp_s"]) * 1_000_000_000)
    return int(row["source_capture_timestamp_ns"])


def _rgb_path(source_id: str, source: dict[str, Any], row: dict[str, Any]) -> Path:
    if source_id.startswith("lilocbench_"):
        return Path(str(source["bundle"]["source_root"])) / str(row["rgb_path"])
    return source["frames_path"].parent / str(row["rgb_path"])


def _frame_row(
    *,
    repo: Path,
    source_id: str,
    sequence_id: str,
    observation: dict[str, Any],
    source: dict[str, Any],
    source_row: dict[str, Any],
    previous_timestamp_ns: int | None,
) -> tuple[dict[str, Any], int]:
    frame_id = int(observation["frame_id"])
    timestamp_ns = _timestamp_ns(source_id, source_row)
    observation_ts = int(observation["source_capture_timestamp_ns"])
    require(timestamp_ns == observation_ts, f"timestamp_join_drift:{source_id}:{sequence_id}:{frame_id}")
    image_path = _rgb_path(source_id, source, source_row)
    require(image_path.is_file(), f"rgb_missing:{source_id}:{sequence_id}:{frame_id}")
    expected_rgb_sha = str(source_row["rgb_sha256"])
    actual_rgb_sha = sha256_file(image_path)
    require(actual_rgb_sha == expected_rgb_sha, f"rgb_sha_drift:{source_id}:{sequence_id}:{frame_id}")
    width, height = _png_size(image_path)
    if source_id.startswith("crowdbot_"):
        camera = source["bundle"]["camera_info"]
        require([width, height] == [int(camera["width"]), int(camera["height"])], f"camera_png_size_drift:{source_id}:{sequence_id}:{frame_id}")
    claimed_size = [int(value) for value in observation["source_size"]]
    size_matches = claimed_size == [width, height]
    boxes = observation.get("person_detections", [])
    person_boxes = [
        row for row in boxes if str(row.get("label", "")) == "person"
    ]
    join_key = f"{source_id}::{sequence_id}::{frame_id}"
    source_rel = _rel(repo, source["frames_path"])
    source_sha = sha256_file(source["frames_path"])
    gap_ns = None if previous_timestamp_ns is None else timestamp_ns - previous_timestamp_ns
    require(gap_ns is None or gap_ns > 0, f"timestamp_not_monotonic:{join_key}")
    fields = {
        "source_geometry": _field(
            origin_authority="encoded_png_raster_and_source_camera_metadata",
            transform_status="identity",
            value_state="known-valid",
            scope="frame",
            state="authoritative",
            reason_code="PNG_IHDR_MATCHES_SOURCE_METADATA",
            parent_path=source_rel,
            parent_sha256=source_sha,
            join_key=join_key,
        ),
        "canonical_transform": _field(
            origin_authority="bound_replay_and_decoder_implementation",
            transform_status="reproducible-unbound",
            value_state="unknown",
            scope="frame",
            state="unknown",
            reason_code="NO_FRAME_BOUND_ROTATION_CROP_FLIP_LETTERBOX_RECEIPT",
            parent_path=_rel(repo, source["bundle_path"]),
            parent_sha256=source["bundle_sha256"],
            join_key=join_key,
        ),
        "bbox_coordinate_frame": _field(
            origin_authority="detector_transport",
            transform_status="hash-bound-deterministic",
            value_state="known-valid" if size_matches else "known-invalid",
            scope="frame",
            state="verifiable_transform" if size_matches else "unknown",
            reason_code=(
                "DECODED_BOX_EXTENT_MATCHES_ENCODED_RASTER"
                if size_matches
                else "DECODED_BOX_EXTENT_CONFLICTS_WITH_ENCODED_RASTER"
            ),
            parent_path=_rel(repo, source["bundle_path"]),
            parent_sha256=source["bundle_sha256"],
            join_key=join_key,
        ),
        "severe_truncation": _field(
            origin_authority="absent_from_source_and_transport",
            transform_status="unavailable",
            value_state="absent",
            scope="observed-track",
            state="absent",
            reason_code="NO_SOURCE_OR_ANNOTATION_SEVERE_TRUNCATION_AUTHORITY",
            parent_path=source_rel,
            parent_sha256=source_sha,
            join_key=join_key,
        ),
        "rgb_continuity": _field(
            origin_authority="source_transport",
            transform_status="hash-bound-deterministic",
            value_state="known-valid",
            scope="frame",
            state="authoritative",
            reason_code="RGB_EXISTS_READABLE_AND_SHA_VERIFIED",
            parent_path=_rel(repo, image_path),
            parent_sha256=actual_rgb_sha,
            join_key=join_key,
        ),
        "capture_time": _field(
            origin_authority="source_transport",
            transform_status="identity",
            value_state="known-valid",
            scope="frame",
            state="authoritative",
            reason_code="SOURCE_TIMESTAMP_JOIN_EXACT_AND_MONOTONIC",
            parent_path=source_rel,
            parent_sha256=source_sha,
            join_key=join_key,
        ),
        "frame_membership": _field(
            origin_authority="frozen_observation_ledger",
            transform_status="hash-bound-deterministic",
            value_state="known-valid",
            scope="frame",
            state="authoritative",
            reason_code="OBSERVATION_IDENTITY_JOINS_SOURCE_FRAME",
            parent_path=source_rel,
            parent_sha256=source_sha,
            join_key=join_key,
        ),
        "background_feature_input": _field(
            origin_authority="rgb_plus_detector_object_adapter",
            transform_status="reproducible-unbound",
            value_state="unknown",
            scope="frame",
            state="inferred",
            reason_code="RGB_AVAILABLE_BUT_PERSON_MASK_AND_FEATURE_QUALITY_NOT_SOURCE_AUTHORITY",
            parent_path=_rel(repo, image_path),
            parent_sha256=actual_rgb_sha,
            join_key=join_key,
        ),
    }
    return (
        {
            "source_id": source_id,
            "sequence_id": sequence_id,
            "frame_id": frame_id,
            "source_capture_timestamp_ns": timestamp_ns,
            "adjacent_gap_ns": gap_ns,
            "source_size": [width, height],
            "observation_source_size": claimed_size,
            "observed_person_box_count": len(person_boxes),
            "fields": fields,
        },
        timestamp_ns,
    )


def build_inventory(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_and_verify_config(repo, config_path)
    output_root = repo / config["outputs"]["root"]
    read_manifest: list[dict[str, Any]] = []
    crowdbot_sources = _crowdbot_sources(repo, config, read_manifest)
    crowdbot_ledgers = _crowdbot_observation_ledgers(repo, config, read_manifest)
    require(set(crowdbot_sources) == set(crowdbot_ledgers), "crowdbot_source_observation_roster_drift")
    liloc_sources = _lilocbench_sources(repo, config, read_manifest)
    liloc_observations = _lilocbench_observations(repo, config, read_manifest)
    ledger_inventory: list[dict[str, Any]] = []
    state_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_frames = 0
    total_person_boxes = 0
    sequence_roster: list[tuple[str, str, list[dict[str, Any]], dict[str, Any]]] = []
    for key in sorted(crowdbot_ledgers):
        sequence_roster.append(
            (
                key[0],
                key[1],
                crowdbot_ledgers[key]["value"]["frames"],
                crowdbot_sources[key],
            )
        )
    for source_id in sorted(liloc_observations):
        sequence_roster.append(
            (source_id, source_id, liloc_observations[source_id], liloc_sources[source_id])
        )
    require(len(sequence_roster) == config["expected_scope"]["sequence_count"], "sequence_roster_count_drift")
    for source_id, sequence_id, observations, source in sequence_roster:
        source_rows = source["frames"]
        ledger_rows: list[dict[str, Any]] = []
        previous: int | None = None
        for observation in observations:
            frame_id = int(observation["frame_id"])
            require(frame_id in source_rows, f"source_frame_join_missing:{source_id}:{sequence_id}:{frame_id}")
            row, previous = _frame_row(
                repo=repo,
                source_id=source_id,
                sequence_id=sequence_id,
                observation=observation,
                source=source,
                source_row=source_rows[frame_id],
                previous_timestamp_ns=previous,
            )
            ledger_rows.append(row)
            total_person_boxes += int(row["observed_person_box_count"])
            for field_name, field in row["fields"].items():
                state_counts[field_name][field["state"]] += 1
        payload = {
            "schema": LEDGER_SCHEMA,
            "stage": STAGE,
            "source_id": source_id,
            "sequence_id": sequence_id,
            "frame_count": len(ledger_rows),
            "frames": ledger_rows,
        }
        digest = hashlib.sha256(f"{source_id}::{sequence_id}".encode()).hexdigest()[:16]
        path = output_root / "frame-ledgers" / f"{digest}.json"
        atomic_write_json(path, payload)
        ledger_inventory.append(
            {
                "source_id": source_id,
                "sequence_id": sequence_id,
                "path": _rel(repo, path),
                "sha256": sha256_file(path),
                "frame_count": len(ledger_rows),
            }
        )
        total_frames += len(ledger_rows)
    require(total_frames == config["expected_scope"]["frame_count"], "inventory_frame_count_drift")
    read_manifest = sorted(read_manifest, key=lambda row: (row["path"], row["role"]))
    read_paths = [row["path"] for row in read_manifest]
    require(len(read_paths) == len(set(read_paths)), "read_manifest_duplicate_path")
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "stage": STAGE,
        "status": "AUTHORITY_INVENTORY_FROZEN",
        "process_id": os.getpid(),
        "config_sha256": sha256_file(config_path),
        "evidence_universe": {
            "root_allowlist": config["source_inputs"]["root_allowlist"],
            "read_manifest": read_manifest,
            "sequence_count": len(ledger_inventory),
            "frame_count": total_frames,
        },
        "authority_taxonomy": {
            "states": sorted(LEGAL_STATES),
            "dimensions": [
                "origin_authority",
                "transform_status",
                "value_state",
                "scope",
            ],
        },
        "field_state_counts": {
            field: dict(sorted(counts.items()))
            for field, counts in sorted(state_counts.items())
        },
        "observed_person_box_count": total_person_boxes,
        "parent_observed_track_record_count": config["expected_scope"][
            "observed_track_record_count_from_parent_gap"
        ],
        "ledger_inventory": ledger_inventory,
        "decoded_counters": {
            "event": 0,
            "truth": 0,
            "oracle": 0,
            "cell": 0,
            "negative": 0,
            "signal": 0,
            "candidate": 0,
            "outcome": 0,
        },
        "claim_boundary": {
            "signal_computed": False,
            "slope_computed": False,
            "schema_repaired": False,
            "android_authority": False,
            "human_authority": False,
            "production_authority": False,
        },
    }
    inventory_path = output_root / config["outputs"]["inventory"]
    atomic_write_json(inventory_path, inventory)
    return {
        "status": inventory["status"],
        "inventory_path": _rel(repo, inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "sequence_count": len(ledger_inventory),
        "frame_count": total_frames,
        "process_id": os.getpid(),
    }
