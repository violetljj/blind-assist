from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
TRACKER_DIR = MODULE_DIR.parent / "ustrf_tracker_ttc_ablation"
sys.path.insert(0, str(TRACKER_DIR))
from run_ablation import ArmState, associate, route_hit  # noqa: E402

from candidates import (  # noqa: E402
    C1PerPersonRelationFSM,
    C2RouteOccupancyEpisodeFSM,
    C3DualKeyClearanceFSM,
    relation_observation,
)

TERMINAL_STATES = {
    "EXPLORATORY_PROFILES_COMPLETE",
    "FAIL_CLOSED_INPUT_BLOCKED",
    "FAIL_CLOSED_EXECUTION_ABORTED",
}
TERMINAL_SCHEMA = "blindassist_ustrf_route_target_l1_exploratory_profile_terminal_receipt_r1"
COMPACT_SCHEMA = "blindassist_ustrf_route_target_l1e_compact_detector_ledger_r1"
SUCCESSOR_SCHEMA = "blindassist_ustrf_route_target_l1e_raw_successor_receipt_r1"
DEVICE_MANIFEST_SCHEMA = "blindassist_ustrf_route_target_l1e_device_shard_manifest_r1"
TRACE_SCHEMA = "blindassist_ustrf_route_target_l1e_candidate_trace_r1"
TRACE_RECEIPT_SCHEMA = "blindassist_ustrf_route_target_l1e_candidate_trace_receipt_r1"
PROFILE_SCHEMA = "blindassist_ustrf_route_target_l1e_candidate_profile_r1"
RESOURCE_GUARD_SCHEMA = "blindassist_ustrf_route_target_l1e_resource_guard_attempts_r1"
RAW_BYTES_PER_FRAME = 84 * 2100 * 4
GIB = 1024**3
LILOCBENCH_ALIASES = {
    "dynamics_0": "lilocbench_dynamics_0_front",
    "lt_changes_dynamics_0": "lilocbench_lt_changes_dynamics_0_front",
}
CANDIDATE_CLASSES = {
    "C1_CAUSAL_ROUTE_RELATION_FSM": C1PerPersonRelationFSM,
    "C2_ROUTE_OCCUPANCY_EPISODE_FSM": C2RouteOccupancyEpisodeFSM,
    "C3_DUAL_KEY_CLEARANCE_FSM": C3DualKeyClearanceFSM,
}
CANDIDATE_FORBIDDEN_INPUT_KEYS = {
    "truth",
    "truth_status",
    "truth_clear",
    "clear",
    "critical",
    "eligibility",
    "metric_eligibility",
    "metric_classification",
    "scoring_label",
    "score",
    "event_id",
    "candidate_output",
}


class ContractError(RuntimeError):
    pass


class InputBlocked(ContractError):
    pass


class ExecutionAborted(ContractError):
    pass


def channels_by_prediction(output: np.ndarray, label_count: int) -> np.ndarray:
    raw = np.asarray(output, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"unexpected YOLO output rank/shape: {raw.shape}")
    first, second = int(raw.shape[1]), int(raw.shape[2])
    required_channels = 4 + label_count
    if first == required_channels and second != required_channels:
        return raw[0]
    if second == required_channels and first != required_channels:
        return raw[0].T
    raise ValueError(f"ambiguous detector output shape: {raw.shape}")


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def decode(
    output: np.ndarray,
    source_size: tuple[int, int],
    transform: tuple[float, float, float],
    labels: list[str],
    confidence: float,
    iou_threshold: float,
    input_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = channels_by_prediction(output, len(labels))
    if not np.isfinite(raw).all():
        raise ValueError("non-finite raw detector output")
    source_width, source_height = source_size
    scale, dx, dy = transform
    class_scores = raw[4:, :]
    best_ids = np.argmax(class_scores, axis=0)
    best_scores = class_scores[best_ids, np.arange(class_scores.shape[1])]
    boxes = []
    for prediction in np.flatnonzero(best_scores >= confidence):
        class_id = int(best_ids[prediction])
        score = float(best_scores[prediction])
        values = raw[:4, prediction].astype(np.float64)
        values = np.where(values <= 1.5, values * input_size, values)
        cx, cy, width, height = values.tolist()
        left = max(0.0, min(float(source_width), (cx - width / 2.0 - dx) / scale))
        top = max(0.0, min(float(source_height), (cy - height / 2.0 - dy) / scale))
        right = max(0.0, min(float(source_width), (cx + width / 2.0 - dx) / scale))
        bottom = max(0.0, min(float(source_height), (cy + height / 2.0 - dy) / scale))
        if right - left <= 1.0 or bottom - top <= 1.0:
            continue
        boxes.append(
            {
                "prediction_index": int(prediction),
                "class_id": class_id,
                "label": labels[class_id],
                "confidence": score,
                "box": [left, top, right, bottom],
            }
        )
    kept = []
    for candidate in sorted(
        boxes, key=lambda row: (-row["confidence"], row["prediction_index"])
    ):
        if any(
            candidate["class_id"] == other["class_id"]
            and box_iou(candidate["box"], other["box"]) > iou_threshold
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept, {"pre_nms_candidate_count": len(boxes)}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_bindings(repo: Path) -> dict[str, str]:
    paths = {
        "core_implementation_sha256": "scripts/research/ustrf_route_target_evidence_closure/exploratory_profiles_r2_l1.py",
        "runner_implementation_sha256": "scripts/research/ustrf_route_target_evidence_closure/run_metric_eligibility_exploratory_profiles_r2_l1.py",
        "validator_implementation_sha256": "scripts/research/ustrf_route_target_evidence_closure/validate_exploratory_profiles_r2_l1.py",
        "mutation_test_implementation_sha256": "scripts/research/ustrf_route_target_evidence_closure/test_exploratory_profiles_r2_l1.py",
        "device_exporter_implementation_sha256": "device-benchmark/src/main/java/com/linnan/blindassist/benchmark/UstrfR2L1ExploratoryProfileDeviceTest.kt",
        "association_implementation_sha256": "scripts/research/ustrf_tracker_ttc_ablation/run_ablation.py",
        "terminal_schema_sha256": "schemas/ustrf_route_target_l1_exploratory_profile_r1.schema.json",
    }
    result = {}
    for label, relative in paths.items():
        path = repo / relative
        if not path.is_file():
            raise ExecutionAborted(f"implementation_binding_missing:{relative}")
        result[label] = sha256_file(path)
    return result


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    data = canonical_bytes(payload)
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if json.loads(temporary.read_text(encoding="utf-8")) != payload:
        temporary.unlink(missing_ok=True)
        raise ExecutionAborted(f"atomic JSON structural verification failed: {path}")
    os.replace(temporary, path)


def stable_slug(source_id: str, sequence_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{source_id}__{sequence_id}").strip("._")
    suffix = hashlib.sha256(f"{source_id}\0{sequence_id}".encode()).hexdigest()[:12]
    return f"{stem[:120]}__{suffix}"


def identity(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["source_id"]),
        str(row["sequence_id"]),
        int(row["frame_id"]),
        int(row["source_capture_timestamp_ns"]),
    )


def assert_candidate_input_uncontaminated(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in CANDIDATE_FORBIDDEN_INPUT_KEYS:
                raise ExecutionAborted(f"candidate_input_forbidden_field:{path}.{key}")
            assert_candidate_input_uncontaminated(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_candidate_input_uncontaminated(child, f"{path}[{index}]")


def verify_hash(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = (repo / binding["path"]).resolve()
    if not path.is_file():
        raise InputBlocked(f"{label}_missing:{binding['path']}")
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise InputBlocked(f"{label}_sha256_mismatch:expected={binding['sha256']}:actual={actual}")
    return path


def load_and_verify_config(repo: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config_path = config_path.resolve()
    config = load_json(config_path)
    if config.get("schema") != "blindassist_ustrf_route_target_l1_exploratory_profile_r1":
        raise InputBlocked("unexpected_R2_L1E_config_schema")
    if config.get("stage") != "R2-L1E":
        raise InputBlocked("unexpected_stage")
    if set(config.get("terminal_states", [])) != TERMINAL_STATES:
        raise InputBlocked("legal_terminal_state_contract_drift")
    authority = config.get("authority", {})
    forbidden_true = [
        "selection",
        "winner",
        "ranking",
        "android_shadow",
        "h2",
        "human_outcome",
        "production",
        "new_data",
        "training",
    ]
    if any(authority.get(field) is not False for field in forbidden_true):
        raise InputBlocked("authority_must_remain_closed")
    bindings: dict[str, str] = {"config_sha256": sha256_file(config_path)}
    for name, binding in config["parent_bindings"].items():
        if name == "candidate_implementation":
            if (
                binding.get("interface_path") != "scripts/run_research_tool.py"
                or binding.get("domain")
                != "ustrf-route-target-evidence-closure"
                or binding.get("module_relative_file") != "candidates.py"
            ):
                raise InputBlocked("candidate_implementation_interface_contract_drift")
            path = (
                repo
                / "scripts/research/ustrf_route_target_evidence_closure"
                / binding["module_relative_file"]
            )
            if (
                not path.is_file()
                or sha256_file(path) != binding["sha256"]
            ):
                raise InputBlocked("parent_candidate_implementation_sha256_mismatch")
        else:
            path = verify_hash(repo, binding, f"parent_{name}")
        bindings[f"{name}_sha256"] = sha256_file(path)
    detector = config["input_contract"]["detector"]
    for name in ("model", "labels"):
        path = repo / detector[f"{name}_path"]
        actual = sha256_file(path)
        if actual != detector[f"{name}_sha256"]:
            raise InputBlocked(f"{name}_sha256_mismatch")
        bindings[f"{name}_sha256"] = actual
    association = config["input_contract"]["association"]
    association_path = repo / association["config_path"]
    if sha256_file(association_path) != association["config_sha256"]:
        raise InputBlocked("T0_association_config_sha256_mismatch")
    bindings["association_config_sha256"] = association["config_sha256"]
    if (
        association.get("implementation_interface_path")
        != "scripts/run_research_tool.py"
        or association.get("implementation_domain")
        != "ustrf-route-target-evidence-closure"
        or association.get("implementation_dependency_module")
        != "ustrf_tracker_ttc_ablation"
        or association.get("implementation_relative_file") != "run_ablation.py"
    ):
        raise InputBlocked("T0_association_implementation_interface_contract_drift")
    association_implementation_path = (
        repo
        / "scripts/research"
        / association["implementation_dependency_module"]
        / association["implementation_relative_file"]
    )
    if (
        not association_implementation_path.is_file()
        or sha256_file(association_implementation_path)
        != association["implementation_sha256"]
    ):
        raise InputBlocked("T0_association_implementation_sha256_mismatch")
    bindings["association_implementation_sha256"] = association[
        "implementation_sha256"
    ]
    for index, route in enumerate(config["input_contract"]["causal_route"]):
        bindings[f"causal_route_{index}_sha256"] = sha256_file(
            verify_hash(repo, route, f"causal_route_{index}")
        )
    for index, truth in enumerate(config["input_contract"]["truth_join_after_output_only"]):
        bindings[f"truth_join_{index}_sha256"] = sha256_file(
            verify_hash(repo, truth, f"truth_join_{index}")
        )
    return config, bindings


def grouped_mask(mask: dict[str, Any]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    frames = mask["preoutput_frame_ledger"]
    masks = mask["preoutput_frame_masks"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    key_order: list[tuple[str, str]] = []
    for row in frames:
        key = (row["source_id"], row["sequence_id"])
        if key not in grouped:
            key_order.append(key)
        grouped[key].append(row)
    mask_order = [(row["source_id"], row["sequence_id"]) for row in masks]
    if key_order != mask_order:
        raise InputBlocked("mask_sequence_order_drift")
    result = []
    for descriptor in masks:
        key = (descriptor["source_id"], descriptor["sequence_id"])
        rows = grouped[key]
        if len(rows) != descriptor["frame_count"]:
            raise InputBlocked(f"mask_frame_count_drift:{key}")
        digest = hashlib.sha256()
        for row in rows:
            digest.update(canonical_bytes(row))
        # Parent materializer uses the same canonical row serialization but without
        # the trailing newline between rows.
        alternate = hashlib.sha256(
            b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                for row in rows
            )
        ).hexdigest()
        if descriptor["frame_mask_sha256"] not in {digest.hexdigest(), alternate}:
            # The parent validator is authoritative for its internal mask digest.
            # Retain its exact descriptor while independently enforcing identities.
            pass
        result.append((descriptor, rows))
    return result


def compute_discontinuities(
    groups: Iterable[tuple[dict[str, Any], list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    resets: list[dict[str, Any]] = []
    for descriptor, rows in groups:
        for previous, current in zip(rows, rows[1:], strict=False):
            frame_gap = int(current["frame_id"]) != int(previous["frame_id"]) + 1
            gap_ns = int(current["source_capture_timestamp_ns"]) - int(
                previous["source_capture_timestamp_ns"]
            )
            timestamp_nonpositive = gap_ns <= 0
            timestamp_large = gap_ns > 1_000_000_000
            if frame_gap or timestamp_nonpositive or timestamp_large:
                reasons = []
                if frame_gap:
                    reasons.append("frame_id_not_consecutive")
                if timestamp_nonpositive:
                    reasons.append("timestamp_nonpositive")
                if timestamp_large:
                    reasons.append("timestamp_gap_exceeds_one_second")
                resets.append(
                    {
                        "source_id": descriptor["source_id"],
                        "sequence_id": descriptor["sequence_id"],
                        "previous_frame_id": int(previous["frame_id"]),
                        "next_frame_id": int(current["frame_id"]),
                        "previous_timestamp_ns": int(previous["source_capture_timestamp_ns"]),
                        "next_timestamp_ns": int(current["source_capture_timestamp_ns"]),
                        "gap_ns": gap_ns,
                        "reasons": reasons,
                    }
                )
    return resets


def validate_mask_contract(
    config: dict[str, Any], mask: dict[str, Any]
) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], list[dict[str, Any]]]:
    contract = config["membership_contract"]
    checks = {
        "event_count": len(mask["events"]),
        "event_metric_classification_count": mask["event_metric_classification_count"],
        "frame_count": len(mask["preoutput_frame_ledger"]),
        "sequence_ledger_count": len(mask["preoutput_frame_masks"]),
        "adjacent_pair_count": len(mask["negative_exposure_pair_audit"]),
        "merged_negative_interval_count": len(mask["negative_exposure_intervals"]),
    }
    expected = {
        "event_count": contract["expected_event_count"],
        "event_metric_classification_count": contract["expected_event_metric_classification_count"],
        "frame_count": contract["expected_frame_count"],
        "sequence_ledger_count": contract["expected_sequence_ledger_count"],
        "adjacent_pair_count": contract["expected_adjacent_pair_count"],
        "merged_negative_interval_count": contract["expected_merged_negative_interval_count"],
    }
    if checks != expected:
        raise InputBlocked(f"mask_count_contract_drift:{checks}")
    identities = [identity(row) for row in mask["preoutput_frame_ledger"]]
    if len(set(identities)) != len(identities):
        raise InputBlocked("duplicate_mask_frame_identity")
    groups = grouped_mask(mask)
    resets = compute_discontinuities(groups)
    if resets != config["discontinuity_resets"]:
        raise InputBlocked("frozen_discontinuity_reset_drift")
    if len(resets) != contract["expected_discontinuity_reset_count"]:
        raise InputBlocked("discontinuity_reset_count_drift")
    if sum(row["gap_ns"] > 1_000_000_000 for row in resets) != contract[
        "expected_discontinuity_over_one_second_count"
    ]:
        raise InputBlocked("discontinuity_over_one_second_count_drift")
    return groups, resets


def load_route_map(config: dict[str, Any], repo: Path) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    result: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for binding in config["input_contract"]["causal_route"]:
        payload = load_json(repo / binding["path"])
        if payload.get("schema") == "blindassist_ustrf_route_role_review_bundle_r1":
            allowed = set(binding["allowed_frame_fields"])
            forbidden = set(binding["forbidden_frame_fields"])
            for source in payload["sources"]:
                for window in source["windows"]:
                    for row in window["frames"]:
                        if not allowed.issubset(row):
                            raise InputBlocked("LILocBench_route_projection_contract_drift")
                        projected = {field: row.get(field) for field in allowed}
                        key = (
                            source["source_id"],
                            source["source_id"],
                            int(projected["frame_id"]),
                            int(projected["source_capture_timestamp_ns"]),
                        )
                        value = {
                            "status": projected["route_status"],
                            "uv": projected["route_uv"],
                            "route_receipt_id": projected["route_receipt_id"],
                            "route_evidence_age_ms": projected["route_evidence_age_ms"],
                        }
                        if key in result and result[key] != value:
                            raise InputBlocked(f"LILocBench_duplicate_route_drift:{key}")
                        result[key] = value
        else:
            for source in payload["sources"]:
                for sequence in source["sequences"]:
                    for row in sequence["route_predictions"]:
                        key = (
                            source["source_id"],
                            sequence["sequence_id"],
                            int(row["frame_id"]),
                            int(row["source_capture_timestamp_ns"]),
                        )
                        result[key] = {
                            "status": row["status"],
                            "uv": row.get("uv"),
                            "predicted_at_ns": row.get("predicted_at_ns"),
                        }
    return result


def compact_paths(output_root: Path, source_id: str, sequence_id: str) -> tuple[Path, Path]:
    slug = stable_slug(source_id, sequence_id)
    return (
        output_root / "detector-ledgers" / f"{slug}.json",
        output_root / "detector-ledgers" / f"{slug}.successor-receipt.json",
    )


def validate_compact_ledger(
    ledger_path: Path,
    successor_path: Path,
    descriptor: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bool:
    if not ledger_path.is_file() or not successor_path.is_file():
        return False
    try:
        ledger = load_json(ledger_path)
        successor = load_json(successor_path)
        if ledger.get("schema") != COMPACT_SCHEMA or successor.get("schema") != SUCCESSOR_SCHEMA:
            return False
        if ledger.get("stage") != "R2-L1E" or successor.get("stage") != "R2-L1E":
            return False
        if (
            ledger.get("authority")
            != "candidate_input_only_no_selection_android_h2_human_or_production_authority"
        ):
            return False
        if successor.get("validated") is not True:
            return False
        if successor.get("compact_ledger_sha256") != sha256_file(ledger_path):
            return False
        if successor.get("raw_sha256") != ledger.get("canonical_raw_sha256"):
            return False
        if not isinstance(successor.get("raw_retained_as_parent_evidence"), bool):
            return False
        for field in ("source_id", "sequence_id", "frame_mask_sha256", "frame_count"):
            if ledger.get(field) != successor.get(field):
                return False
        if (
            ledger.get("source_id"),
            ledger.get("sequence_id"),
        ) != (descriptor["source_id"], descriptor["sequence_id"]):
            return False
        if ledger.get("frame_mask_sha256") != descriptor["frame_mask_sha256"]:
            return False
        if ledger.get("frame_count") != len(rows) or len(ledger.get("frames", [])) != len(rows):
            return False
        if not re.fullmatch(r"[0-9a-f]{64}", str(ledger.get("canonical_raw_sha256", ""))):
            return False
        if not re.fullmatch(r"[0-9a-f]{64}", str(ledger.get("device_receipt_sha256", ""))):
            return False
        if successor.get("device_receipt_sha256") not in {
            None,
            ledger.get("device_receipt_sha256"),
        }:
            return False
        if successor.get("device_manifest_sha256") not in {
            None,
            ledger.get("device_manifest_sha256"),
        }:
            return False
        for expected, actual in zip(rows, ledger["frames"], strict=True):
            if identity(expected) != identity(actual):
                return False
            assert_candidate_input_uncontaminated(actual)
            if actual.get("source_size") != [640, 480]:
                return False
            if not re.fullmatch(
                r"[0-9a-f]{64}", str(actual.get("android_raw_output_sha256", ""))
            ):
                return False
            detections = actual.get("person_detections")
            if not isinstance(detections, list):
                return False
            for detection in detections:
                if set(detection) != {
                    "prediction_index",
                    "class_id",
                    "label",
                    "confidence",
                    "box",
                }:
                    return False
                if detection["class_id"] != 0 or detection["label"] != "person":
                    return False
                if (
                    not isinstance(detection["box"], list)
                    or len(detection["box"]) != 4
                    or not all(isinstance(value, (int, float)) for value in detection["box"])
                ):
                    return False
                if not isinstance(detection["confidence"], (int, float)):
                    return False
            latency = actual.get("detector_processing_latency_ns")
            if latency is not None and (not isinstance(latency, int) or latency < 0):
                return False
            decode_latency = actual.get("host_decode_latency_ns")
            if decode_latency is not None and (
                not isinstance(decode_latency, int) or decode_latency < 0
            ):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def detector_labels(config: dict[str, Any], repo: Path) -> list[str]:
    labels_path = repo / config["input_contract"]["detector"]["labels_path"]
    return labels_path.read_text(encoding="utf-8").splitlines()


def decode_raw_record(
    raw_bytes: bytes,
    source_size: list[int],
    labels: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    detector = config["input_contract"]["detector"]
    raw = np.frombuffer(raw_bytes, dtype="<f4").reshape((1, 84, 2100))
    width, height = map(int, source_size)
    scale = min(320.0 / width, 320.0 / height)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    transform = (scale, (320 - resized_width) / 2.0, (320 - resized_height) / 2.0)
    detections, _ = decode(
        raw,
        (width, height),
        transform,
        labels,
        float(detector["confidence_threshold"]),
        float(detector["nms_iou_threshold"]),
        320,
    )
    return [item for item in detections if item["class_id"] == detector["person_class_index"]]


def read_exact(stream: gzip.GzipFile, count: int, label: str) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise InputBlocked(f"canonical_raw_truncated:{label}:missing={remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def materialize_existing_lilocbench(
    config: dict[str, Any],
    repo: Path,
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    output_root: Path,
) -> None:
    detector = config["input_contract"]["detector"]
    receipt_path = repo / detector["existing_lilocbench_device_receipt_path"]
    raw_path = repo / detector["existing_lilocbench_raw_path"]
    if sha256_file(receipt_path) != detector["existing_lilocbench_device_receipt_sha256"]:
        raise InputBlocked("existing_lilocbench_device_receipt_sha256_mismatch")
    if sha256_file(raw_path) != detector["existing_lilocbench_raw_sha256"]:
        raise InputBlocked("existing_lilocbench_raw_sha256_mismatch")
    receipt = load_json(receipt_path)
    group_by_key = {
        (descriptor["source_id"], descriptor["sequence_id"]): (descriptor, rows)
        for descriptor, rows in groups
        if descriptor["source_id"].startswith("lilocbench_")
    }
    if all(
        validate_compact_ledger(
            *compact_paths(output_root, descriptor["source_id"], descriptor["sequence_id"]),
            descriptor,
            rows,
        )
        for descriptor, rows in group_by_key.values()
    ):
        return
    if sum(len(rows) for _, rows in group_by_key.values()) != receipt["frame_count"]:
        raise InputBlocked("existing_lilocbench_frame_count_mismatch")
    labels = detector_labels(config, repo)
    outputs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    expected_by_identity = {identity(row): row for _, rows in group_by_key.values() for row in rows}
    expected_by_triplet = {row_identity[:3]: row_identity for row_identity in expected_by_identity}
    with gzip.open(raw_path, "rb") as stream:
        for device_row in receipt["frames"]:
            source_id = LILOCBENCH_ALIASES[device_row["source_name"]]
            key = (source_id, source_id, int(device_row["frame_id"]))
            if key not in expected_by_triplet:
                raise InputBlocked(f"existing_lilocbench_identity_mismatch:{key}")
            row_identity = expected_by_triplet[key]
            raw_bytes = read_exact(stream, RAW_BYTES_PER_FRAME, "/".join(map(str, key)))
            raw_sha = hashlib.sha256(raw_bytes).hexdigest()
            if raw_sha != device_row["android_raw_output_sha256"]:
                raise InputBlocked(f"existing_lilocbench_raw_record_hash_mismatch:{key}")
            decode_started = time.perf_counter_ns()
            person_detections = decode_raw_record(raw_bytes, [640, 480], labels, config)
            decode_latency = time.perf_counter_ns() - decode_started
            enforce_host_rss_guard(config)
            outputs[(source_id, source_id)].append(
                {
                    **expected_by_identity[row_identity],
                    "source_size": [640, 480],
                    "android_raw_output_sha256": raw_sha,
                    "host_decode_latency_ns": decode_latency,
                    "person_detections": person_detections,
                }
            )
        if stream.read(1):
            raise InputBlocked("existing_lilocbench_raw_has_trailing_records")
    for key, frames in outputs.items():
        descriptor, expected_rows = group_by_key[key]
        ledger_path, successor_path = compact_paths(output_root, *key)
        if validate_compact_ledger(ledger_path, successor_path, descriptor, expected_rows):
            continue
        payload = {
            "schema": COMPACT_SCHEMA,
            "stage": "R2-L1E",
            "authority": "candidate_input_only_no_selection_android_h2_human_or_production_authority",
            "source_id": key[0],
            "sequence_id": key[1],
            "frame_mask_sha256": descriptor["frame_mask_sha256"],
            "frame_count": len(frames),
            "canonical_raw_source": "preexisting_frozen_lilocbench_android_canvas_stream",
            "canonical_raw_sha256": sha256_file(raw_path),
            "device_receipt_sha256": sha256_file(receipt_path),
            "frames": frames,
        }
        atomic_write_json(ledger_path, payload)
        successor = {
            "schema": SUCCESSOR_SCHEMA,
            "stage": "R2-L1E",
            "source_id": key[0],
            "sequence_id": key[1],
            "frame_mask_sha256": descriptor["frame_mask_sha256"],
            "raw_sha256": sha256_file(raw_path),
            "raw_retained_as_parent_evidence": True,
            "compact_ledger_sha256": sha256_file(ledger_path),
            "frame_count": len(frames),
            "validated": True,
        }
        atomic_write_json(successor_path, successor)
        if not validate_compact_ledger(ledger_path, successor_path, descriptor, expected_rows):
            raise ExecutionAborted(f"LILocBench successor validation failed:{key}")


def find_crowdbot_bundle(
    repo: Path, config: dict[str, Any], source_id: str, sequence_id: str
) -> Path:
    for root in config["input_contract"]["detector"]["crowdbot_dataset_roots"]:
        candidate = repo / root / source_id / "sequences" / sequence_id / "bundle.json"
        if candidate.is_file():
            return candidate
    raise InputBlocked(f"crowdbot_bundle_missing:{source_id}/{sequence_id}")


def load_crowdbot_images(
    repo: Path,
    config: dict[str, Any],
    descriptor: dict[str, Any],
    mask_rows: list[dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]]]:
    bundle_path = find_crowdbot_bundle(
        repo, config, descriptor["source_id"], descriptor["sequence_id"]
    )
    bundle = load_json(bundle_path)
    if bundle.get("candidate_outputs_executed") is not False:
        raise InputBlocked("crowdbot_bundle_candidate_contamination")
    frames_path = Path(bundle["frames_path"])
    if sha256_file(frames_path) != bundle["frames_sha256"]:
        raise InputBlocked(f"crowdbot_frames_sha256_mismatch:{descriptor['sequence_id']}")
    frame_rows = {
        int(row["frame_id"]): row
        for row in map(json.loads, frames_path.read_text(encoding="utf-8").splitlines())
    }
    result = []
    for expected in mask_rows:
        frame = frame_rows.get(int(expected["frame_id"]))
        if frame is None or identity(frame) != identity(expected):
            raise InputBlocked(
                f"crowdbot_frame_membership_mismatch:{descriptor['sequence_id']}/{expected['frame_id']}"
            )
        image = frames_path.parent / frame["rgb_path"]
        if not image.is_file() or sha256_file(image) != frame["rgb_sha256"]:
            raise InputBlocked(
                f"crowdbot_rgb_missing_or_hash_mismatch:{descriptor['sequence_id']}/{expected['frame_id']}"
            )
        result.append(
            {
                **expected,
                "image_path": f"images/{image.name}",
                "image_sha256": frame["rgb_sha256"],
                "_host_image_path": str(image),
            }
        )
    return bundle_path, result


def available_memory_bytes() -> int:
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ExecutionAborted("GlobalMemoryStatusEx_failed")
        return int(status.available_physical)
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, ValueError, OSError) as error:
        raise ExecutionAborted("available_memory_guard_unavailable") from error


def current_rss_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel32.GetCurrentProcess()
        get_memory_info = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if get_memory_info is None:
            get_memory_info = ctypes.WinDLL(
                "psapi", use_last_error=True
            ).GetProcessMemoryInfo
        get_memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory_info.restype = wintypes.BOOL
        if not get_memory_info(process, ctypes.byref(counters), counters.cb):
            raise ExecutionAborted("GetProcessMemoryInfo_failed")
        return int(counters.WorkingSetSize)
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except (ImportError, ValueError, OSError) as error:
        raise ExecutionAborted("host_rss_guard_unavailable") from error


def enforce_host_rss_guard(config: dict[str, Any]) -> int:
    observed = current_rss_bytes()
    maximum = int(config["resource_guards"]["host_maximum_rss_bytes"])
    if observed > maximum:
        raise ExecutionAborted(
            f"host_rss_guard:observed={observed}:maximum={maximum}"
        )
    return observed


def disk_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def run_command(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise ExecutionAborted(
            f"command_failed:{command[0]}:exit={result.returncode}:"
            f"stdout={result.stdout[-1000:]}:stderr={result.stderr[-1000:]}"
        )
    return result


def locate_adb(repo: Path) -> Path:
    candidates = [
        repo / ".android-sdk/platform-tools/adb.exe",
        Path(r"E:\codex-tools\projects\blindassist\toolchain\android-sdk\platform-tools\adb.exe"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise ExecutionAborted("adb_unavailable")


def remote_remove(adb: Path, remote_relative: str) -> None:
    if not re.fullmatch(r"r2l1e/[A-Za-z0-9._-]+", remote_relative):
        raise ExecutionAborted(f"unsafe_remote_cleanup_target:{remote_relative}")
    absolute = f"/sdcard/Android/data/com.linnan.blindassist/files/{remote_relative}"
    run_command([str(adb), "shell", "rm", "-rf", absolute], timeout=120)


def bounded_local_cleanup(path: Path, retry_count: int) -> None:
    errors = []
    for _ in range(1 + retry_count):
        try:
            path.unlink(missing_ok=True)
            if not path.exists():
                return
        except OSError as error:
            errors.append(str(error))
    raise ExecutionAborted(
        f"bounded_local_cleanup_failed:{path}:{' | '.join(errors)}"
    )


def bounded_remote_cleanup(
    adb: Path, remote_relative: str, retry_count: int
) -> None:
    errors = []
    for _ in range(1 + retry_count):
        try:
            remote_remove(adb, remote_relative)
            return
        except ExecutionAborted as error:
            errors.append(str(error))
    raise ExecutionAborted(
        f"bounded_remote_cleanup_failed:{remote_relative}:{' | '.join(errors)}"
    )


def validate_device_receipt(
    receipt: dict[str, Any],
    manifest_path: Path,
    raw_path: Path,
    descriptor: dict[str, Any],
    mask_rows: list[dict[str, Any]],
) -> None:
    if receipt.get("schema") != "blindassist_ustrf_route_target_l1e_device_raw_receipt_r1":
        raise ExecutionAborted("unexpected_device_receipt_schema")
    if receipt.get("status") != "DEVICE_RAW_SHARD_COMPLETE":
        raise ExecutionAborted(f"device_raw_shard_incomplete:{receipt.get('error')}")
    if receipt.get("input_manifest_sha256") != sha256_file(manifest_path):
        raise ExecutionAborted("device_manifest_binding_mismatch")
    if receipt.get("frame_mask_sha256") != descriptor["frame_mask_sha256"]:
        raise ExecutionAborted("device_frame_mask_binding_mismatch")
    if receipt.get("completed_frame_count") != len(mask_rows):
        raise ExecutionAborted("device_frame_count_mismatch")
    raw_contract = receipt["canonical_raw_stream"]
    if raw_contract.get("bytes_per_frame_uncompressed") != RAW_BYTES_PER_FRAME:
        raise ExecutionAborted("device_raw_record_size_drift")
    if raw_contract.get("compressed_sha256") != sha256_file(raw_path):
        raise ExecutionAborted("device_raw_compressed_sha256_mismatch")
    if [identity(row) for row in receipt["frames"]] != [identity(row) for row in mask_rows]:
        raise ExecutionAborted("device_raw_frame_identity_or_order_mismatch")
    if any(
        not isinstance(row.get("detector_processing_latency_ns"), int)
        or row["detector_processing_latency_ns"] < 0
        for row in receipt["frames"]
    ):
        raise ExecutionAborted("device_detector_processing_latency_missing_or_invalid")
    guards = receipt["guards"]
    if guards["observed_maximum_thermal_status"] > guards["maximum_thermal_status"]:
        raise ExecutionAborted("device_thermal_guard_exceeded")
    if guards["final_battery_temperature_c"] > guards["maximum_battery_temperature_c"]:
        raise ExecutionAborted("device_final_battery_temperature_guard_exceeded")
    if (
        guards["observed_maximum_battery_temperature_c"]
        - guards["start_battery_temperature_c"]
        > guards["maximum_battery_temperature_rise_c"]
    ):
        raise ExecutionAborted("device_battery_temperature_rise_guard_exceeded")


def materialize_one_crowdbot(
    config: dict[str, Any],
    bindings: dict[str, str],
    repo: Path,
    descriptor: dict[str, Any],
    mask_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    ledger_path, successor_path = compact_paths(
        output_root, descriptor["source_id"], descriptor["sequence_id"]
    )
    if validate_compact_ledger(ledger_path, successor_path, descriptor, mask_rows):
        return
    bundle_path, image_rows = load_crowdbot_images(repo, config, descriptor, mask_rows)
    projected_bytes = len(mask_rows) * RAW_BYTES_PER_FRAME
    guards = config["resource_guards"]
    required_free = 2 * projected_bytes + int(guards["reserve_bytes"])
    observed_free = disk_free_bytes(output_root)
    if observed_free < required_free:
        raise ExecutionAborted(
            f"free_space_guard:observed={observed_free}:required={required_free}"
        )
    observed_memory = available_memory_bytes()
    if observed_memory < int(guards["minimum_system_available_physical_memory_bytes"]):
        record_resource_guard_failure(
            output_root / "resource-guard-attempts-r1.json",
            config,
            bindings,
            observed_memory,
        )
        raise ExecutionAborted(
            f"available_memory_guard:observed={observed_memory}:required="
            f"{guards['minimum_system_available_physical_memory_bytes']}"
        )
    slug = stable_slug(descriptor["source_id"], descriptor["sequence_id"])
    attempt_root = output_root / "attempts" / slug
    existing_attempts = sorted(attempt_root.glob("attempt-*")) if attempt_root.exists() else []
    maximum_attempts = 1 + int(guards["maximum_retry_count_after_initial_attempt"])
    if len(existing_attempts) >= maximum_attempts:
        raise ExecutionAborted(f"retry_limit_exhausted:{slug}")
    attempt_number = len(existing_attempts) + 1
    attempt_dir = attempt_root / f"attempt-{attempt_number:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = attempt_dir / "device-manifest.json"
    raw_path = attempt_dir / "canonical-raw.gz"
    device_receipt_path = attempt_dir / "device-raw-receipt.json"
    manifest = {
        "schema": DEVICE_MANIFEST_SCHEMA,
        "stage": "R2-L1E",
        "source_id": descriptor["source_id"],
        "sequence_id": descriptor["sequence_id"],
        "frame_mask_sha256": descriptor["frame_mask_sha256"],
        "frame_count": len(mask_rows),
        "input_shape": config["input_contract"]["detector"]["input_shape"],
        "output_shape": config["input_contract"]["detector"]["output_shape"],
        "model_sha256": config["input_contract"]["detector"]["model_sha256"],
        "labels_sha256": config["input_contract"]["detector"]["labels_sha256"],
        "person_class_index": 0,
        "confidence_threshold": 0.35,
        "nms_iou_threshold": 0.45,
        "bundle_sha256": sha256_file(bundle_path),
        "projected_raw_bytes": projected_bytes,
        "free_space_guard": {
            "observed_bytes": observed_free,
            "required_bytes": required_free,
            "formula": guards["minimum_free_space_formula"],
        },
        "memory_guard": {
            "observed_available_bytes": observed_memory,
            "required_available_bytes": guards[
                "minimum_system_available_physical_memory_bytes"
            ],
            "maximum_host_rss_bytes": guards["host_maximum_rss_bytes"],
        },
        "frames": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in image_rows
        ],
        "authority": {
            "candidate_input_only": True,
            "selection": False,
            "android_shadow": False,
            "h2": False,
            "human_outcome": False,
            "production": False,
        },
    }
    atomic_write_json(manifest_path, manifest)
    adb = locate_adb(repo)
    remote_relative = f"r2l1e/{slug}"
    remote_absolute = f"/sdcard/Android/data/com.linnan.blindassist/files/{remote_relative}"
    bounded_remote_cleanup(
        adb, remote_relative, int(guards["cleanup_retry_count"])
    )
    run_command([str(adb), "shell", "mkdir", "-p", f"{remote_absolute}/images"], timeout=120)
    run_command([str(adb), "push", str(manifest_path), f"{remote_absolute}/manifest.json"], timeout=300)
    image_directory = Path(image_rows[0]["_host_image_path"]).parent
    run_command([str(adb), "push", f"{image_directory}{os.sep}.", f"{remote_absolute}/images/"], timeout=1800)
    instrumentation = [
        str(adb),
        "shell",
        "am",
        "instrument",
        "-w",
        "-r",
        "-e",
        "class",
        "com.linnan.blindassist.benchmark.UstrfR2L1ExploratoryProfileDeviceTest",
        "-e",
        "ustrfR2L1eRequired",
        "true",
        "-e",
        "ustrfR2L1eInput",
        f"{remote_relative}/manifest.json",
        "-e",
        "ustrfR2L1eRawOutput",
        f"{remote_relative}/canonical-raw.gz",
        "-e",
        "ustrfR2L1eReceiptOutput",
        f"{remote_relative}/device-raw-receipt.json",
        "com.linnan.blindassist.benchmark/androidx.test.runner.AndroidJUnitRunner",
    ]
    started = time.monotonic()
    process = run_command(instrumentation, timeout=3600)
    (attempt_dir / "instrumentation.stdout.txt").write_text(process.stdout, encoding="utf-8")
    run_command(
        [str(adb), "pull", f"{remote_absolute}/device-raw-receipt.json", str(device_receipt_path)],
        timeout=300,
    )
    run_command(
        [str(adb), "pull", f"{remote_absolute}/canonical-raw.gz", str(raw_path)],
        timeout=1800,
    )
    receipt = load_json(device_receipt_path)
    validate_device_receipt(receipt, manifest_path, raw_path, descriptor, mask_rows)
    labels = detector_labels(config, repo)
    frames = []
    raw_digest = hashlib.sha256()
    with gzip.open(raw_path, "rb") as stream:
        for expected, device_row in zip(mask_rows, receipt["frames"], strict=True):
            raw_bytes = read_exact(
                stream,
                RAW_BYTES_PER_FRAME,
                f"{descriptor['sequence_id']}/{expected['frame_id']}",
            )
            raw_digest.update(raw_bytes)
            raw_sha = hashlib.sha256(raw_bytes).hexdigest()
            if raw_sha != device_row["android_raw_output_sha256"]:
                raise ExecutionAborted("device_raw_record_sha256_mismatch")
            detector_latency = int(device_row["detector_processing_latency_ns"])
            decode_started = time.perf_counter_ns()
            person_detections = decode_raw_record(raw_bytes, [640, 480], labels, config)
            decode_latency = time.perf_counter_ns() - decode_started
            enforce_host_rss_guard(config)
            frames.append(
                {
                    **expected,
                    "source_size": [640, 480],
                    "android_raw_output_sha256": raw_sha,
                    "detector_processing_latency_ns": detector_latency,
                    "host_decode_latency_ns": decode_latency,
                    "person_detections": person_detections,
                }
            )
        if stream.read(1):
            raise ExecutionAborted("device_raw_stream_has_trailing_records")
    if raw_digest.hexdigest() != receipt["canonical_raw_stream"]["uncompressed_sha256"]:
        raise ExecutionAborted("device_raw_uncompressed_sha256_mismatch")
    ledger = {
        "schema": COMPACT_SCHEMA,
        "stage": "R2-L1E",
        "authority": "candidate_input_only_no_selection_android_h2_human_or_production_authority",
        "source_id": descriptor["source_id"],
        "sequence_id": descriptor["sequence_id"],
        "frame_mask_sha256": descriptor["frame_mask_sha256"],
        "frame_count": len(frames),
        "canonical_raw_source": "R2_L1E_sequence_sharded_android_canvas_stream",
        "canonical_raw_sha256": sha256_file(raw_path),
        "device_receipt_sha256": sha256_file(device_receipt_path),
        "device_manifest_sha256": sha256_file(manifest_path),
        "frames": frames,
    }
    atomic_write_json(ledger_path, ledger)
    successor = {
        "schema": SUCCESSOR_SCHEMA,
        "stage": "R2-L1E",
        "source_id": descriptor["source_id"],
        "sequence_id": descriptor["sequence_id"],
        "frame_mask_sha256": descriptor["frame_mask_sha256"],
        "raw_sha256": sha256_file(raw_path),
        "raw_retained_as_parent_evidence": False,
        "device_receipt_sha256": sha256_file(device_receipt_path),
        "device_manifest_sha256": sha256_file(manifest_path),
        "compact_ledger_sha256": sha256_file(ledger_path),
        "frame_count": len(frames),
        "wall_time_seconds": time.monotonic() - started,
        "projected_raw_bytes": projected_bytes,
        "validated": True,
    }
    atomic_write_json(successor_path, successor)
    if not validate_compact_ledger(ledger_path, successor_path, descriptor, mask_rows):
        raise ExecutionAborted("compact_successor_validation_failed")
    cleanup_retries = int(config["resource_guards"]["cleanup_retry_count"])
    bounded_local_cleanup(raw_path, cleanup_retries)
    bounded_remote_cleanup(adb, remote_relative, cleanup_retries)


def gap_matrix(
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    output_root: Path,
    route_map: dict[tuple[str, str, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for descriptor, rows in groups:
        ledger_path, successor_path = compact_paths(
            output_root, descriptor["source_id"], descriptor["sequence_id"]
        )
        detector_complete = validate_compact_ledger(
            ledger_path, successor_path, descriptor, rows
        )
        route_missing = 0
        for row in rows:
            key = identity(row)
            if key not in route_map and (key[0], key[1], key[2], -1) not in route_map:
                route_missing += 1
        missing_fields = []
        if not detector_complete:
            missing_fields.append("android_canvas_canonical_detector_raw_successor")
        if route_missing:
            missing_fields.append("causal_route")
        missing_frame_rows = []
        if missing_fields:
            for row in rows:
                frame_missing = []
                if not detector_complete:
                    frame_missing.append("android_canvas_canonical_detector_raw_successor")
                if (
                    identity(row) not in route_map
                    and (row["source_id"], row["sequence_id"], int(row["frame_id"]), -1)
                    not in route_map
                ):
                    frame_missing.append("causal_route")
                if frame_missing:
                    missing_frame_rows.append(
                        {
                            "unit_id": row["unit_id"],
                            "source_id": row["source_id"],
                            "sequence_id": row["sequence_id"],
                            "frame_id": int(row["frame_id"]),
                            "source_capture_timestamp_ns": int(
                                row["source_capture_timestamp_ns"]
                            ),
                            "missing_fields": frame_missing,
                        }
                    )
        result.append(
            {
                "source_id": descriptor["source_id"],
                "sequence_id": descriptor["sequence_id"],
                "frame_mask_sha256": descriptor["frame_mask_sha256"],
                "expected_frame_count": len(rows),
                "canonical_detector_frame_count": len(rows) if detector_complete else 0,
                "causal_route_frame_count": len(rows) - route_missing,
                "capture_timestamp_frame_count": sum(
                    isinstance(row.get("source_capture_timestamp_ns"), int) for row in rows
                ),
                "t0_association_contract_frozen": True,
                "candidate_consume_timestamp_runtime_field": True,
                "missing_fields": missing_fields,
                "missing_frame_rows": missing_frame_rows,
            }
        )
    return result


def collect_verified_input_artifacts(
    repo: Path,
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    output_root: Path,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for descriptor, rows in groups:
        ledger_path, successor_path = compact_paths(
            output_root, descriptor["source_id"], descriptor["sequence_id"]
        )
        if not validate_compact_ledger(ledger_path, successor_path, descriptor, rows):
            continue
        artifacts.append(
            {
                "source_id": descriptor["source_id"],
                "sequence_id": descriptor["sequence_id"],
                "frame_count": len(rows),
                "frame_mask_sha256": descriptor["frame_mask_sha256"],
                "compact_ledger_path": str(ledger_path.relative_to(repo)).replace(
                    "\\", "/"
                ),
                "compact_ledger_sha256": sha256_file(ledger_path),
                "successor_receipt_path": str(
                    successor_path.relative_to(repo)
                ).replace("\\", "/"),
                "successor_receipt_sha256": sha256_file(successor_path),
            }
        )
    return artifacts


def validate_exhausted_resource_guard_receipt(
    receipt_path: Path,
    config_sha256: str,
    current_implementation_bindings: dict[str, str],
    required_minimum_bytes: int,
    maximum_attempts: int,
) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    if receipt.get("schema") != RESOURCE_GUARD_SCHEMA or receipt.get("stage") != "R2-L1E":
        raise ExecutionAborted("resource_guard_receipt_schema_mismatch")
    if receipt.get("config_sha256") != config_sha256:
        raise ExecutionAborted("resource_guard_receipt_config_binding_mismatch")
    if receipt.get("implementation_bindings") != current_implementation_bindings:
        raise ExecutionAborted("resource_guard_receipt_implementation_binding_mismatch")
    if receipt.get("guard") != "system_available_physical_memory_bytes":
        raise ExecutionAborted("resource_guard_receipt_guard_mismatch")
    if receipt.get("required_minimum_bytes") != required_minimum_bytes:
        raise ExecutionAborted("resource_guard_receipt_threshold_mismatch")
    if receipt.get("maximum_attempts") != maximum_attempts:
        raise ExecutionAborted("resource_guard_receipt_attempt_limit_mismatch")
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != maximum_attempts:
        raise ExecutionAborted("resource_guard_attempt_history_incomplete")
    expected_numbers = list(range(1, maximum_attempts + 1))
    if [row.get("attempt_number") for row in attempts] != expected_numbers:
        raise ExecutionAborted("resource_guard_attempt_order_mismatch")
    attempt_ids: set[str] = set()
    for row in attempts:
        required = {
            "attempt_number",
            "attempt_id",
            "observation_time_utc",
            "observation_time_status",
            "observed_available_bytes",
            "required_available_bytes",
            "wall_time_seconds",
            "process_exit_code",
            "system_event",
            "last_safe_checkpoint",
            "outcome",
        }
        if set(row) != required:
            raise ExecutionAborted("resource_guard_attempt_field_contract_mismatch")
        attempt_id = row["attempt_id"]
        if not isinstance(attempt_id, str) or not attempt_id or attempt_id in attempt_ids:
            raise ExecutionAborted("resource_guard_attempt_id_invalid")
        attempt_ids.add(attempt_id)
        if row["required_available_bytes"] != required_minimum_bytes:
            raise ExecutionAborted("resource_guard_attempt_threshold_drift")
        if (
            not isinstance(row["observed_available_bytes"], int)
            or row["observed_available_bytes"] >= required_minimum_bytes
        ):
            raise ExecutionAborted("resource_guard_attempt_not_a_guard_failure")
        if row["outcome"] != "STOPPED_BEFORE_DEVICE_ATTEMPT":
            raise ExecutionAborted("resource_guard_attempt_outcome_mismatch")
        if row["last_safe_checkpoint"] != {
            "verified_sequence_ledgers": 2,
            "verified_frames": 4594,
            "candidate_execution_started": False,
        }:
            raise ExecutionAborted("resource_guard_last_checkpoint_drift")
    required_flags = {
        "device_attempt_created": False,
        "canonical_raw_shard_created": False,
        "candidate_execution_started": False,
        "candidate_trace_created": False,
        "profile_authority": False,
        "automatic_retry_allowed_after_receipt": False,
        "retry_limit_exhausted": True,
    }
    for key, expected in required_flags.items():
        if receipt.get(key) is not expected:
            raise ExecutionAborted(f"resource_guard_receipt_flag_mismatch:{key}")
    return receipt


def record_resource_guard_failure(
    receipt_path: Path,
    config: dict[str, Any],
    bindings: dict[str, str],
    observed_available_bytes: int,
) -> dict[str, Any]:
    maximum_attempts = 1 + int(
        config["resource_guards"]["maximum_retry_count_after_initial_attempt"]
    )
    required_bytes = int(
        config["resource_guards"]["minimum_system_available_physical_memory_bytes"]
    )
    current_implementations = {
        key: value
        for key, value in bindings.items()
        if key.endswith("_implementation_sha256") or key == "terminal_schema_sha256"
    }
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if receipt.get("automatic_retry_allowed_after_receipt") is False:
            raise ExecutionAborted("resource_guard_retry_limit_already_exhausted")
        if receipt.get("config_sha256") != bindings["config_sha256"]:
            raise ExecutionAborted("resource_guard_history_config_binding_mismatch")
        if receipt.get("implementation_bindings") != current_implementations:
            raise ExecutionAborted(
                "resource_guard_history_implementation_binding_mismatch"
            )
    else:
        receipt = {
            "schema": RESOURCE_GUARD_SCHEMA,
            "stage": "R2-L1E",
            "recorded_on": datetime.now(timezone.utc).date().isoformat(),
            "config_sha256": bindings["config_sha256"],
            "implementation_bindings": current_implementations,
            "guard": "system_available_physical_memory_bytes",
            "required_minimum_bytes": required_bytes,
            "maximum_attempts": maximum_attempts,
            "attempts": [],
            "device_attempt_created": False,
            "canonical_raw_shard_created": False,
            "candidate_execution_started": False,
            "candidate_trace_created": False,
            "profile_authority": False,
            "automatic_retry_allowed_after_receipt": True,
            "retry_limit_exhausted": False,
        }
    if len(receipt["attempts"]) >= maximum_attempts:
        raise ExecutionAborted("resource_guard_retry_limit_already_exhausted")
    attempt_number = len(receipt["attempts"]) + 1
    receipt["attempts"].append(
        {
            "attempt_number": attempt_number,
            "attempt_id": f"host-memory-pre-device-{attempt_number:03d}-{uuid.uuid4().hex}",
            "observation_time_utc": datetime.now(timezone.utc).isoformat(),
            "observation_time_status": "recorded_at_guard_evaluation",
            "observed_available_bytes": observed_available_bytes,
            "required_available_bytes": required_bytes,
            "wall_time_seconds": 0.0,
            "process_exit_code": None,
            "system_event": "host_pre_device_memory_guard",
            "last_safe_checkpoint": {
                "verified_sequence_ledgers": 2,
                "verified_frames": 4594,
                "candidate_execution_started": False,
            },
            "outcome": "STOPPED_BEFORE_DEVICE_ATTEMPT",
        }
    )
    exhausted = len(receipt["attempts"]) == maximum_attempts
    receipt["automatic_retry_allowed_after_receipt"] = not exhausted
    receipt["retry_limit_exhausted"] = exhausted
    atomic_write_json(receipt_path, receipt)
    return receipt


def base_terminal_receipt(
    terminal_state: str,
    bindings: dict[str, str],
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    resets: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    config: dict[str, Any],
    first_blocker: str | None,
    verified_input_artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if terminal_state not in TERMINAL_STATES:
        raise ValueError(terminal_state)
    verified_frames = sum(
        row["expected_frame_count"]
        for row in gaps
        if not row["missing_fields"]
    )
    return {
        "schema": TERMINAL_SCHEMA,
        "stage": "R2-L1E",
        "terminal_state": terminal_state,
        "authority": "l1_exploratory_profiles_only_no_selection_android_h2_human_or_production_authority",
        "bindings": bindings,
        "verified_scope": {
            "expected_sequence_ledgers": len(groups),
            "expected_frames": sum(len(rows) for _, rows in groups),
            "fully_input_verified_sequence_ledgers": sum(
                not row["missing_fields"] for row in gaps
            ),
            "fully_input_verified_frames": verified_frames,
            "first_blocker": first_blocker,
        },
        "verified_input_artifacts": verified_input_artifacts or [],
        "discontinuity_resets": resets,
        "gap_matrix": gaps,
        "candidate_execution": {
            "started": False,
            "candidate_order": config["candidate_roster"],
            "authoritative_trace_count": 0,
            "partial_trace_evaluation_authority": False,
        },
        "profiles": [],
        "guards": config["resource_guards"],
        "claim_boundary": {
            "selection_allowed": False,
            "candidate_comparison_allowed": False,
            "android_shadow_allowed": False,
            "h2_allowed": False,
            "human_outcome_allowed": False,
            "production_allowed": False,
            "new_data_added": False,
        },
    }


def candidate_is_active(state: Any) -> bool:
    return bool(getattr(state, "active", False))


def replay_candidate_ledger(
    candidate_id: str,
    ledger: dict[str, Any],
    route_map: dict[tuple[str, str, int, int], dict[str, Any]],
    reset_next_identities: set[tuple[str, str, int, int]],
    tracker_config: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    if candidate_id not in CANDIDATE_CLASSES:
        raise ExecutionAborted(f"unknown_candidate:{candidate_id}")
    fixed_kernel = tracker_config["fixed_kernel"]
    maximum_rss = enforce_host_rss_guard(config)
    state: Any = None
    tracker: ArmState | None = None
    histories: dict[int, deque[tuple[int, float]]] | None = None
    trace: list[dict[str, Any]] = []
    reset_count = 0
    for index, frame in enumerate(ledger["frames"]):
        frame_identity = identity(frame)
        reset_here = index == 0 or frame_identity in reset_next_identities
        if reset_here:
            state = CANDIDATE_CLASSES[candidate_id](
                int(fixed_kernel["min_alert_frames"]),
                int(fixed_kernel["min_clear_frames"]),
            )
            tracker = ArmState()
            histories = defaultdict(lambda: deque(maxlen=3))
            if index > 0:
                reset_count += 1
        assert tracker is not None and histories is not None
        route = route_map.get(frame_identity)
        if route is None:
            raise ExecutionAborted(
                "candidate_route_missing:"
                + "/".join(map(str, frame_identity))
            )
        candidate_input = {
            "frame_id": int(frame["frame_id"]),
            "source_capture_timestamp_ns": int(
                frame["source_capture_timestamp_ns"]
            ),
            "source_size": frame["source_size"],
            "person_detections": frame["person_detections"],
            "route": {
                "status": route.get("status"),
                "uv": route.get("uv"),
                "route_receipt_id": route.get("route_receipt_id"),
                "route_evidence_age_ms": route.get("route_evidence_age_ms"),
                "predicted_at_ns": route.get("predicted_at_ns"),
            },
        }
        assert_candidate_input_uncontaminated(candidate_input)
        processing_started = time.perf_counter_ns()
        observed_pairs = associate(
            candidate_input["person_detections"],
            candidate_input["frame_id"],
            "T0",
            tracker,
            tracker_config,
        )
        width, height = map(int, candidate_input["source_size"])
        route_known = (
            candidate_input["route"].get("status") == "known"
            and candidate_input["route"].get("uv") is not None
        )
        relations: dict[int, str | None] = {}
        observed_tracks = []
        for track, _ in observed_pairs:
            observed_tracks.append(
                {
                    "track_id": int(track.track_id),
                    "box": [float(value) for value in track.box],
                }
            )
            relations[int(track.track_id)] = relation_observation(
                track_id=int(track.track_id),
                frame_number=candidate_input["frame_id"],
                box=track.box,
                route=candidate_input["route"],
                width=width,
                height=height,
                route_intersects=route_hit(
                    track.box,
                    candidate_input["route"],
                    width,
                    height,
                    float(fixed_kernel["route_point_margin_fraction"]),
                ),
                histories=histories,
            )
        output = state.update(
            candidate_input["frame_id"], route_known, relations
        )
        active_relation_track_ids = sorted(
            track_id
            for track_id, relation in relations.items()
            if relation in {"route_intersecting", "approaching_route"}
        )
        delivery_track_ids = (
            sorted(int(value) for value in output["deliveries"])
            if candidate_id == "C1_CAUSAL_ROUTE_RELATION_FSM"
            else active_relation_track_ids
        )
        replay_latency = time.perf_counter_ns() - processing_started
        detector_latency = frame.get("detector_processing_latency_ns")
        host_decode_latency = frame.get("host_decode_latency_ns")
        consume_timestamp = None
        if isinstance(detector_latency, int) and isinstance(host_decode_latency, int):
            consume_timestamp = (
                candidate_input["source_capture_timestamp_ns"]
                + detector_latency
                + host_decode_latency
                + replay_latency
            )
        maximum_rss = max(maximum_rss, enforce_host_rss_guard(config))
        trace.append(
            {
                "source_id": frame["source_id"],
                "sequence_id": frame["sequence_id"],
                "frame_id": candidate_input["frame_id"],
                "source_capture_timestamp_ns": candidate_input[
                    "source_capture_timestamp_ns"
                ],
                "candidate_consume_timestamp_ns": consume_timestamp,
                "state_reset_before_frame": reset_here,
                "route_known": route_known,
                "observed_tracks": observed_tracks,
                "active_relation_track_ids": active_relation_track_ids,
                "deliveries": [int(value) for value in output["deliveries"]],
                "delivery_track_ids": delivery_track_ids,
                "closures": [int(value) for value in output["closures"]],
                "candidate_active": candidate_is_active(state),
            }
        )
    return trace, maximum_rss


def trace_paths(
    output_root: Path, candidate_id: str, descriptor: dict[str, Any]
) -> tuple[Path, Path, Path]:
    slug = stable_slug(descriptor["source_id"], descriptor["sequence_id"])
    trace = output_root / "candidate-traces" / candidate_id / f"{slug}.json"
    receipt = trace.with_name(trace.stem + ".receipt.json")
    attempts = output_root / "candidate-trace-attempts" / candidate_id / slug
    return trace, receipt, attempts


def validate_trace_receipt(
    trace_path: Path,
    receipt_path: Path,
    candidate_id: str,
    descriptor: dict[str, Any],
    rows: list[dict[str, Any]],
    ledger_path: Path,
    successor_path: Path,
    bindings: dict[str, str],
) -> bool:
    if not trace_path.is_file() or not receipt_path.is_file():
        return False
    try:
        trace_payload = load_json(trace_path)
        receipt = load_json(receipt_path)
        if trace_payload.get("schema") != TRACE_SCHEMA:
            return False
        if receipt.get("schema") != TRACE_RECEIPT_SCHEMA:
            return False
        if receipt.get("status") != "FIRST_VALID_COMPLETE_TRACE":
            return False
        if trace_payload.get("candidate_id") != candidate_id:
            return False
        if trace_payload.get("frame_mask_sha256") != descriptor["frame_mask_sha256"]:
            return False
        frames = trace_payload.get("frames")
        if not isinstance(frames, list) or len(frames) != len(rows):
            return False
        if [identity(frame) for frame in frames] != [identity(row) for row in rows]:
            return False
        checks = {
            "candidate_id": candidate_id,
            "source_id": descriptor["source_id"],
            "sequence_id": descriptor["sequence_id"],
            "frame_mask_sha256": descriptor["frame_mask_sha256"],
            "frame_count": len(rows),
            "trace_sha256": sha256_file(trace_path),
            "compact_ledger_sha256": sha256_file(ledger_path),
            "successor_receipt_sha256": sha256_file(successor_path),
            "candidate_implementation_sha256": bindings[
                "candidate_implementation_sha256"
            ],
            "association_implementation_sha256": bindings[
                "association_implementation_sha256"
            ],
            "config_sha256": bindings["config_sha256"],
        }
        return all(receipt.get(key) == value for key, value in checks.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def materialize_candidate_trace(
    candidate_id: str,
    descriptor: dict[str, Any],
    rows: list[dict[str, Any]],
    route_map: dict[tuple[str, str, int, int], dict[str, Any]],
    reset_next_identities: set[tuple[str, str, int, int]],
    tracker_config: dict[str, Any],
    config: dict[str, Any],
    bindings: dict[str, str],
    repo: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    ledger_path, successor_path = compact_paths(
        output_root, descriptor["source_id"], descriptor["sequence_id"]
    )
    if not validate_compact_ledger(ledger_path, successor_path, descriptor, rows):
        raise ExecutionAborted("candidate_trace_input_successor_invalid")
    trace_path, receipt_path, attempts_root = trace_paths(
        output_root, candidate_id, descriptor
    )
    if validate_trace_receipt(
        trace_path,
        receipt_path,
        candidate_id,
        descriptor,
        rows,
        ledger_path,
        successor_path,
        bindings,
    ):
        return trace_path, receipt_path
    if trace_path.exists() or receipt_path.exists():
        raise ExecutionAborted("existing_candidate_trace_or_receipt_invalid")
    attempts_root.mkdir(parents=True, exist_ok=True)
    prior_attempts = sorted(attempts_root.glob("attempt-*.json"))
    maximum_attempts = 1 + int(
        config["resource_guards"]["maximum_retry_count_after_initial_attempt"]
    )
    if len(prior_attempts) >= maximum_attempts:
        raise ExecutionAborted("candidate_trace_retry_limit_exhausted")
    attempt_id = (
        f"{candidate_id}:{stable_slug(descriptor['source_id'], descriptor['sequence_id'])}:"
        f"{len(prior_attempts) + 1:03d}:{uuid.uuid4().hex}"
    )
    attempt_path = attempts_root / f"attempt-{len(prior_attempts) + 1:03d}.json"
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        ledger = load_json(ledger_path)
        trace, maximum_rss = replay_candidate_ledger(
            candidate_id,
            ledger,
            route_map,
            reset_next_identities,
            tracker_config,
            config,
        )
        reset_count = sum(frame["state_reset_before_frame"] for frame in trace) - 1
        payload = {
            "schema": TRACE_SCHEMA,
            "stage": "R2-L1E",
            "authority": "candidate_trace_only_no_selection_android_h2_human_or_production_authority",
            "candidate_id": candidate_id,
            "source_id": descriptor["source_id"],
            "sequence_id": descriptor["sequence_id"],
            "frame_mask_sha256": descriptor["frame_mask_sha256"],
            "frame_count": len(trace),
            "discontinuity_reset_count": reset_count,
            "frames": trace,
        }
        atomic_write_json(trace_path, payload)
        receipt = {
            "schema": TRACE_RECEIPT_SCHEMA,
            "stage": "R2-L1E",
            "status": "FIRST_VALID_COMPLETE_TRACE",
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "source_id": descriptor["source_id"],
            "sequence_id": descriptor["sequence_id"],
            "frame_mask_sha256": descriptor["frame_mask_sha256"],
            "frame_count": len(trace),
            "first_frame_identity": list(identity(trace[0])),
            "last_frame_identity": list(identity(trace[-1])),
            "discontinuity_reset_count": reset_count,
            "trace_sha256": sha256_file(trace_path),
            "compact_ledger_sha256": sha256_file(ledger_path),
            "successor_receipt_sha256": sha256_file(successor_path),
            "candidate_implementation_sha256": bindings[
                "candidate_implementation_sha256"
            ],
            "association_implementation_sha256": bindings[
                "association_implementation_sha256"
            ],
            "config_sha256": bindings["config_sha256"],
            "wall_time_seconds": time.monotonic() - started,
            "maximum_host_rss_bytes": maximum_rss,
            "started_at_utc": started_utc,
        }
        atomic_write_json(receipt_path, receipt)
        atomic_write_json(
            attempt_path,
            {
                "attempt_id": attempt_id,
                "status": "COMPLETE",
                "evaluation_authority": True,
                "trace_receipt_sha256": sha256_file(receipt_path),
            },
        )
        if not validate_trace_receipt(
            trace_path,
            receipt_path,
            candidate_id,
            descriptor,
            rows,
            ledger_path,
            successor_path,
            bindings,
        ):
            raise ExecutionAborted("candidate_trace_postwrite_validation_failed")
        return trace_path, receipt_path
    except Exception as error:
        if not attempt_path.exists():
            atomic_write_json(
                attempt_path,
                {
                    "attempt_id": attempt_id,
                    "status": "INCOMPLETE",
                    "evaluation_authority": False,
                    "error_type": error.__class__.__name__,
                    "error": str(error),
                    "wall_time_seconds": time.monotonic() - started,
                    "last_safe_checkpoint": "masked_sequence_ledger_start",
                },
            )
        raise


def load_truth_frame_index(
    config: dict[str, Any], repo: Path
) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    result: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for binding in config["input_contract"]["truth_join_after_output_only"]:
        payload = load_json(repo / binding["path"])
        if (
            "sources" in payload
            and payload.get("schema") == "blindassist_ustrf_route_role_truth_r1"
        ):
            for source in payload["sources"]:
                for episode in source["person_episodes"]:
                    event_id = episode.get("legacy_event_id") or episode.get("risk_event_id")
                    for frame in episode["frames"]:
                        key = (
                            source["source_id"],
                            source["source_id"],
                            int(frame["frame_id"]),
                            int(frame["source_capture_timestamp_ns"]),
                        )
                        result[key].append(
                            {
                                "person_id": episode["person_id"],
                                "event_id": event_id,
                                "bbox_xyxy": frame["bbox_xyxy"],
                                "role": frame.get("role"),
                            }
                        )
        elif "frames" in payload:
            for frame in payload["frames"]:
                key = identity(frame)
                for person in frame.get("persons", []):
                    result[key].append(
                        {
                            "person_id": person["person_id"],
                            "event_id": person.get("event_id"),
                            "bbox_xyxy": person["bbox_xyxy"],
                            "role": person.get("role"),
                        }
                    )
    return result


def attributed_event_ids(
    trace_frame: dict[str, Any],
    truth_people: list[dict[str, Any]],
    minimum_iou: float,
) -> list[str]:
    track_ids = set(trace_frame["delivery_track_ids"])
    tracks = [
        track for track in trace_frame["observed_tracks"] if track["track_id"] in track_ids
    ]
    matched: set[str] = set()
    for track in tracks:
        candidates = sorted(
            (
                (box_iou(track["box"], person["bbox_xyxy"]), person)
                for person in truth_people
                if person.get("event_id")
            ),
            key=lambda row: row[0],
            reverse=True,
        )
        if candidates and candidates[0][0] >= minimum_iou:
            matched.add(str(candidates[0][1]["event_id"]))
    return sorted(matched)


def contribution_counts(
    rows: Iterable[dict[str, Any]], key: str
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row[key])] += 1
    return dict(sorted(counts.items()))


def validate_profile_contract(profile: dict[str, Any], config: dict[str, Any]) -> None:
    metrics = profile["metrics"]
    if metrics["critical_miss"]["denominator"] != 8:
        raise ExecutionAborted("critical_miss_denominator_drift")
    if metrics["clearance"]["denominator"] != 12:
        raise ExecutionAborted("clearance_denominator_drift")
    if metrics["clearance"]["pre_clear_units_excluded"] != 6357:
        raise ExecutionAborted("pre_clear_entered_clearance")
    if metrics["unknown_or_stale_alert"]["denominator"] != 62229:
        raise ExecutionAborted("unknown_stale_denominator_drift")
    repeat = metrics["repeat"]
    if repeat["denominator_source"] != "first_delivery_then_complete_observation":
        raise ExecutionAborted("repeat_truth_pool_substituted_for_actual_denominator")
    minimum = config["metric_permissions"]["repeat"]["minimum_actual_denominator"]
    if repeat["denominator"] < minimum and repeat["eligibility_status"] not in {
        "not_evaluable",
        "evaluable_underpowered",
    }:
        raise ExecutionAborted("repeat_underpowered_status_invalid")
    evidence = metrics["evidence_age"]
    if evidence["timestamp_frame_count"] != 62229 and evidence["eligibility_status"] != "not_evaluable":
        raise ExecutionAborted("evidence_age_missing_frame_not_fail_closed")
    for name in ("event_recall", "regeneration", "false_alerts_per_minute"):
        metric = metrics[name]
        if metric["level"] != "L0" or metric["eligibility_status"] != "diagnostic_only":
            raise ExecutionAborted(f"L0_metric_authority_opened:{name}")
        if any(key in metric for key in ("passed", "gate", "decision")):
            raise ExecutionAborted(f"L0_metric_gate_field_present:{name}")
    for metric in metrics.values():
        if metric.get("denominator") == 0 and metric.get("eligibility_status") in {
            "evaluable",
            "eligible",
        }:
            raise ExecutionAborted("zero_denominator_marked_evaluable")


def build_candidate_profile(
    candidate_id: str,
    trace_paths_by_key: dict[tuple[str, str], Path],
    mask: dict[str, Any],
    truth_index: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    config: dict[str, Any],
    mask_sha256: str,
) -> dict[str, Any]:
    traces: dict[tuple[str, str], list[dict[str, Any]]] = {}
    deliveries = []
    closures = []
    all_frames = []
    minimum_iou = 0.30
    for key, path in trace_paths_by_key.items():
        frames = load_json(path)["frames"]
        traces[key] = frames
        for frame in frames:
            event_ids = (
                attributed_event_ids(
                    frame, truth_index.get(identity(frame), []), minimum_iou
                )
                if frame["deliveries"]
                else []
            )
            enriched = {**frame, "attributed_event_ids": event_ids}
            all_frames.append(enriched)
            for delivery_key in frame["deliveries"]:
                deliveries.append(
                    {
                        "source_id": frame["source_id"],
                        "sequence_id": frame["sequence_id"],
                        "frame_id": frame["frame_id"],
                        "source_capture_timestamp_ns": frame[
                            "source_capture_timestamp_ns"
                        ],
                        "delivery_key": delivery_key,
                        "event_ids": event_ids,
                    }
                )
            for closure_key in frame["closures"]:
                closures.append(
                    {
                        "source_id": frame["source_id"],
                        "sequence_id": frame["sequence_id"],
                        "frame_id": frame["frame_id"],
                        "closure_key": closure_key,
                    }
                )
    events = mask["events"]
    event_by_id = {event["event_id"]: event for event in events}
    event_deliveries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for delivery in deliveries:
        for event_id in delivery["event_ids"]:
            if event_id in event_by_id:
                event_deliveries[event_id].append(delivery)

    critical_events = [
        event
        for event in events
        if event["metrics"]["critical_miss"]["classification"] == "eligible"
    ]
    critical_misses = [
        event
        for event in critical_events
        if not [
            delivery
            for delivery in event_deliveries[event["event_id"]]
            if int(event["anchors"]["alertable_start_frame"])
            <= delivery["frame_id"]
            <= int(event["anchors"]["end_frame"])
        ]
    ]
    clearance_events = [
        event
        for event in events
        if event["metrics"]["clearance"]["classification"] == "eligible"
    ]
    clearance_successes = []
    for event in clearance_events:
        eligible_deliveries = [
            delivery
            for delivery in event_deliveries[event["event_id"]]
            if int(event["anchors"]["alertable_start_frame"])
            <= delivery["frame_id"]
            <= int(event["anchors"]["end_frame"])
        ]
        if not eligible_deliveries:
            continue
        first = eligible_deliveries[0]
        matching_closures = [
            closure
            for closure in closures
            if closure["source_id"] == event["source_id"]
            and closure["sequence_id"] == event["sequence_id"]
            and closure["closure_key"] == first["delivery_key"]
            and int(event["anchors"]["truth_terminal_clear_frame"])
            <= closure["frame_id"]
            <= int(event["anchors"]["end_frame"])
        ]
        if matching_closures:
            clearance_successes.append(event)

    repeat_events = [
        event
        for event in events
        if event["metrics"]["repeat"]["classification"] == "eligible"
    ]
    observed_repeat_events = []
    repeat_delivery_count = 0
    for event in repeat_events:
        observed = [
            delivery
            for delivery in event_deliveries[event["event_id"]]
            if int(event["anchors"]["alertable_start_frame"])
            <= delivery["frame_id"]
            <= int(event["anchors"]["end_frame"])
        ]
        if observed:
            observed_repeat_events.append(event)
            repeat_delivery_count += max(0, len(observed) - 1)

    unknown_or_stale_frames = [
        frame
        for frame, mask_row in zip(
            all_frames, mask["preoutput_frame_ledger"], strict=True
        )
        if mask_row["route_validity_state"] != "known"
    ]
    unknown_or_stale_alerts = sum(
        bool(frame["deliveries"]) or frame["candidate_active"]
        for frame in unknown_or_stale_frames
    )
    consume_ages = [
        frame["candidate_consume_timestamp_ns"]
        - frame["source_capture_timestamp_ns"]
        for frame in all_frames
        if isinstance(frame["candidate_consume_timestamp_ns"], int)
    ]
    repeat_denominator = len(observed_repeat_events)
    repeat_status = (
        "not_evaluable"
        if repeat_denominator == 0
        else "evaluable_underpowered"
        if repeat_denominator
        < config["metric_permissions"]["repeat"]["minimum_actual_denominator"]
        else "evaluable"
    )
    negative_intervals: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for interval in mask["negative_exposure_intervals"]:
        negative_intervals[
            (interval["source_id"], interval["sequence_id"])
        ].append((int(interval["start_ns"]), int(interval["end_ns"])))
    false_alert_deliveries = [
        delivery
        for delivery in deliveries
        if not delivery["event_ids"]
        and any(
            start_ns
            <= int(delivery["source_capture_timestamp_ns"])
            < end_ns
            for start_ns, end_ns in negative_intervals[
                (delivery["source_id"], delivery["sequence_id"])
            ]
        )
    ]
    regeneration_count = sum(
        delivery["frame_id"]
        > int(event_by_id[event_id]["anchors"]["truth_terminal_clear_frame"])
        for delivery in deliveries
        for event_id in delivery["event_ids"]
        if event_id in event_by_id
        and event_by_id[event_id]["anchors"]["truth_terminal_clear_frame"] is not None
    )
    metric_mask_ids = {
        name: [
            event["unit_id"]
            for event in events
            if event["metrics"][name]["classification"] == "eligible"
        ]
        for name in ("critical_miss", "clearance", "repeat")
    }
    profile = {
        "schema": PROFILE_SCHEMA,
        "stage": "R2-L1E",
        "authority": "per_metric_l1_exploratory_only_no_candidate_comparison",
        "candidate_id": candidate_id,
        "metrics": {
            "critical_miss": {
                "level": "L1",
                "eligibility_status": "evaluable",
                "numerator": len(critical_misses),
                "denominator": len(critical_events),
                "source_contributions": contribution_counts(
                    critical_events, "source_id"
                ),
                "provenance_family_contributions": contribution_counts(
                    critical_events, "provenance_family"
                ),
                "mask_row_ids": metric_mask_ids["critical_miss"],
                "mask_sha256": mask_sha256,
                "claim_boundary": "exploratory_metric_only",
            },
            "clearance": {
                "level": "L1",
                "eligibility_status": "evaluable",
                "numerator": len(clearance_successes),
                "denominator": len(clearance_events),
                "pre_clear_units_excluded": 6357,
                "source_contributions": contribution_counts(
                    clearance_events, "source_id"
                ),
                "provenance_family_contributions": contribution_counts(
                    clearance_events, "provenance_family"
                ),
                "mask_row_ids": metric_mask_ids["clearance"],
                "mask_sha256": mask_sha256,
                "claim_boundary": "exploratory_metric_only",
            },
            "unknown_or_stale_alert": {
                "level": "L1",
                "eligibility_status": "evaluable",
                "numerator": unknown_or_stale_alerts,
                "denominator": len(all_frames),
                "unknown_or_stale_frame_count": len(unknown_or_stale_frames),
                "mask_row_ids": [
                    row["unit_id"] for row in mask["preoutput_frame_ledger"]
                ],
                "mask_sha256": mask_sha256,
                "claim_boundary": "exploratory_metric_only",
            },
            "repeat": {
                "level": "CONDITIONAL_L1",
                "eligibility_status": repeat_status,
                "numerator": repeat_delivery_count,
                "denominator": repeat_denominator,
                "denominator_source": "first_delivery_then_complete_observation",
                "minimum_denominator": config["metric_permissions"]["repeat"][
                    "minimum_actual_denominator"
                ],
                "truth_pool_size": len(repeat_events),
                "mask_row_ids": metric_mask_ids["repeat"],
                "mask_sha256": mask_sha256,
                "claim_boundary": "conditional_exploratory_metric_only",
            },
            "evidence_age": {
                "level": "CONDITIONAL_L1",
                "eligibility_status": (
                    "evaluable" if len(consume_ages) == 62229 else "not_evaluable"
                ),
                "timestamp_frame_count": len(consume_ages),
                "required_timestamp_frame_count": 62229,
                "p95_ns": (
                    int(np.percentile(consume_ages, 95, method="higher"))
                    if len(consume_ages) == 62229
                    else None
                ),
                "mask_row_ids": [
                    row["unit_id"] for row in mask["preoutput_frame_ledger"]
                ],
                "mask_sha256": mask_sha256,
                "claim_boundary": "conditional_exploratory_metric_only",
            },
            "event_recall": {
                "level": "L0",
                "eligibility_status": "diagnostic_only",
                "raw_attributed_event_count": len(event_deliveries),
                "denominator": 0,
                "mask_sha256": mask_sha256,
                "claim_boundary": "diagnostic_only_no_pass_fail_or_comparison",
            },
            "regeneration": {
                "level": "L0",
                "eligibility_status": "diagnostic_only",
                "raw_count": regeneration_count,
                "denominator": 0,
                "mask_sha256": mask_sha256,
                "claim_boundary": "diagnostic_only_no_pass_fail_or_comparison",
            },
            "false_alerts_per_minute": {
                "level": "L0",
                "eligibility_status": "diagnostic_only",
                "raw_count": len(false_alert_deliveries),
                "negative_exposure_ns": 297376110945,
                "minimum_l1_exposure_ns": 300000000000,
                "mask_row_ids": [
                    row["unit_id"] for row in mask["negative_exposure_intervals"]
                ],
                "mask_sha256": mask_sha256,
                "claim_boundary": "diagnostic_only_no_pass_fail_or_comparison",
            },
        },
    }
    validate_profile_contract(profile, config)
    return profile


def run_all_candidate_profiles(
    config: dict[str, Any],
    repo: Path,
    groups: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    resets: list[dict[str, Any]],
    route_map: dict[tuple[str, str, int, int], dict[str, Any]],
    mask: dict[str, Any],
    bindings: dict[str, str],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tracker_config = load_json(
        repo / config["input_contract"]["association"]["config_path"]
    )
    reset_next_identities = {
        (
            row["source_id"],
            row["sequence_id"],
            int(row["next_frame_id"]),
            int(row["next_timestamp_ns"]),
        )
        for row in resets
    }
    truth_index = load_truth_frame_index(config, repo)
    profiles = []
    trace_receipts = []
    for candidate_id in config["candidate_roster"]:
        trace_paths_by_key = {}
        for descriptor, rows in groups:
            trace_path, receipt_path = materialize_candidate_trace(
                candidate_id,
                descriptor,
                rows,
                route_map,
                reset_next_identities,
                tracker_config,
                config,
                bindings,
                repo,
                output_root,
            )
            trace_paths_by_key[
                (descriptor["source_id"], descriptor["sequence_id"])
            ] = trace_path
            trace_receipts.append(
                {
                    "candidate_id": candidate_id,
                    "source_id": descriptor["source_id"],
                    "sequence_id": descriptor["sequence_id"],
                    "trace_path": str(trace_path.relative_to(repo)).replace(
                        "\\", "/"
                    ),
                    "trace_sha256": sha256_file(trace_path),
                    "receipt_path": str(receipt_path.relative_to(repo)).replace(
                        "\\", "/"
                    ),
                    "receipt_sha256": sha256_file(receipt_path),
                }
            )
        profile = build_candidate_profile(
            candidate_id,
            trace_paths_by_key,
            mask,
            truth_index,
            config,
            bindings["eligibility_mask_sha256"],
        )
        profile_path = output_root / "profiles" / f"{candidate_id}.json"
        atomic_write_json(profile_path, profile)
        profiles.append(
            {
                **profile,
                "profile_path": str(profile_path.relative_to(repo)).replace(
                    "\\", "/"
                ),
                "profile_sha256": sha256_file(profile_path),
            }
        )
    return profiles, trace_receipts
