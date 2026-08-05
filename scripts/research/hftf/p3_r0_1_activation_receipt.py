#!/usr/bin/env python3
"""Generate and verify the static P3 R0.1 activation receipt.

This module intentionally uses only the Python standard library. It hashes
checkpoint/cache/bundle bytes but never imports an ML runtime or parses the
sealed target bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_dav2_temporal_392_student_p3_r0_1_activation_receipt"
PROTOCOL_SCHEMA = (
    "blindassist_dav2_temporal_392_student_p3_r0_1_activation_protocol"
)
BINDINGS_SCHEMA = (
    "blindassist_dav2_temporal_392_student_p3_r0_1_activation_bindings"
)
READY = "P3_R0_1_ACTIVATION_READY_HOLDOUT_UNOPENED_MODEL_UNLOADED"
INVALID = "P3_R0_1_ACTIVATION_BINDING_INVALID_NO_MODEL_LOAD"
STATES = ("CLEAR", "OCCUPIED", "UNKNOWN_GROUND")
TRANSITIONS = tuple(f"{left}_TO_{right}" for left in STATES for right in STATES)
SHA_CHARS = frozenset("0123456789ABCDEF")

TRAIN_FRAME_FIELDS = frozenset(
    {
        "frame_id",
        "video_id",
        "parent_id",
        "timestamp_ns",
        "rgb_identity",
        "rgb_sha256",
        "teacher_depth_ref",
        "teacher_depth_sha256",
        "teacher_timestamp_ns",
        "teacher_valid",
        "tof_valid",
        "frozen_a2_mean_abs_log_depth_disagreement",
        "clearance_m",
        "geometry_state",
        "geometry_target_valid",
    }
)
HOLDOUT_FRAME_FIELDS = frozenset(
    {
        "frame_id",
        "video_id",
        "parent_id",
        "timestamp_ns",
        "sealed_target_id",
        "rgb_identity",
        "rgb_sha256",
    }
)


@dataclass(frozen=True)
class RoleSummary:
    role: str
    parents: frozenset[str]
    clip_count: int
    frame_count: int
    transition_counts: dict[str, int] | None
    outcomes_opened: bool | None


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def valid_sha(value: Any) -> bool:
    normalized = str(value).upper()
    return len(normalized) == 64 and set(normalized) <= SHA_CHARS


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def _inside_repo(repo_root: Path, relative: str) -> Path:
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"binding path leaves repository: {relative}") from error
    return path


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    observed: Any = None,
    expected: Any = None,
    detail: str = "",
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "detail": detail,
        }
    )


def _snapshot_files(
    repo_root: Path,
    bindings: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for name, binding in bindings["files"].items():
        relative = str(binding["path"])
        path = _inside_repo(repo_root, relative)
        expected = binding.get("expected_sha256")
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        size = path.stat().st_size if exists else None
        expected_bound = valid_sha(expected)
        passed = exists and expected_bound and actual == str(expected).upper()
        snapshots[name] = {
            "path": relative,
            "exists": exists,
            "size_bytes": size,
            "expected_sha256": str(expected).upper() if expected_bound else None,
            "actual_sha256": actual,
        }
        _check(
            checks,
            f"file:{name}",
            passed,
            observed=actual,
            expected=str(expected).upper() if expected_bound else "BOUND_SHA256_REQUIRED",
            detail="file missing" if not exists else ("expected SHA missing" if not expected_bound else ""),
        )
    return snapshots


def _safe_json(
    repo_root: Path,
    snapshots: dict[str, dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    snapshot = snapshots.get(name)
    if not snapshot or not snapshot["exists"]:
        return None
    try:
        return load_json(_inside_repo(repo_root, snapshot["path"]))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None


def _role_manifest(value: dict[str, Any], expected_role: str) -> RoleSummary:
    top_fields = {"schema", "protocol_sha256", "role", "clips"}
    if expected_role == "public_holdout":
        top_fields.add("outcomes_opened")
    if set(value) != top_fields:
        raise ValueError("role manifest top-level field drift")
    if value["schema"] != "blindassist_dav2_temporal_392_student_p3_r0_1_role_manifest":
        raise ValueError("role manifest schema drift")
    if value["role"] != expected_role:
        raise ValueError("role manifest role drift")
    if not valid_sha(value["protocol_sha256"]):
        raise ValueError("role manifest protocol SHA invalid")
    clips = value["clips"]
    if not isinstance(clips, list) or not clips:
        raise ValueError("role manifest has no clips")
    parents: set[str] = set()
    seen_frames: set[str] = set()
    transition_counts = {name: 0 for name in TRANSITIONS}
    frame_count = 0
    for clip in clips:
        if set(clip) != {"clip_id", "video_id", "parent_id", "frames"}:
            raise ValueError("role clip field drift")
        parent_id = str(clip["parent_id"])
        video_id = str(clip["video_id"])
        if not parent_id or not video_id:
            raise ValueError("empty parent/video id")
        parents.add(parent_id)
        frames = clip["frames"]
        if not isinstance(frames, list) or len(frames) != 4:
            raise ValueError("role clip must contain exactly four frames")
        timestamps: list[int] = []
        states: list[list[str]] = []
        validities: list[list[bool]] = []
        for frame in frames:
            expected_fields = (
                HOLDOUT_FRAME_FIELDS
                if expected_role == "public_holdout"
                else TRAIN_FRAME_FIELDS
            )
            if set(frame) != set(expected_fields):
                raise ValueError("role frame exact allowlist failed")
            frame_id = str(frame["frame_id"])
            if not frame_id or frame_id in seen_frames:
                raise ValueError("role frame reused")
            seen_frames.add(frame_id)
            if frame["parent_id"] != parent_id or frame["video_id"] != video_id:
                raise ValueError("frame parent/video drift")
            timestamp = frame["timestamp_ns"]
            if not isinstance(timestamp, int) or timestamp <= 0:
                raise ValueError("invalid real timestamp")
            timestamps.append(timestamp)
            if not valid_sha(frame["rgb_sha256"]):
                raise ValueError("invalid RGB SHA")
            if expected_role == "public_holdout":
                if not str(frame["sealed_target_id"]):
                    raise ValueError("sealed target id missing")
                continue
            if not valid_sha(frame["teacher_depth_sha256"]):
                raise ValueError("invalid teacher depth SHA")
            state = frame["geometry_state"]
            valid = frame["geometry_target_valid"]
            if not (
                isinstance(state, list)
                and len(state) == 3
                and all(item in STATES for item in state)
                and isinstance(valid, list)
                and len(valid) == 3
                and all(isinstance(item, bool) for item in valid)
            ):
                raise ValueError("invalid geometry targets")
            states.append(state)
            validities.append(valid)
        gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        if not all(0 < gap <= 500_000_000 for gap in gaps):
            raise ValueError("invalid clip cadence")
        frame_count += 4
        if expected_role != "public_holdout":
            for index, (previous, current) in enumerate(zip(states, states[1:])):
                for band, (left, right) in enumerate(zip(previous, current)):
                    if validities[index][band] and validities[index + 1][band]:
                        transition_counts[f"{left}_TO_{right}"] += 1
    return RoleSummary(
        role=expected_role,
        parents=frozenset(parents),
        clip_count=len(clips),
        frame_count=frame_count,
        transition_counts=(
            None if expected_role == "public_holdout" else transition_counts
        ),
        outcomes_opened=(
            bool(value["outcomes_opened"])
            if expected_role == "public_holdout"
            else None
        ),
    )


def _effective_number_weights(counts: dict[str, int], beta: float) -> list[float]:
    if set(counts) != set(TRANSITIONS) or any(int(counts[name]) <= 0 for name in TRANSITIONS):
        raise ValueError("all nine train transitions need positive support")
    raw = [(1.0 - beta) / (1.0 - beta ** int(counts[name])) for name in TRANSITIONS]
    mean = sum(raw) / len(raw)
    return [value / mean for value in raw]


def _semantic_checks(
    repo_root: Path,
    protocol: dict[str, Any],
    bindings: dict[str, Any],
    snapshots: dict[str, dict[str, Any]],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "parents_by_role": None,
        "transition_counts": None,
        "class_weight_derivation_receipt": None,
        "class_weight_derivation_receipt_sha256": None,
    }
    r0_1 = _safe_json(repo_root, snapshots, "r0_1_protocol")
    expected_r0_1_schema = "blindassist_dav2_temporal_392_student_p3_r0_1_protocol"
    _check(
        checks,
        "r0_1_protocol_schema_and_status",
        bool(
            r0_1
            and r0_1.get("schema") == expected_r0_1_schema
            and r0_1.get("current_terminal")
            == "P3_R0_1_PRE_ACTIVATION_CORRECTION_COMPLETE_BINDINGS_PENDING_HOLDOUT_UNOPENED"
        ),
    )
    if r0_1:
        for key, implementation_key in (
            ("training_module", "evidence_loss_manifest_module_sha256"),
            ("clip_p1_evaluator", "clip_p1_evaluator_sha256"),
            ("module_tests", "module_tests_sha256"),
            ("evaluator_tests", "evaluator_tests_sha256"),
        ):
            observed = snapshots.get(key, {}).get("actual_sha256")
            expected = r0_1.get("implementation", {}).get(implementation_key)
            _check(
                checks,
                f"r0_1_implementation_binding:{key}",
                observed == expected,
                observed=observed,
                expected=expected,
            )
    for key in ("activation_generator", "activation_tests"):
        protocol_key = "generator" if key == "activation_generator" else "tests"
        expected = protocol.get("implementation", {}).get(protocol_key, {}).get("sha256")
        observed = snapshots.get(key, {}).get("actual_sha256")
        _check(
            checks,
            f"activation_implementation_binding:{key}",
            observed == expected,
            observed=observed,
            expected=expected,
        )

    a2_protocol = _safe_json(repo_root, snapshots, "a2_protocol")
    a2_result = _safe_json(repo_root, snapshots, "a2_result")
    a2_training = _safe_json(repo_root, snapshots, "a2_training_receipt")
    checkpoint_sha = snapshots.get("a2_checkpoint", {}).get("actual_sha256")
    a2_valid = bool(
        a2_protocol
        and a2_result
        and a2_training
        and a2_protocol.get("schema")
        == "blindassist_dav2_392_distillation_a2_r0_protocol"
        and a2_result.get("schema")
        == "blindassist_dav2_model_variant_gate_r0_result"
        and a2_training.get("schema")
        == "blindassist_dav2_392_distillation_a2_r0_training_result"
        and a2_training.get("terminal") == "A2_DISTILLATION_TRAINING_COMPLETE_P1_UNOPENED"
        and a2_training.get("truth_inputs_opened") is False
        and int(a2_training.get("epochs_completed", -1)) == 3
        and a2_training.get("protocol_sha256")
        == snapshots.get("a2_protocol", {}).get("actual_sha256")
        and a2_training.get("checkpoint", {}).get("sha256") == checkpoint_sha
        and bindings.get("a2", {}).get("selected_checkpoint_sha256") == checkpoint_sha
        and bindings.get("a2", {}).get("training_receipt_sha256")
        == snapshots.get("a2_training_receipt", {}).get("actual_sha256")
        and bindings.get("a2", {}).get("protocol_sha256")
        == snapshots.get("a2_protocol", {}).get("actual_sha256")
        and snapshots.get("a2_result", {}).get("actual_sha256")
        == bindings.get("a2", {}).get("result_sha256")
    )
    _check(checks, "a2_selected_checkpoint_and_receipts", a2_valid)

    exclusion = _safe_json(repo_root, snapshots, "legacy_p1_exclusion_ledger")
    p1_roster_sha = snapshots.get("legacy_p1_roster", {}).get("actual_sha256")
    exclusion_parents: set[str] = set()
    exclusion_valid = False
    if exclusion:
        exclusion_parents = {str(item) for item in exclusion.get("parent_ids", [])}
        exclusion_valid = bool(
            exclusion.get("schema")
            == "blindassist_p3_r0_1_legacy_p1_ancestry_exclusion_ledger"
            and exclusion.get("status") == "FROZEN_CONSUMED_EXCLUSION_ONLY"
            and exclusion.get("source_p1_roster_sha256")
            == bindings["legacy_p1"]["roster_sha256"]
            and p1_roster_sha == bindings["legacy_p1"]["roster_sha256"]
            and exclusion_parents
        )
    _check(checks, "legacy_p1_ancestry_exclusion_ledger", exclusion_valid)

    role_values = {
        role: _safe_json(repo_root, snapshots, file_key)
        for role, file_key in (
            ("train", "train_manifest"),
            ("validation", "validation_manifest"),
            ("public_holdout", "public_holdout_manifest"),
        )
    }
    role_summaries: dict[str, RoleSummary] = {}
    for role, value in role_values.items():
        try:
            if value is None:
                raise ValueError("manifest missing")
            role_summaries[role] = _role_manifest(value, role)
            passed = value["protocol_sha256"] == snapshots["r0_1_protocol"]["actual_sha256"]
        except (KeyError, TypeError, ValueError):
            passed = False
        _check(checks, f"role_manifest:{role}", passed)

    overlap_valid = False
    weights_valid = False
    if len(role_summaries) == 3 and exclusion_valid:
        train = role_summaries["train"]
        validation = role_summaries["validation"]
        holdout = role_summaries["public_holdout"]
        overlap_valid = bool(
            train.parents.isdisjoint(validation.parents)
            and train.parents.isdisjoint(holdout.parents)
            and validation.parents.isdisjoint(holdout.parents)
            and train.parents.isdisjoint(exclusion_parents)
            and validation.parents.isdisjoint(exclusion_parents)
            and holdout.parents.isdisjoint(exclusion_parents)
        )
        summary["parents_by_role"] = {
            role: sorted(item.parents) for role, item in role_summaries.items()
        }
        summary["transition_counts"] = {
            role: item.transition_counts
            for role, item in role_summaries.items()
            if item.transition_counts is not None
        }
        try:
            validation_counts = validation.transition_counts or {}
            if any(int(validation_counts[name]) <= 0 for name in TRANSITIONS):
                raise ValueError("validation transition support incomplete")
            beta = float(protocol["class_weight_derivation"]["beta"])
            train_counts = train.transition_counts or {}
            weights = _effective_number_weights(train_counts, beta)
            derivation = {
                "schema": "blindassist_p3_r0_1_transition_class_weight_derivation_receipt",
                "protocol_sha256": snapshots["r0_1_protocol"]["actual_sha256"],
                "source_role": "train",
                "transition_order": list(TRANSITIONS),
                "transition_counts": train_counts,
                "formula": "(1-beta)/(1-beta^count), normalized to mean 1",
                "beta": beta,
                "weights": weights,
                "validation_transition_counts": validation_counts,
                "holdout_used": False,
            }
            summary["class_weight_derivation_receipt"] = derivation
            summary["class_weight_derivation_receipt_sha256"] = sha256_bytes(
                canonical_bytes(derivation)
            )
            weights_valid = True
        except (KeyError, TypeError, ValueError, OverflowError):
            weights_valid = False
    _check(checks, "zero_parent_overlap_all_roles_and_legacy_p1", overlap_valid)
    _check(checks, "nine_class_counts_and_frozen_weight_derivation", weights_valid)

    holdout_unopened = bool(
        role_summaries.get("public_holdout")
        and role_summaries["public_holdout"].outcomes_opened is False
    )
    _check(checks, "holdout_outcomes_unopened", holdout_unopened, observed=holdout_unopened, expected=True)

    disagreement = _safe_json(repo_root, snapshots, "disagreement_manifest")
    disagreement_valid = bool(
        disagreement
        and disagreement.get("schema")
        == "blindassist_p3_r0_1_frozen_a2_disagreement_manifest"
        and disagreement.get("status") == "FROZEN_PARENT_A2_ONLY"
        and disagreement.get("current_student_used") is False
        and disagreement.get("a2_checkpoint_sha256") == checkpoint_sha
        and disagreement.get("cache_sha256")
        == snapshots.get("disagreement_cache", {}).get("actual_sha256")
        and disagreement.get("producer_sha256")
        == snapshots.get("disagreement_producer", {}).get("actual_sha256")
    )
    _check(checks, "frozen_a2_disagreement_cache_and_producer", disagreement_valid)

    coverage = _safe_json(repo_root, snapshots, "sealed_coverage_receipt")
    coverage_valid = bool(
        coverage
        and coverage.get("schema")
        == "blindassist_dav2_temporal_392_student_p3_r0_1_sealed_coverage_receipt"
        and coverage.get("status") == "SEALED_COVERAGE_VERIFIED"
        and coverage.get("label_rows_disclosed") is False
        and coverage.get("created_before_training_activation") is True
        and int(coverage.get("evaluable_clip_count", -1)) >= 32
        and int(coverage.get("video_parent_count", -1)) >= 8
        and coverage.get("sealed_bundle_sha256")
        == snapshots.get("sealed_target_bundle", {}).get("actual_sha256")
        and coverage.get("coverage_producer_sha256")
        == snapshots.get("coverage_producer", {}).get("actual_sha256")
        and role_summaries.get("public_holdout")
        and coverage.get("identity_manifest_sha256")
        == snapshots.get("public_holdout_manifest", {}).get("actual_sha256")
    )
    if coverage_valid:
        key_counts = coverage.get("key_transition_counts", {})
        geometry_counts = coverage.get("geometry_transition_counts", {})
        coverage_valid = bool(
            set(key_counts)
            == {
                "CLEAR_TO_OCCUPIED",
                "OCCUPIED_TO_CLEAR",
                "KNOWN_TO_UNKNOWN_GROUND",
                "UNKNOWN_GROUND_TO_KNOWN",
            }
            and all(int(value) >= 8 for value in key_counts.values())
            and set(geometry_counts) == set(TRANSITIONS)
            and all(int(value) >= 0 for value in geometry_counts.values())
        )
    _check(checks, "sealed_bundle_coverage_receipt_and_producer", coverage_valid)
    return summary


def build_receipt(
    repo_root: Path,
    activation_protocol: dict[str, Any],
    bindings: dict[str, Any],
    *,
    current_git_commit: str,
    activation_protocol_sha256: str,
) -> dict[str, Any]:
    if activation_protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("activation protocol schema drift")
    if bindings.get("schema") != BINDINGS_SCHEMA:
        raise ValueError("activation bindings schema drift")
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "current_git_commit",
        len(current_git_commit) == 40
        and all(character in "0123456789abcdef" for character in current_git_commit),
        observed=current_git_commit,
        expected="40 lowercase hexadecimal characters",
    )
    snapshots = _snapshot_files(repo_root, bindings, checks)
    summary = _semantic_checks(
        repo_root, activation_protocol, bindings, snapshots, checks
    )
    flags = bindings["runtime_state_assertions"]
    for name in (
        "model_loaded",
        "optimizer_constructed",
        "training_started",
        "holdout_bundle_parsed",
    ):
        _check(checks, name, flags.get(name) is False, observed=flags.get(name), expected=False)
    candidate_path = _inside_repo(repo_root, bindings["candidate_output_path"])
    candidate_absent = not candidate_path.exists()
    _check(
        checks,
        "candidate_output_directory_absent",
        candidate_absent,
        observed=candidate_path.exists(),
        expected=False,
    )
    passed = all(check["passed"] for check in checks)
    return {
        "schema": SCHEMA,
        "activation_protocol_sha256": activation_protocol_sha256,
        "bindings_sha256": sha256_bytes(canonical_bytes(bindings)),
        "git_commit": current_git_commit,
        "static_only": True,
        "sealed_target_bundle_parsed": False,
        "runtime_state": {
            "model_loaded": False,
            "optimizer_constructed": False,
            "training_started": False,
        },
        "candidate_output_path": bindings["candidate_output_path"],
        "candidate_output_exists": not candidate_absent,
        "binding_snapshots": snapshots,
        "semantic_summary": summary,
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "terminal": READY if passed else INVALID,
    }


def verify_receipt(
    receipt: dict[str, Any],
    repo_root: Path,
    activation_protocol: dict[str, Any],
    bindings: dict[str, Any],
    *,
    current_git_commit: str,
    activation_protocol_sha256: str,
) -> dict[str, Any]:
    expected = build_receipt(
        repo_root,
        activation_protocol,
        bindings,
        current_git_commit=current_git_commit,
        activation_protocol_sha256=activation_protocol_sha256,
    )
    exact = receipt == expected
    return {
        "schema": "blindassist_dav2_temporal_392_student_p3_r0_1_activation_validation",
        "receipt_exact_reproduction": exact,
        "receipt_terminal": receipt.get("terminal"),
        "expected_terminal": expected["terminal"],
        "valid": exact,
        "terminal": (
            "P3_R0_1_ACTIVATION_RECEIPT_VALID"
            if exact
            else "P3_R0_1_ACTIVATION_RECEIPT_INVALID"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--activation-protocol", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    protocol = load_json(args.activation_protocol.resolve())
    protocol_sha256 = sha256_file(args.activation_protocol.resolve())
    bindings = load_json(args.bindings.resolve())
    head = git_head(repo_root)
    receipt = build_receipt(
        repo_root,
        protocol,
        bindings,
        current_git_commit=head,
        activation_protocol_sha256=protocol_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validation = verify_receipt(
        load_json(args.output),
        repo_root,
        protocol,
        bindings,
        current_git_commit=head,
        activation_protocol_sha256=protocol_sha256,
    )
    args.validation_output.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"receipt": receipt["terminal"], "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
