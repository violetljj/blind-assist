"""Close and analyze the frozen 496-arm RCLE R2 formal bundle.

The CLI reads only paths explicitly named by the formal bundle success file
and the frozen formal identity lock.  It never discovers candidate outputs.
Incomplete bundles are never scientifically analyzed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    p3_analysis_r0,
)


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
SCHEMA = "rcle.periodic_self_motion_counterfactual.p4_formal_result.v1"
IDENTITY_SCHEMA = (
    "rcle.periodic_self_motion_counterfactual.p4_formal_identity_lock.v1"
)
BUNDLE_SCHEMA = "rcle.periodic_self_motion_counterfactual.p4_bundle_success.v1"
ARM_SCHEMA = "rcle.periodic_self_motion_counterfactual.p4_arm_receipt.v1"
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MAIN_ARMS = (
    "STATIC_CAMERA__CLEAN",
    "STATIC_CAMERA__BLUR",
    "STATIC_CAMERA__LOW_TEXTURE",
    "PERIODIC_6DOF_SELF_MOTION__CLEAN",
    "PERIODIC_6DOF_SELF_MOTION__BLUR",
    "PERIODIC_6DOF_SELF_MOTION__LOW_TEXTURE",
)
GUARD_ARMS = (
    "MONOTONIC_APPROACH_ONLY__CLEAN",
    "MONOTONIC_APPROACH_PLUS_PERIODIC_6DOF__CLEAN",
)
PAIR_COUNT = 601
FRAME_COUNT = 602
EXPECTED_MAIN = 480
EXPECTED_GUARD = 16
EXPECTED_TOTAL = 496
EXPECTED_FRAMES = 298_592
EXPECTED_PAIRS = 298_096
THRESHOLD = 0.01
REQUIRED_CONSECUTIVE = 3
CLEAN_PAIR_MINIMUM = math.ceil(0.70 * PAIR_COUNT)
SHA256_HEX = frozenset("0123456789abcdef")


class ClosureError(ValueError):
    """A fail-closed formal bundle error with an execution state."""

    def __init__(self, state: str, code: str) -> None:
        super().__init__(f"{state}:{code}")
        self.state = state
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise ClosureError("INVALID", f"{label}_SHA256")
    return value


def _is_frame_hash_row(row: dict[str, Any]) -> bool:
    return all(
        isinstance(row.get(field), str)
        and len(row[field]) == 64
        and all(character in SHA256_HEX for character in row[field])
        for field in ("rgb_sha256", "valid_mask_sha256")
    )


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ClosureError("INVALID", f"{label}_NONFINITE")
    return float(value)


def _exact_seed(role: str, block: str, ordinal: int) -> int:
    namespace = "MAIN" if role == "MAIN_FACTORIAL" else "GUARD"
    token = f"{PROTOCOL_ID}|{namespace}|{block}|{ordinal:02d}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def _expected_identity_keys() -> set[tuple[str, str, int, str]]:
    return {
        ("MAIN_FACTORIAL", block, ordinal, arm)
        for block in BLOCKS
        for ordinal in range(20)
        for arm in MAIN_ARMS
    } | {
        ("POSITIVE_GUARDRAIL", block, ordinal, arm)
        for block in BLOCKS
        for ordinal in range(2)
        for arm in GUARD_ARMS
    }


def validate_identity_lock(
    identity_lock: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if (
        identity_lock.get("schema") != IDENTITY_SCHEMA
        or identity_lock.get("protocol_id") != PROTOCOL_ID
        or identity_lock.get("formal_execution_authorized") is not False
    ):
        raise ClosureError("INVALID", "IDENTITY_LOCK_HEADER")
    identities = identity_lock.get("identities")
    if not isinstance(identities, list):
        raise ClosureError("INVALID", "IDENTITY_LIST")
    if len(identities) < EXPECTED_TOTAL:
        raise ClosureError("EXECUTION_INCOMPLETE", "IDENTITY_COUNT")
    if len(identities) != EXPECTED_TOTAL:
        raise ClosureError("INVALID", "IDENTITY_COUNT")
    expected = _expected_identity_keys()
    seen_keys: set[tuple[str, str, int, str]] = set()
    by_sequence: dict[str, dict[str, Any]] = {}
    for item in identities:
        if not isinstance(item, dict):
            raise ClosureError("INVALID", "IDENTITY_OBJECT")
        role = item.get("role")
        block = item.get("block")
        ordinal = item.get("ordinal")
        arm = item.get("arm")
        if (
            role not in {"MAIN_FACTORIAL", "POSITIVE_GUARDRAIL"}
            or block not in BLOCKS
            or not isinstance(ordinal, int)
            or isinstance(ordinal, bool)
            or not isinstance(arm, str)
        ):
            raise ClosureError("INVALID", "IDENTITY_FIELDS")
        key = (role, block, ordinal, arm)
        if key not in expected or key in seen_keys:
            raise ClosureError("INVALID", "IDENTITY_KEYSET")
        seen_keys.add(key)
        sequence_id = item.get("sequence_id")
        cluster_id = item.get("cluster_id")
        if (
            not isinstance(sequence_id, str)
            or not sequence_id
            or sequence_id in by_sequence
            or not isinstance(cluster_id, str)
            or not cluster_id
        ):
            raise ClosureError("INVALID", "SEQUENCE_OR_CLUSTER_ID")
        if (
            item.get("numeric_seed_uint64")
            != _exact_seed(role, block, ordinal)
            or item.get("frame_count") != FRAME_COUNT
            or item.get("pair_count") != PAIR_COUNT
        ):
            raise ClosureError("INVALID", "IDENTITY_NUMERIC_BINDING")
        for field in ("scene_geometry_sha256", "trajectory_sha256"):
            _sha(item.get(field), f"IDENTITY_{field}")
        by_sequence[sequence_id] = item
    if seen_keys != expected:
        raise ClosureError("EXECUTION_INCOMPLETE", "FROZEN_IDENTITY_GRID")
    counts = identity_lock.get("counts")
    expected_counts = {
        "main_clusters": 80,
        "guardrail_clusters": 8,
        "main_sequences": EXPECTED_MAIN,
        "guardrail_sequences": EXPECTED_GUARD,
        "total_sequences": EXPECTED_TOTAL,
        "frames": EXPECTED_FRAMES,
        "pairs": EXPECTED_PAIRS,
    }
    if not isinstance(counts, dict) or any(
        counts.get(key) != value for key, value in expected_counts.items()
    ):
        raise ClosureError("INVALID", "IDENTITY_COUNTS")
    observed_set_sha = sha256_bytes(canonical_bytes(identities))
    if identity_lock.get("identity_set_sha256") != observed_set_sha:
        raise ClosureError("INVALID", "IDENTITY_SET_HASH")
    return by_sequence


def reduce_pair_ledger(
    rows: Iterable[dict[str, Any]],
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    materialized = list(rows)
    if len(materialized) < PAIR_COUNT:
        raise ClosureError("EXECUTION_INCOMPLETE", "PAIR_COUNT")
    if len(materialized) != PAIR_COUNT:
        raise ClosureError("INVALID", "PAIR_COUNT")
    if identity is None:
        first = materialized[0]
        if not isinstance(first, dict):
            raise ClosureError("INVALID", "PAIR_ROW_OBJECT")
        identity = {
            field: first.get(field)
            for field in (
                "sequence_id",
                "cluster_id",
                "block",
                "ordinal",
                "role",
                "arm",
            )
        }
    streak = 0
    trigger_count = 0
    failure_count = 0
    clean_count = 0
    evaluable_count = 0
    envelope = (
        "sequence_id",
        "cluster_id",
        "block",
        "ordinal",
        "role",
        "arm",
    )
    for index, row in enumerate(materialized):
        if not isinstance(row, dict) or row.get("pair_index") != index:
            raise ClosureError("INVALID", "PAIR_ORDER_OR_IDENTITY")
        if any(row.get(field) != identity.get(field) for field in envelope):
            raise ClosureError("INVALID", "PAIR_ENVELOPE")
        evaluable = row.get("evaluable")
        response = row.get("compensated_expansion_median_per_s")
        if evaluable is True:
            response_value = _finite(response, "EVALUABLE_RESPONSE")
            evaluable_count += 1
            streak = streak + 1 if response_value > THRESHOLD else 0
        elif evaluable is False:
            if response is not None:
                raise ClosureError("INVALID", "ABSTENTION_RESPONSE_PRESENT")
            streak = 0
        else:
            raise ClosureError("INVALID", "EVALUABLE_FLAG")
        trigger = streak >= REQUIRED_CONSECUTIVE
        if (
            "compensated_three_pair_trigger" in row
            and row["compensated_three_pair_trigger"] is not trigger
        ):
            raise ClosureError("INVALID", "FORGED_THREE_PAIR_TRIGGER")
        trigger_count += int(trigger)

        detected = row.get("detected_feature_count")
        consistent = row.get("forward_backward_consistent_count")
        fraction = row.get("forward_backward_consistent_fraction")
        occupied = row.get("occupied_3x3_cells")
        if (
            not isinstance(detected, int)
            or isinstance(detected, bool)
            or detected < 0
            or not isinstance(consistent, int)
            or isinstance(consistent, bool)
            or consistent < 0
            or consistent > detected
            or not isinstance(occupied, int)
            or isinstance(occupied, bool)
            or occupied not in range(10)
        ):
            raise ClosureError("INVALID", "TRACKING_COUNT")
        fraction_value = _finite(fraction, "FB_FRACTION")
        if not 0.0 <= fraction_value <= 1.0:
            raise ClosureError("INVALID", "FB_FRACTION_RANGE")
        fb_raw = row.get("median_forward_backward_error_px")
        if fb_raw is None:
            fb_error = None
        else:
            fb_error = _finite(fb_raw, "FB_ERROR")
            if fb_error < 0.0:
                raise ClosureError("INVALID", "FB_ERROR_RANGE")
        collapse = detected < 60 or consistent < 60 or fraction_value < 0.50
        fb_failure = fb_error is None or fb_error > 0.75
        failure_count += int(collapse or fb_failure)
        clean_count += int(
            not collapse
            and not fb_failure
            and occupied >= 5
        )
    return {
        "scheduled_pair_count": PAIR_COUNT,
        "evaluable_pair_count": evaluable_count,
        "trigger_count": trigger_count,
        "trigger_density": trigger_count / PAIR_COUNT,
        "quality_failure_union_count": failure_count,
        "quality_failure_union_density": failure_count / PAIR_COUNT,
        "clean_trackable_pair_count": clean_count,
        "clean_sequence_trackable": clean_count >= CLEAN_PAIR_MINIMUM,
    }


def reduce_pair_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Runner-facing reducer; identity is inferred then checked on every row."""

    return reduce_pair_ledger(rows)


def _classify_terminal(
    analysis: dict[str, Any],
    clean_gate_pass: bool,
    guard_gate_pass: bool,
) -> tuple[str, str | None]:
    estimands = analysis["estimands"]

    def status(name: str) -> str:
        return str(estimands[name]["classification"])

    if not clean_gate_pass or not guard_gate_pass:
        return "INTERVENTION_NOT_EVALUABLE", None
    motion = status("MOTION_CLEAN")
    blur = status("BLUR_STATIC")
    low = status("LOW_TEXTURE_STATIC")
    x_blur = status("MOTION_X_BLUR")
    x_low = status("MOTION_X_LOW_TEXTURE")
    mixed = (
        motion == "SUPPORTED" and (blur == "SUPPORTED" or low == "SUPPORTED")
    ) or (
        x_blur == "SUPPORTED"
        and status("MOTION_BLUR_VS_STATIC_CLEAN") == "SUPPORTED"
    ) or (
        x_low == "SUPPORTED"
        and status("MOTION_LOW_TEXTURE_VS_STATIC_CLEAN") == "SUPPORTED"
    )
    if mixed:
        return "MIXED", None
    ruled = "RULED_OUT_AS_MATERIAL"
    if (
        motion == "SUPPORTED"
        and blur == ruled
        and low == ruled
        and x_blur == ruled
        and x_low == ruled
    ):
        return "MOTION_SUPPORTED", None
    supported_quality: list[str] = []
    if (
        blur == "SUPPORTED"
        and status("BLUR_FAILURE_UNION_STATIC") == "SUPPORTED"
    ):
        supported_quality.append("BLUR")
    if (
        low == "SUPPORTED"
        and status("LOW_TEXTURE_FAILURE_UNION_STATIC") == "SUPPORTED"
    ):
        supported_quality.append("LOW_TEXTURE")
    if (
        supported_quality
        and motion == ruled
        and x_blur == ruled
        and x_low == ruled
    ):
        subtype = "BOTH" if len(supported_quality) == 2 else supported_quality[0]
        return "QUALITY_SUPPORTED", subtype
    return "NO_SEPARATION_HOLD", None


def _gate_summaries(
    summaries: dict[str, dict[str, Any]],
    identities: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_blocks: dict[str, Any] = {}
    clean_pass = True
    for block in BLOCKS:
        clean_blocks[block] = {}
        for arm in (
            "STATIC_CAMERA__CLEAN",
            "PERIODIC_6DOF_SELF_MOTION__CLEAN",
        ):
            selected = [
                summaries[sequence_id]["clean_sequence_trackable"]
                for sequence_id, identity in identities.items()
                if identity["role"] == "MAIN_FACTORIAL"
                and identity["block"] == block
                and identity["arm"] == arm
            ]
            count = sum(selected)
            passed = len(selected) == 20 and count >= 18
            clean_pass = clean_pass and passed
            clean_blocks[block][arm] = {
                "sequence_count": len(selected),
                "trackable_sequence_count": count,
                "passed": passed,
            }
    guard_arms: dict[str, Any] = {}
    guard_pass = True
    for arm in GUARD_ARMS:
        block_means: dict[str, float] = {}
        for block in BLOCKS:
            values = [
                summaries[sequence_id]["trigger_density"]
                for sequence_id, identity in identities.items()
                if identity["role"] == "POSITIVE_GUARDRAIL"
                and identity["block"] == block
                and identity["arm"] == arm
            ]
            if len(values) != 2:
                raise ClosureError("EXECUTION_INCOMPLETE", "GUARD_BLOCK_COUNT")
            block_means[block] = sum(values) / 2.0
        overall = sum(block_means.values()) / 4.0
        passing_blocks = sum(value >= 0.50 for value in block_means.values())
        passed = overall >= 0.50 and passing_blocks >= 3
        guard_pass = guard_pass and passed
        guard_arms[arm] = {
            "block_means": block_means,
            "overall_mean": overall,
            "blocks_at_least_0_50": passing_blocks,
            "passed": passed,
        }
    return (
        {"passed": clean_pass, "blocks": clean_blocks},
        {"passed": guard_pass, "arms": guard_arms},
    )


def assemble_loaded_bundle(
    bundle: dict[str, Any],
    identity_lock: dict[str, Any],
    receipt_loader: Callable[[str], dict[str, Any]],
    receipt_hash_loader: Callable[[str], str],
    ledger_loader: Callable[[str], list[dict[str, Any]]],
    ledger_hash_loader: Callable[[str], str],
    *,
    bundle_sha256: str,
    identity_lock_sha256: str,
) -> dict[str, Any]:
    identities = validate_identity_lock(identity_lock)
    if (
        bundle.get("schema") != BUNDLE_SCHEMA
        or bundle.get("protocol_id") != PROTOCOL_ID
        or bundle.get("terminal")
        not in {"BUNDLE_COMPLETE", "FORMAL_BUNDLE_COMPLETE"}
    ):
        raise ClosureError("INVALID", "BUNDLE_HEADER")
    if bundle.get("identity_manifest_sha256") != identity_lock_sha256:
        raise ClosureError("INVALID", "BUNDLE_IDENTITY_HASH")
    activation_sha = _sha(bundle.get("activation_sha256"), "ACTIVATION")
    authority_bindings = {
        field: _sha(bundle.get(field), field.upper())
        for field in (
            "manipulation_receipt_sha256",
            "manipulation_producer_receipt_sha256",
            "trajectory_manifest_sha256",
            "runner_sha256",
        )
    }
    if (
        bundle.get("arm_count") != EXPECTED_TOTAL
        or bundle.get("frame_count") != EXPECTED_FRAMES
        or bundle.get("pair_count") != EXPECTED_PAIRS
        or bundle.get("residual_worker_pids") != []
        or bundle.get("rgb_frames_retained") is not False
        or bundle.get("sequence16_android_realtime") is not False
        or bundle.get("scientific_outcome_interpreted") is not False
    ):
        raise ClosureError("INVALID", "BUNDLE_CLOSURE_FIELDS")
    gates = bundle.get("prerequisite_gates")
    required_gates = (
        "geometry_validation",
        "quality_strength_lock",
        "formal_main_manipulation",
    )
    if not isinstance(gates, dict) or any(
        gates.get(name) != "PASS" for name in required_gates
    ):
        raise ClosureError("INVALID", "PREREQUISITE_GATES")
    entries = bundle.get("arms")
    if not isinstance(entries, list):
        raise ClosureError("INVALID", "BUNDLE_ARMS")
    if len(entries) < EXPECTED_TOTAL:
        raise ClosureError("EXECUTION_INCOMPLETE", "BUNDLE_ARM_COUNT")
    if len(entries) != EXPECTED_TOTAL:
        raise ClosureError("INVALID", "BUNDLE_ARM_COUNT")
    seen: set[str] = set()
    summaries: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ClosureError("INVALID", "BUNDLE_ARM_ENTRY")
        sequence_id = entry.get("sequence_id")
        if sequence_id not in identities or sequence_id in seen:
            raise ClosureError("INVALID", "BUNDLE_SEQUENCE_KEYSET")
        seen.add(sequence_id)
        receipt_path = entry.get("receipt_path")
        if not isinstance(receipt_path, str) or not receipt_path:
            raise ClosureError("INVALID", "RECEIPT_PATH")
        expected_receipt_sha = _sha(
            entry.get("receipt_sha256"), "RECEIPT_ENTRY"
        )
        try:
            actual_receipt_sha = receipt_hash_loader(receipt_path)
            receipt = receipt_loader(receipt_path)
        except FileNotFoundError as error:
            raise ClosureError(
                "EXECUTION_INCOMPLETE", "ATOMIC_RECEIPT_MISSING"
            ) from error
        if actual_receipt_sha != expected_receipt_sha:
            raise ClosureError("INVALID", "ATOMIC_RECEIPT_HASH")
        identity = identities[sequence_id]
        if (
            receipt.get("schema") != ARM_SCHEMA
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("terminal") != "ARM_COMPLETE"
            or receipt.get("activation_sha256") != activation_sha
            or receipt.get("identity_manifest_sha256")
            != identity_lock_sha256
        ):
            raise ClosureError("INVALID", "ATOMIC_RECEIPT_HEADER")
        if any(
            receipt.get(field) != value
            for field, value in authority_bindings.items()
        ):
            raise ClosureError("INVALID", "ATOMIC_AUTHORITY_BINDING")
        for field in (
            "sequence_id",
            "cluster_id",
            "block",
            "ordinal",
            "role",
            "arm",
            "numeric_seed_uint64",
            "frame_count",
            "pair_count",
        ):
            if receipt.get(field) != identity.get(field):
                raise ClosureError("INVALID", f"ATOMIC_IDENTITY_{field}")
        ledger_path = receipt.get("pair_ledger_path")
        if not isinstance(ledger_path, str) or not ledger_path:
            raise ClosureError("INVALID", "LEDGER_PATH")
        expected_ledger_sha = _sha(
            receipt.get("pair_ledger_sha256"), "PAIR_LEDGER"
        )
        try:
            actual_ledger_sha = ledger_hash_loader(ledger_path)
            ledger = ledger_loader(ledger_path)
        except FileNotFoundError as error:
            raise ClosureError(
                "EXECUTION_INCOMPLETE", "PAIR_LEDGER_MISSING"
            ) from error
        if actual_ledger_sha != expected_ledger_sha:
            raise ClosureError("INVALID", "PAIR_LEDGER_HASH")
        frame_path = receipt.get("frame_manifest_path")
        reduced_path = receipt.get("reduced_metrics_path")
        if (
            not isinstance(frame_path, str)
            or not frame_path
            or not isinstance(reduced_path, str)
            or not reduced_path
        ):
            raise ClosureError("INVALID", "ATOMIC_ARTIFACT_PATH")
        expected_frame_sha = _sha(
            receipt.get("frame_manifest_sha256"), "FRAME_MANIFEST"
        )
        expected_reduced_sha = _sha(
            receipt.get("reduced_metrics_sha256"), "REDUCED_METRICS"
        )
        try:
            if ledger_hash_loader(frame_path) != expected_frame_sha:
                raise ClosureError("INVALID", "FRAME_MANIFEST_HASH")
            frame_rows = ledger_loader(frame_path)
            if receipt_hash_loader(reduced_path) != expected_reduced_sha:
                raise ClosureError("INVALID", "REDUCED_METRICS_HASH")
            reduced_metrics = receipt_loader(reduced_path)
        except FileNotFoundError as error:
            raise ClosureError(
                "EXECUTION_INCOMPLETE", "ATOMIC_ARTIFACT_MISSING"
            ) from error
        if len(frame_rows) != FRAME_COUNT:
            raise ClosureError("INVALID", "FRAME_MANIFEST_COUNT")
        for frame_index, frame_row in enumerate(frame_rows):
            if (
                not isinstance(frame_row, dict)
                or frame_row.get("frame_index") != frame_index
                or not _is_frame_hash_row(frame_row)
            ):
                raise ClosureError("INVALID", "FRAME_MANIFEST_ROW")
        summary = reduce_pair_ledger(ledger, identity)
        if reduced_metrics != summary:
            raise ClosureError("INVALID", "REDUCED_METRICS_MISMATCH")
        summaries[sequence_id] = summary
    if seen != set(identities):
        raise ClosureError("EXECUTION_INCOMPLETE", "BUNDLE_IDENTITY_GRID")

    clusters: list[dict[str, Any]] = []
    for block in BLOCKS:
        for ordinal in range(20):
            arms: dict[str, Any] = {}
            for sequence_id, identity in identities.items():
                if (
                    identity["role"] == "MAIN_FACTORIAL"
                    and identity["block"] == block
                    and identity["ordinal"] == ordinal
                ):
                    summary = summaries[sequence_id]
                    arms[identity["arm"]] = {
                        "trigger_density": summary["trigger_density"],
                        "quality_failure_union_density": summary[
                            "quality_failure_union_density"
                        ],
                    }
            if set(arms) != set(MAIN_ARMS):
                raise ClosureError("EXECUTION_INCOMPLETE", "MAIN_CLUSTER_ARMS")
            clusters.append({"block": block, "ordinal": ordinal, "arms": arms})
    analysis = p3_analysis_r0.analyze(clusters)
    clean_gate, guard_gate = _gate_summaries(summaries, identities)
    terminal, subtype = _classify_terminal(
        analysis, clean_gate["passed"], guard_gate["passed"]
    )
    stage_mapping = {
        "MOTION_SUPPORTED": "IMPLEMENTATION_READY_FOR_CONFIRMATION",
        "QUALITY_SUPPORTED": "IMPLEMENTATION_READY_FOR_CONFIRMATION",
        "MIXED": "IMPLEMENTATION_READY_FOR_CONFIRMATION",
        "NO_SEPARATION_HOLD": "IMPLEMENTATION_NOT_READY",
        "INTERVENTION_NOT_EVALUABLE": "NOT_EVALUABLE",
    }
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "run_id": bundle.get("run_id"),
        "execution_state": "VALID_COMPLETE",
        "scientific_terminal": terminal,
        "quality_subtype": subtype,
        "stage_terminal": stage_mapping[terminal],
        "authority": (
            "SYNTHETIC_DEVELOPMENT_MECHANISM_ONLY_NO_AUTOMATIC_"
            "ALGORITHM_ANDROID_PRODUCT_OR_SAFETY_AUTHORITY"
        ),
        "inputs": {
            "bundle_success_sha256": bundle_sha256,
            "identity_lock_sha256": identity_lock_sha256,
            "activation_sha256": activation_sha,
        },
        "closure": {
            "atomic_receipt_count": len(entries),
            "main_sequence_count": EXPECTED_MAIN,
            "guardrail_sequence_count": EXPECTED_GUARD,
            "frame_count": EXPECTED_FRAMES,
            "pair_count": EXPECTED_PAIRS,
            "missing_sequence_count": 0,
            "duplicate_sequence_count": 0,
        },
        "prerequisite_gates": gates,
        "clean_tracking_gate": clean_gate,
        "positive_guardrail_gate": guard_gate,
        "analysis": analysis,
    }


def _resolve_artifact(
    repo_root: Path, bundle_root: Path, relative: str
) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ClosureError("INVALID", "ARTIFACT_PATH")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ClosureError("INVALID", "ARTIFACT_PATH_SCOPE")
    resolved = (bundle_root / value).resolve()
    artifacts = (repo_root / "artifacts.local").resolve()
    try:
        resolved.relative_to(artifacts)
    except ValueError as error:
        raise ClosureError("INVALID", "ARTIFACT_PATH_SCOPE") from error
    return resolved


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ClosureError("INVALID", "JSON_OBJECT")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ClosureError("INVALID", "LEDGER_ROW_OBJECT")
                rows.append(value)
    return rows


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def assemble_files(
    repo_root: Path,
    bundle_path: Path,
    identity_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    bundle_path = bundle_path.resolve()
    identity_path = identity_path.resolve()
    bundle = _read_object(bundle_path)
    identity = _read_object(identity_path)
    bundle_root = bundle_path.parent

    def path(relative: str) -> Path:
        return _resolve_artifact(repo_root, bundle_root, relative)

    return assemble_loaded_bundle(
        bundle,
        identity,
        lambda relative: _read_object(path(relative)),
        lambda relative: sha256_file(path(relative)),
        lambda relative: _read_jsonl(path(relative)),
        lambda relative: sha256_file(path(relative)),
        bundle_sha256=sha256_file(bundle_path),
        identity_lock_sha256=sha256_file(identity_path),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bundle-success", type=Path, required=True)
    parser.add_argument("--identity-lock", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = assemble_files(
            args.repo_root, args.bundle_success, args.identity_lock
        )
        exit_code = 0
    except ClosureError as error:
        result = {
            "schema": SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "execution_state": error.state,
            "error": error.code,
            "scientific_terminal": None,
            "analysis_performed": False,
        }
        exit_code = 2
    _write_exclusive(args.result.resolve(), result)
    print(
        json.dumps(
            {
                "execution_state": result["execution_state"],
                "scientific_terminal": result.get("scientific_terminal"),
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
