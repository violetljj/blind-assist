"""Independent validator and descriptive analyzer for the 48-sequence DEV run.

This module never imports the DEV producer, QMS operator, R3 transport, or
formal analysis.  It independently derives identities and recomputes every
trigger, arm summary, cluster contrast, and block descriptive from JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any
from unittest import mock

from . import generator_geometry as geometry


PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
TASK_ID = (
    "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_FOUR_BLOCK_DEV_DIAGNOSTIC_R0"
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
QUALITIES = ("CLEAN", "BLUR", "LOW_TEXTURE")
ARMS = tuple(f"{motion}__{quality}" for motion in MOTIONS for quality in QUALITIES)
FRAME_COUNT = 602
PAIR_COUNT = 601
CONTRASTS = (
    "MOTION_CLEAN",
    "BLUR_STATIC",
    "LOW_TEXTURE_STATIC",
    "MOTION_X_BLUR",
    "MOTION_X_LOW_TEXTURE",
)
EXCLUSION_FILES = (
    (
        "OLD_FORMAL",
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "P4_FORMAL_IDENTITY_LOCK_R0_2026-07-29.json",
    ),
    (
        "QMS_R1_PREDECESSOR_DEV",
        "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1/predecessor_dev/identity_manifest.json",
    ),
    (
        "QMS_R1_CAL",
        "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1/new_cal/identity_manifest.json",
    ),
    (
        "OLD_PREFLIGHT",
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "PREFLIGHT_IDENTITY_LOCK_R0_2026-07-29.json",
    ),
    (
        "QMS_R1_SUCCESSOR_FORMAL",
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "QMS_R1_SUCCESSOR_FORMAL_IDENTITY_LOCK_R0_2026-07-29.json",
    ),
    (
        "QMS_R1_ACTIVATION_PREFLIGHT",
        "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "QMS_R1_FORMAL_ACTIVATION_PREFLIGHT_IDENTITY_LOCK_R0_2026-07-29.json",
    ),
)
PREFLIGHT_DIRS = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p3_transport_analysis_runtime_preflight_r0/w4/sequences",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p3_transport_analysis_runtime_preflight_r0/w8/sequences",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p3_transport_analysis_runtime_preflight_r1/w4/sequences",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p3_transport_analysis_runtime_preflight_r2/w8/sequences",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p3_transport_analysis_runtime_preflight_r3/w8/sequences",
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1_formal_activation_preflight_r0/w8/sequences",
)
TRAJECTORY_MANIFEST = (
    "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "p1_geometry_r2_keyset_repair_r0/trajectory_manifest.json"
)
OPERATOR_SOURCE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/material_residual_contraction_r1.py"
)
OPERATOR_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_OPERATOR_LOCK_R0_2026-07-29.json"
)
TRANSPORT_SOURCE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/p3_transport_r0.py"
)
TRANSPORT_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "R3_TRANSPORT_EQUIVALENCE_LOCK_R0_2026-07-29.json"
)
ANALYSIS_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "ANALYSIS_IMPLEMENTATION_LOCK_R0_2026-07-29.json"
)
RUNNER_SOURCE = (
    "scripts/research/egomotion_compensated_looming/"
    "periodic_self_motion_counterfactual_r2/"
    "qms_r1_four_block_dev_diagnostic_r0.py"
)
FORMAL_DECISION = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_SUCCESSOR_FORMAL_ACTIVATION_DECISION_R0_2026-07-29.json"
)
SUCCESSOR_FORMAL_LOCK = (
    "docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
    "QMS_R1_SUCCESSOR_FORMAL_IDENTITY_LOCK_R0_2026-07-29.json"
)
GIB = 1024**3


class InvalidIndependentDev(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    rows = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    return hashlib.sha256(canonical_bytes(rows)).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(os.fspath(path), flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _seed(block: str, ordinal: int) -> dict[str, Any]:
    token = f"{TASK_ID}|DEV_DIAGNOSTIC|{block}|MAIN|{ordinal:02d}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return {
        "token": token,
        "token_sha256": digest,
        "numeric_seed_uint64": int.from_bytes(bytes.fromhex(digest)[:8], "big"),
    }


def _scene_hash(block: str, ordinal: int) -> str:
    seed = _seed(block, ordinal)
    with mock.patch.object(
        geometry, "derive_seed", return_value=seed["numeric_seed_uint64"]
    ):
        return geometry.build_scene(block, ordinal, "MAIN")[
            "scene_geometry_sha256"
        ]


def _collect(value: Any, key: str) -> set[Any]:
    found: set[Any] = set()
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key and isinstance(child, (str, int)):
                found.add(child)
            found.update(_collect(child, key))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect(child, key))
    return found


def validate_contract(root: Path, path: Path) -> dict[str, Any]:
    value = load_json(path)
    design = value.get("design", {})
    outcomes = value.get("outcomes", {})
    reporting = value.get("reporting", {})
    bindings = value.get("bindings", {})
    expected_contrasts = {
        "MOTION_CLEAN": "PERIODIC_CLEAN-STATIC_CLEAN",
        "BLUR_STATIC": "STATIC_BLUR-STATIC_CLEAN",
        "LOW_TEXTURE_STATIC": "STATIC_LOW_TEXTURE-STATIC_CLEAN",
        "MOTION_X_BLUR": (
            "(PERIODIC_BLUR-PERIODIC_CLEAN)-(STATIC_BLUR-STATIC_CLEAN)"
        ),
        "MOTION_X_LOW_TEXTURE": (
            "(PERIODIC_LOW_TEXTURE-PERIODIC_CLEAN)-"
            "(STATIC_LOW_TEXTURE-STATIC_CLEAN)"
        ),
    }
    observed_contrasts = {
        item.get("name"): item.get("formula")
        for item in value.get("contrasts", [])
        if isinstance(item, dict)
    }
    if (
        value.get("task_id") != TASK_ID
        or value.get("stage") != "DEVELOPMENT_DIAGNOSTIC"
        or design.get("blocks") != list(BLOCKS)
        or design.get("new_scene_seeds_per_block") != 2
        or design.get("latent_cluster_count") != 8
        or design.get("arms") != list(ARMS)
        or design.get("sequence_count") != 48
        or design.get("observational_unit")
        != "scene_seed_x_motion_block_cluster"
        or outcomes.get("threshold_operator") != "strict_greater_than"
        or outcomes.get("threshold_per_s") != 0.01
        or outcomes.get("required_consecutive_pairs") != 3
        or outcomes.get("abstention_resets_streak") is not True
        or observed_contrasts != expected_contrasts
        or reporting.get("bootstrap_or_confidence_interval") is not False
        or reporting.get("p_values") is not False
        or reporting.get("formal_max_t") is not False
        or value.get("formal_identity_execution") is not False
        or value.get("formal_identity_lock_read_for_exclusion_only") is not True
        or value.get("formal_execution_authorized_by_this_contract") is not False
    ):
        raise InvalidIndependentDev("CONTRACT")
    expected_bindings = {
        "qms_r1_operator_source_sha256": sha256_file(root / OPERATOR_SOURCE),
        "qms_r1_operator_lock_sha256": sha256_file(root / OPERATOR_LOCK),
        "r3_transport_lock_sha256": sha256_file(root / TRANSPORT_LOCK),
        "r3_transport_source_sha256": sha256_file(root / TRANSPORT_SOURCE),
        "formal_analysis_lock_sha256": sha256_file(root / ANALYSIS_LOCK),
        "formal_activation_decision_sha256": sha256_file(
            root / FORMAL_DECISION
        ),
    }
    for key, expected in expected_bindings.items():
        if bindings.get(key) != expected:
            raise InvalidIndependentDev(f"CONTRACT_BINDING:{key}")
    return value


def _exclusion_union(
    root: Path, source_records: list[dict[str, Any]]
) -> dict[str, set[Any]]:
    fields = (
        "numeric_seed_uint64",
        "token",
        "token_sha256",
        "cluster_id",
        "sequence_id",
        "scene_geometry_sha256",
    )
    union = {field: set() for field in fields}
    expected_files = dict(EXCLUSION_FILES)
    observed_file_records = {
        item.get("label"): item
        for item in source_records
        if item.get("label") in expected_files
    }
    if set(observed_file_records) != set(expected_files):
        raise InvalidIndependentDev("EXCLUSION_FILE_SET")
    for label, relative in EXCLUSION_FILES:
        record = observed_file_records[label]
        path = root / relative
        if (
            record.get("path") != relative
            or record.get("sha256") != sha256_file(path)
        ):
            raise InvalidIndependentDev(f"EXCLUSION_FILE:{label}")
        value = load_json(path)
        for field in fields:
            union[field].update(_collect(value, field))
    directory_records = [
        item
        for item in source_records
        if item.get("label") == "PREFLIGHT_RUNTIME_SCENE_RECEIPTS"
    ]
    if {item.get("path") for item in directory_records} != set(PREFLIGHT_DIRS):
        raise InvalidIndependentDev("EXCLUSION_DIRECTORY_SET")
    for record in directory_records:
        directory = root / record["path"]
        if record.get("tree_sha256") != tree_sha256(directory):
            raise InvalidIndependentDev("EXCLUSION_DIRECTORY_DRIFT")
        actual_scenes = set()
        for receipt in directory.glob("*/receipt.json"):
            actual_scenes.update(
                _collect(load_json(receipt), "scene_geometry_sha256")
            )
        if sorted(actual_scenes) != record.get("scene_geometry_sha256"):
            raise InvalidIndependentDev("EXCLUSION_SCENE_SET")
        union["scene_geometry_sha256"].update(actual_scenes)
    return union


def validate_identity_lock(
    root: Path, path: Path, contract_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, int]]:
    value = load_json(path)
    counts = value.get("counts", {})
    identities = value.get("identities")
    seeds = value.get("seeds")
    if (
        value.get("task_id") != TASK_ID
        or value.get("role") != "DEVELOPMENT_DIAGNOSTIC_ONLY"
        or counts
        != {
            "blocks": 4,
            "seeds_per_block": 2,
            "clusters": 8,
            "arms_per_cluster": 6,
            "sequences": 48,
            "frames": 28896,
            "pairs": 28848,
        }
        or not isinstance(identities, list)
        or len(identities) != 48
        or not isinstance(seeds, list)
        or len(seeds) != 8
        or value.get("contract", {}).get("path")
        != contract_path.relative_to(root).as_posix()
        or value.get("contract", {}).get("sha256")
        != sha256_file(contract_path)
        or value.get("formal_execution_authorized") is not False
        or value.get("formal_sequences_run") != 0
    ):
        raise InvalidIndependentDev("IDENTITY_LOCK_HEADER")
    trajectories = load_json(root / TRAJECTORY_MANIFEST)
    expected: dict[str, dict[str, Any]] = {}
    expected_seeds = []
    for block in BLOCKS:
        for ordinal in (0, 1):
            seed = _seed(block, ordinal)
            scene_hash = _scene_hash(block, ordinal)
            cluster_id = f"QMSR1_DEV_{block}_MAIN_{ordinal:02d}"
            expected_seeds.append(
                {
                    "block": block,
                    "ordinal": ordinal,
                    **seed,
                    "cluster_id": cluster_id,
                    "scene_geometry_sha256": scene_hash,
                }
            )
            for arm_ordinal, arm in enumerate(ARMS):
                motion, quality_name = arm.split("__", 1)
                sequence_id = (
                    f"QMSR1_DEV_{block}_MAIN_{ordinal:02d}__{arm}"
                )
                expected[sequence_id] = {
                    "sequence_id": sequence_id,
                    "cluster_id": cluster_id,
                    "block": block,
                    "ordinal": ordinal,
                    "role": "DEV_DIAGNOSTIC",
                    "arm": arm,
                    "arm_ordinal": arm_ordinal,
                    "motion": motion,
                    "quality": quality_name,
                    **seed,
                    "scene_geometry_sha256": scene_hash,
                    "trajectory_sha256": (
                        trajectories[block]["periodic_pose_sha256"]
                        if motion == "PERIODIC_6DOF_SELF_MOTION"
                        else hashlib.sha256(
                            geometry.canonical_bytes({"static": FRAME_COUNT})
                        ).hexdigest()
                    ),
                    "frame_count": FRAME_COUNT,
                    "pair_count": PAIR_COUNT,
                }
    if seeds != expected_seeds:
        raise InvalidIndependentDev("SEED_GRID")
    if identities != list(expected.values()):
        raise InvalidIndependentDev("IDENTITY_GRID")
    if value.get("identity_set_sha256") != hashlib.sha256(
        canonical_bytes(identities)
    ).hexdigest():
        raise InvalidIndependentDev("IDENTITY_SET_HASH")
    excluded = _exclusion_union(root, value.get("exclusion_sources", []))
    overlap_counts = {}
    for field, old_values in excluded.items():
        new_values = _collect(identities + seeds, field)
        overlap_counts[field] = len(new_values & old_values)
        if overlap_counts[field]:
            raise InvalidIndependentDev(f"IDENTITY_OVERLAP:{field}")
    return value, expected, overlap_counts


def _type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def reduce_rows(
    rows: list[dict[str, Any]], identity: dict[str, Any]
) -> dict[str, Any]:
    if len(rows) != PAIR_COUNT:
        raise InvalidIndependentDev("PAIR_COUNT")
    envelope_fields = (
        "sequence_id",
        "cluster_id",
        "block",
        "ordinal",
        "role",
        "arm",
    )
    streak = trigger_count = failure_count = clean_count = evaluable_count = 0
    responses = []
    for index, row in enumerate(rows):
        if (
            row.get("pair_index") != index
            or any(row.get(key) != identity[key] for key in envelope_fields)
        ):
            raise InvalidIndependentDev("PAIR_ORDER_OR_ENVELOPE")
        evaluable = row.get("evaluable")
        response = row.get("compensated_expansion_median_per_s")
        if evaluable is True:
            if (
                not isinstance(response, (int, float))
                or isinstance(response, bool)
                or not math.isfinite(response)
            ):
                raise InvalidIndependentDev("EVALUABLE_RESPONSE")
            response_value = float(response)
            responses.append(response_value)
            evaluable_count += 1
            streak = streak + 1 if response_value > 0.01 else 0
        elif evaluable is False:
            if response is not None:
                raise InvalidIndependentDev("ABSTENTION_RESPONSE")
            streak = 0
        else:
            raise InvalidIndependentDev("EVALUABLE_FLAG")
        trigger = streak >= 3
        if row.get("compensated_three_pair_trigger") is not trigger:
            raise InvalidIndependentDev("FORGED_TRIGGER")
        trigger_count += int(trigger)
        detected = row.get("detected_feature_count")
        consistent = row.get("forward_backward_consistent_count")
        fraction = row.get("forward_backward_consistent_fraction")
        occupied = row.get("occupied_3x3_cells")
        fb_error = row.get("median_forward_backward_error_px")
        if (
            not isinstance(detected, int)
            or isinstance(detected, bool)
            or detected < 0
            or not isinstance(consistent, int)
            or isinstance(consistent, bool)
            or consistent < 0
            or consistent > detected
            or not isinstance(occupied, int)
            or occupied not in range(10)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0.0 <= float(fraction) <= 1.0
        ):
            raise InvalidIndependentDev("TRACKING_FIELDS")
        if fb_error is not None and (
            not isinstance(fb_error, (int, float))
            or not math.isfinite(fb_error)
            or float(fb_error) < 0.0
        ):
            raise InvalidIndependentDev("FB_ERROR")
        collapse = (
            detected < 60 or consistent < 60 or float(fraction) < 0.50
        )
        fb_failure = fb_error is None or float(fb_error) > 0.75
        failure_count += int(collapse or fb_failure)
        clean_count += int(
            not collapse and not fb_failure and occupied >= 5
        )
    return {
        "scheduled_pair_count": PAIR_COUNT,
        "evaluable_pair_count": evaluable_count,
        "trigger_count": trigger_count,
        "trigger_density": trigger_count / PAIR_COUNT,
        "quality_failure_union_count": failure_count,
        "quality_failure_union_density": failure_count / PAIR_COUNT,
        "clean_trackable_pair_count": clean_count,
        "clean_sequence_trackable": clean_count >= 421,
        "evaluable_fraction": evaluable_count / PAIR_COUNT,
        "response_median_per_s": (
            statistics.median(responses) if responses else None
        ),
        "response_mean_per_s": (
            sum(responses) / len(responses) if responses else None
        ),
        "response_p90_per_s": (
            _type7(responses, 0.90) if responses else None
        ),
    }


def _compare_summary(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    if set(observed) != set(expected):
        raise InvalidIndependentDev("REDUCED_KEYSET")
    for key, value in expected.items():
        actual = observed[key]
        if isinstance(value, float):
            if not math.isclose(actual, value, rel_tol=0.0, abs_tol=1e-15):
                raise InvalidIndependentDev(f"REDUCED_FLOAT:{key}")
        elif actual != value:
            raise InvalidIndependentDev(f"REDUCED_VALUE:{key}")


def _contrasts(by_arm: dict[str, dict[str, Any]]) -> dict[str, float]:
    density = {arm: by_arm[arm]["trigger_density"] for arm in ARMS}
    sc = density["STATIC_CAMERA__CLEAN"]
    sb = density["STATIC_CAMERA__BLUR"]
    sl = density["STATIC_CAMERA__LOW_TEXTURE"]
    pc = density["PERIODIC_6DOF_SELF_MOTION__CLEAN"]
    pb = density["PERIODIC_6DOF_SELF_MOTION__BLUR"]
    pl = density["PERIODIC_6DOF_SELF_MOTION__LOW_TEXTURE"]
    return {
        "MOTION_CLEAN": pc - sc,
        "BLUR_STATIC": sb - sc,
        "LOW_TEXTURE_STATIC": sl - sc,
        "MOTION_X_BLUR": (pb - pc) - (sb - sc),
        "MOTION_X_LOW_TEXTURE": (pl - pc) - (sl - sc),
    }


def validate_bundle(
    root: Path,
    bundle: Path,
    identity_path: Path,
    expected: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    lock = load_json(identity_path)
    contract_path = root / lock.get("contract", {}).get("path", "")
    expected_bindings = {
        "contract_sha256": sha256_file(contract_path),
        "identity_lock_sha256": sha256_file(identity_path),
        "operator_source_sha256": sha256_file(root / OPERATOR_SOURCE),
        "operator_lock_sha256": sha256_file(root / OPERATOR_LOCK),
        "r3_transport_lock_sha256": sha256_file(root / TRANSPORT_LOCK),
        "r3_transport_source_sha256": sha256_file(root / TRANSPORT_SOURCE),
        "runner_source_sha256": sha256_file(root / RUNNER_SOURCE),
    }
    run_path = bundle / "run_receipt.json"
    run = load_json(run_path)
    if (
        run.get("task_id") != TASK_ID
        or run.get("terminal")
        != "DEV_DIAGNOSTIC_EXECUTION_COMPLETE / VALIDATION_REQUIRED"
        or run.get("identity_lock_sha256") != sha256_file(identity_path)
        or run.get("counts")
        != {"clusters": 8, "sequences": 48, "frames": 28896, "pairs": 28848}
        or run.get("bindings") != expected_bindings
        or run.get("formal_execution_authorized_by_this_run") is not False
    ):
        raise InvalidIndependentDev("RUN_RECEIPT")
    resource = run.get("resource", {})
    if (
        resource.get("available_ram_at_launch_bytes", 0) < 8 * GIB
        or resource.get("minimum_available_ram_bytes", 0) < 4 * GIB
        or resource.get("swap_in_delta") != 0
        or resource.get("swap_out_delta") != 0
        or resource.get("heartbeat_max_interval_seconds", 31) > 30
        or resource.get("residual_worker_pids") != []
    ):
        raise InvalidIndependentDev("RESOURCE")
    firewall = run.get("formal_firewall", {})
    successor_formal = (
        root
        / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
        "qms_r1_successor_formal"
    )
    if (
        firewall.get("predecessor_formal_tree_sha256_before")
        != firewall.get("predecessor_formal_tree_sha256_after")
        or firewall.get("successor_formal_path_absent") is not True
        or successor_formal.exists()
        or firewall.get("formal_activation_decision_sha256_before")
        != firewall.get("formal_activation_decision_sha256_after")
        or firewall.get("formal_activation_decision_sha256_after")
        != sha256_file(root / FORMAL_DECISION)
        or firewall.get("successor_formal_identity_lock_sha256_before")
        != firewall.get("successor_formal_identity_lock_sha256_after")
        or firewall.get("successor_formal_identity_lock_sha256_after")
        != sha256_file(root / SUCCESSOR_FORMAL_LOCK)
        or firewall.get("formal_sequences_run") != 0
        or firewall.get("successor_formal_authority_consumed") is not False
    ):
        raise InvalidIndependentDev("FORMAL_FIREWALL")
    if any((bundle / "staging").glob("*")):
        raise InvalidIndependentDev("NONATOMIC_STAGING_REMAINS")
    clusters = run.get("clusters")
    if not isinstance(clusters, list) or len(clusters) != 8:
        raise InvalidIndependentDev("CLUSTER_COUNT")
    arm_summaries: dict[str, dict[str, Any]] = {}
    seen_clusters = set()
    for cluster in clusters:
        cluster_id = cluster.get("cluster_id")
        if cluster_id in seen_clusters:
            raise InvalidIndependentDev("DUPLICATE_CLUSTER")
        seen_clusters.add(cluster_id)
        cluster_receipt_path = bundle / cluster.get(
            "cluster_receipt_path", ""
        )
        try:
            cluster_receipt_path.resolve().relative_to(bundle.resolve())
        except ValueError as error:
            raise InvalidIndependentDev("CLUSTER_PATH_OUTSIDE") from error
        if (
            not cluster_receipt_path.is_file()
            or sha256_file(cluster_receipt_path)
            != cluster.get("cluster_receipt_sha256")
        ):
            raise InvalidIndependentDev("CLUSTER_RECEIPT_HASH")
        disk_cluster = load_json(cluster_receipt_path)
        for key in (
            "cluster_id",
            "block",
            "ordinal",
            "sequence_count",
            "arm_outputs",
            "qms_render_pair_calls",
            "static_render_pair_calls",
            "static_render_reuse_hits",
            "periodic_render_pair_calls",
            "operator_source_sha256",
            "terminal",
        ):
            if disk_cluster.get(key) != cluster.get(key):
                raise InvalidIndependentDev(f"CLUSTER_RECEIPT:{key}")
        if (
            cluster.get("qms_render_pair_calls") != 603
            or cluster.get("static_render_pair_calls") != 1
            or cluster.get("static_render_reuse_hits") != 601
            or cluster.get("periodic_render_pair_calls") != 602
            or cluster.get("operator_source_sha256")
            != sha256_file(root / OPERATOR_SOURCE)
            or cluster.get("terminal") != "DEV_CLUSTER_COMPLETE"
        ):
            raise InvalidIndependentDev("CLUSTER_EXECUTION")
        outputs = cluster.get("arm_outputs")
        if not isinstance(outputs, list) or len(outputs) != 6:
            raise InvalidIndependentDev("CLUSTER_ARM_COUNT")
        for output in outputs:
            sequence_id = output.get("sequence_id")
            identity = expected.get(sequence_id)
            if identity is None or identity["cluster_id"] != cluster_id:
                raise InvalidIndependentDev("ARM_IDENTITY")
            receipt_path = bundle / output.get("receipt_path", "")
            try:
                receipt_path.resolve().relative_to(bundle.resolve())
            except ValueError as error:
                raise InvalidIndependentDev("ARM_PATH_OUTSIDE") from error
            if (
                not receipt_path.is_file()
                or sha256_file(receipt_path)
                != output.get("receipt_sha256")
            ):
                raise InvalidIndependentDev("ARM_RECEIPT_HASH")
            receipt = load_json(receipt_path)
            if receipt.get("bindings") != expected_bindings:
                raise InvalidIndependentDev("ARM_RECEIPT_BINDINGS")
            for key in (
                "sequence_id",
                "cluster_id",
                "block",
                "ordinal",
                "role",
                "arm",
                "numeric_seed_uint64",
                "scene_geometry_sha256",
                "trajectory_sha256",
                "token",
                "token_sha256",
                "motion",
                "quality",
                "frame_count",
                "pair_count",
            ):
                if receipt.get(key) != identity[key]:
                    raise InvalidIndependentDev(f"ARM_RECEIPT_IDENTITY:{key}")
            arm_dir = receipt_path.parent
            ledger_path = arm_dir / "pair_ledger.jsonl"
            reduced_path = arm_dir / "reduced_metrics.json"
            if (
                receipt.get("pair_ledger_sha256")
                != sha256_file(ledger_path)
                or receipt.get("reduced_metrics_sha256")
                != sha256_file(reduced_path)
                or receipt.get("terminal") != "DEV_ARM_COMPLETE"
            ):
                raise InvalidIndependentDev("ARM_ARTIFACT_BINDING")
            rows = load_jsonl(ledger_path)
            reduced = reduce_rows(rows, identity)
            _compare_summary(load_json(reduced_path), reduced)
            arm_summaries[sequence_id] = reduced
    if set(arm_summaries) != set(expected):
        raise InvalidIndependentDev("ARM_KEYSET")
    return run, arm_summaries


def analyze(
    expected: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    units = []
    for block in BLOCKS:
        for ordinal in (0, 1):
            identities = [
                item
                for item in expected.values()
                if item["block"] == block and item["ordinal"] == ordinal
            ]
            by_arm = {
                item["arm"]: summaries[item["sequence_id"]]
                for item in identities
            }
            units.append(
                {
                    "cluster_id": identities[0]["cluster_id"],
                    "block": block,
                    "ordinal": ordinal,
                    "arm_metrics": by_arm,
                    "trigger_density_contrasts": _contrasts(by_arm),
                }
            )
    block_summaries = {}
    for block in BLOCKS:
        block_units = [unit for unit in units if unit["block"] == block]
        block_summaries[block] = {
            name: {
                "n_clusters": 2,
                "mean": statistics.mean(
                    unit["trigger_density_contrasts"][name]
                    for unit in block_units
                ),
                "minimum": min(
                    unit["trigger_density_contrasts"][name]
                    for unit in block_units
                ),
                "maximum": max(
                    unit["trigger_density_contrasts"][name]
                    for unit in block_units
                ),
                "positive_count": sum(
                    unit["trigger_density_contrasts"][name] > 0.0
                    for unit in block_units
                ),
            }
            for name in CONTRASTS
        }
    overall = {}
    for name in CONTRASTS:
        values = [
            unit["trigger_density_contrasts"][name] for unit in units
        ]
        block_means = [
            block_summaries[block][name]["mean"] for block in BLOCKS
        ]
        overall[name] = {
            "n_clusters": 8,
            "cluster_mean": statistics.mean(values),
            "cluster_median": statistics.median(values),
            "cluster_minimum": min(values),
            "cluster_maximum": max(values),
            "positive_cluster_count": sum(value > 0.0 for value in values),
            "negative_cluster_count": sum(value < 0.0 for value in values),
            "zero_cluster_count": sum(value == 0.0 for value in values),
            "equal_four_block_mean": statistics.mean(block_means),
            "positive_block_count": sum(value > 0.0 for value in block_means),
        }
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_four_block_dev_result.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "terminal": "DEV_DIAGNOSTIC_COMPLETE / DESCRIPTIVE_ONLY",
        "units": units,
        "block_summaries": block_summaries,
        "overall_trigger_density_contrasts": overall,
        "analysis_controls": {
            "inferential_analysis_performed": False,
            "bootstrap_performed": False,
            "confidence_intervals_reported": False,
            "p_values_reported": False,
            "max_t_performed": False,
            "formal_classification_performed": False,
        },
        "claim_ceiling": (
            "CONTROLLED_GENERATOR_INTERNAL_DEVELOPMENT_DIAGNOSTIC_ONLY"
        ),
        "formal_authority": "UNCHANGED_NOT_CONSUMED",
    }


def validate_all(
    root: Path,
    contract_path: Path,
    identity_path: Path,
    bundle: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_contract(root, contract_path)
    lock, expected, overlaps = validate_identity_lock(
        root, identity_path, contract_path
    )
    run, summaries = validate_bundle(root, bundle, identity_path, expected)
    result = analyze(expected, summaries)
    receipt = {
        "schema": "rcle.periodic_self_motion_counterfactual.qms_r1_four_block_dev_independent_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "task_id": TASK_ID,
        "validated": True,
        "errors": [],
        "terminal": "VALID / DEV_DIAGNOSTIC_COMPLETE",
        "counts": lock["counts"],
        "identity_set_sha256": lock["identity_set_sha256"],
        "overlap_counts": overlaps,
        "run_receipt_sha256": sha256_file(bundle / "run_receipt.json"),
        "analysis_result_sha256": hashlib.sha256(
            canonical_bytes(result)
        ).hexdigest(),
        "resource": run["resource"],
        "formal_firewall": run["formal_firewall"],
        "analysis_controls": result["analysis_controls"],
        "claim_ceiling": result["claim_ceiling"],
        "formal_authority": "UNCHANGED_NOT_CONSUMED",
        "formal_sequences_run": 0,
        "formal_r3_pair_core_calls": 0,
        "validator_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    return result, receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--identity-lock", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    bundle = (root / args.bundle).resolve()
    result, receipt = validate_all(
        root,
        (root / args.contract).resolve(),
        (root / args.identity_lock).resolve(),
        bundle,
    )
    result_path = (root / args.result).resolve()
    receipt_path = (root / args.receipt).resolve()
    write_exclusive(result_path, result)
    receipt["analysis_result_path"] = result_path.relative_to(root).as_posix()
    receipt["analysis_result_file_sha256"] = sha256_file(result_path)
    write_exclusive(receipt_path, receipt)
    print(
        json.dumps(
            {
                "terminal": receipt["terminal"],
                "clusters": receipt["counts"]["clusters"],
                "sequences": receipt["counts"]["sequences"],
                "formal_sequences_run": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
