#!/usr/bin/env python3
"""Qualify one R4 SANPO source from obstacle reference opportunity only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_swept_envelope_label_mechanics import (
    _swept_prism_counts,
    _swept_prism_probes_world,
)
from plan_r4_obstacle_inventory_candidates import (
    LEDGER_SCHEMA,
    PROTOCOL_SCHEMA,
    SCHEMA as PLAN_SCHEMA,
    _validate_frozen_inputs,
)
from qualify_stage_b_reference_opportunity import (
    _expected_from_current_source,
    _missing_geometry_bindings,
)
from run_geometry_teacher_canary import (
    _anchor_basis,
    _known_field,
    _obstacle_points_world,
    _sha256,
    _theta_edges,
    _validate_authority,
)
from verify_sanpo_pose_geometry_authority import _load_json, _load_jsonl


SCHEMA = "blindassist_hftf_r4_obstacle_reference_opportunity_source_result"
QUALIFIED = "R4_SOURCE_OBSTACLE_REFERENCE_OPPORTUNITY_QUALIFIED"
REJECTED = "R4_SOURCE_OBSTACLE_REFERENCE_OPPORTUNITY_REJECTED"
LAYERS = ("foot", "body", "head")
EXPECTED_MECHANICS_SHA256 = (
    "a69d25d77f1e2b72f407980f005c758b965517fd032562a009f91746ea1e0e6a"
)


def _decision(
    known_coverage: dict[str, float],
    positive_by_height: dict[str, int],
    negative_by_height: dict[str, int],
    sensitivity: dict[str, dict[str, int]],
    gates: dict[str, Any],
) -> dict[str, bool]:
    return {
        "obstacle_known_coverage_each_height": all(
            known_coverage[name]
            >= float(gates["minimum_known_coverage_each_height"])
            for name in LAYERS
        ),
        "obstacle_primary_positive_each_height": all(
            positive_by_height[name]
            >= int(gates["minimum_positive_known_cells_each_height"])
            for name in LAYERS
        ),
        "obstacle_primary_negative_each_height": all(
            negative_by_height[name]
            >= int(gates["minimum_negative_known_cells_each_height"])
            for name in LAYERS
        ),
        "obstacle_all_sensitivity_thresholds_have_micro_opportunity": all(
            item["positive"] > 0 and item["negative"] > 0
            for item in sensitivity.values()
        ),
    }


def _firewall(*, reference_computed: bool) -> dict[str, bool]:
    return {
        "obstacle_reference_grid_computed": reference_computed,
        "ground_reference_computed": False,
        "candidate_grid_computed": False,
        "angular_baseline_computed": False,
        "arm_metric_or_delta_computed": False,
    }


def run(
    protocol_path: Path,
    ledger_path: Path,
    inventory_plan_path: Path,
    mechanics_path: Path,
    replay_root: Path,
    authority_path: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    ledger = _load_json(ledger_path)
    plan = _load_json(inventory_plan_path)
    mechanics = _load_json(mechanics_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "FROZEN_BEFORE_R4_OUTCOME"
        or ledger.get("schema") != LEDGER_SCHEMA
    ):
        raise ValueError("R4 protocol or burn ledger schema mismatch")
    burned = _validate_frozen_inputs(
        protocol, protocol_path, ledger, ledger_path
    )
    if _sha256(mechanics_path) != EXPECTED_MECHANICS_SHA256:
        raise ValueError("R4 mechanics protocol hash mismatch")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("terminal")
        != "R4_OBSTACLE_INVENTORY_CANDIDATE_PLAN_READY"
        or plan.get("protocol_sha256") != _sha256(protocol_path)
        or plan.get("burn_ledger_sha256") != _sha256(ledger_path)
        or plan.get("reference_outcome_read") is not False
        or plan.get("ground_outcome_read") is not False
        or plan.get("candidate_outcome_read") is not False
        or plan.get("baseline_outcome_read") is not False
    ):
        raise ValueError("R4 obstacle inventory plan binding mismatch")

    replay_root = replay_root.resolve()
    rows = _load_jsonl(replay_root / "manifest.replay.jsonl")
    if not rows:
        raise ValueError("Replay manifest is empty")
    session_ids = {str(row.get("session_id")) for row in rows}
    if len(session_ids) != 1:
        raise ValueError("R4 qualification requires one source session")
    session_id = next(iter(session_ids))
    if session_id in burned:
        raise ValueError(f"Burned source cannot enter R4: {session_id}")
    inventory_by_id = {
        str(item["session_id"]): item
        for item in plan["inventory_candidates"]
    }
    if session_id not in inventory_by_id:
        raise ValueError(
            f"Source is outside frozen R4 inventory plan: {session_id}"
        )
    source = protocol["obstacle_source_role"]
    if len(rows) != int(source["replay"]["frame_count"]):
        raise ValueError("R4 replay frame count mismatch")
    authority, authority_validation = _validate_authority(
        replay_root,
        rows,
        authority_path,
        _expected_from_current_source(
            replay_root, authority_path, session_id
        ),
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "source_session_id": session_id,
        "workflow_profile": protocol["workflow_profile"],
        "evidence_role": "OBSTACLE_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "burn_ledger_path": str(ledger_path.resolve()),
        "burn_ledger_sha256": _sha256(ledger_path),
        "inventory_plan_path": str(inventory_plan_path.resolve()),
        "inventory_plan_sha256": _sha256(inventory_plan_path),
        "inventory_eligible_rank": int(
            inventory_by_id[session_id]["inventory_eligible_rank"]
        ),
        "mechanics_protocol_path": str(mechanics_path.resolve()),
        "mechanics_protocol_sha256": _sha256(mechanics_path),
        "authority_report_path": str(authority_path.resolve()),
        "authority_report_sha256": _sha256(authority_path),
        "manifest_sha256": _sha256(
            replay_root / "manifest.replay.jsonl"
        ),
        "dataset_spec_sha256": _sha256(
            replay_root / "dataset_spec.json"
        ),
        "camera_poses_sha256": _sha256(
            replay_root / "source_metadata/camera_poses.csv"
        ),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "authority_validation": authority_validation,
    }
    if not authority_validation["ok"]:
        result.update(
            {
                "reference_only_assertions": _firewall(
                    reference_computed=False
                ),
                "terminal": REJECTED,
                "qualified": False,
                "checks": {"authority": False},
                "arm_outcome_authorized": False,
                "stage_c_execution_authorized": False,
                "student_training_authorized": False,
                "research_mainline_changed": False,
                "default_app_changed": False,
            }
        )
        return result

    spec = _load_json(replay_root / "dataset_spec.json")
    bindings = authority["source_pose_authority"]["bindings"]
    binding_by_id = {
        str(item["manifest_id"]): item for item in bindings
    }
    plane_by_id = {
        str(item["manifest_id"]): item["local_ground_plane"]
        for item in authority["ground_and_body_proxy_canary"]["per_frame"]
        if item.get("local_ground_plane") is not None
    }
    missing = _missing_geometry_bindings(
        {str(row["id"]) for row in rows},
        set(binding_by_id),
        set(plane_by_id),
    )
    if any(missing.values()):
        result.update(
            {
                "reference_only_assertions": _firewall(
                    reference_computed=False
                ),
                "terminal": REJECTED,
                "qualified": False,
                "checks": {
                    "authority": True,
                    "complete_pose_and_local_ground_bindings": False,
                },
                "missing_geometry_bindings": missing,
                "arm_outcome_authorized": False,
                "stage_c_execution_authorized": False,
                "student_training_authorized": False,
                "research_mainline_changed": False,
                "default_app_changed": False,
            }
        )
        return result

    field = mechanics["field"]
    theta_edges = _theta_edges(field)
    distance_edges = np.asarray(
        field["distance_edges_m"], dtype=np.float64
    )
    height_bands = [
        tuple(float(value) for value in field["height_bands_m"][name])
        for name in LAYERS
    ]
    widths = np.asarray(
        [
            mechanics["standard_synthetic_envelope"][
                "effective_lateral_half_width_m"
            ][name]
            for name in LAYERS
        ],
        dtype=np.float64,
    )
    obstacle = mechanics["obstacle_support"]
    known_contract = mechanics["known_support"]
    comparison_reference = source["formal_comparison"]["reference"]
    gates = source["qualification_gates"]
    thresholds = [
        int(value)
        for value in gates[
            "all_sensitivity_thresholds_micro_positive_and_negative_required"
        ]
    ]
    primary = int(gates["primary_reference_count_threshold"])
    known_count = np.zeros(3, dtype=np.int64)
    positive = np.zeros(3, dtype=np.int64)
    negative = np.zeros(3, dtype=np.int64)
    sensitivity = {
        threshold: {"positive": 0, "negative": 0}
        for threshold in thresholds
    }
    for row in rows:
        binding = binding_by_id[str(row["id"])]
        basis = _anchor_basis(binding, plane_by_id[str(row["id"])])
        points, dynamic = _obstacle_points_world(
            replay_root,
            row,
            binding,
            spec["camera"],
            stride=int(comparison_reference["point_sample_stride_xy"]),
            offset=int(comparison_reference["point_sample_offset_xy"]),
            excluded_classes=set(
                obstacle["excluded_semantic_class_ids"]
            ),
            dynamic_classes=set(
                obstacle["dynamic_provenance_class_ids"]
            ),
        )
        counts, _ = _swept_prism_counts(
            points,
            dynamic,
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        probes = _swept_prism_probes_world(
            basis,
            theta_edges,
            distance_edges,
            height_bands,
            widths,
        )
        known, _ = _known_field(
            probes,
            replay_root,
            row,
            binding,
            spec["camera"],
            len(theta_edges) - 1,
            len(distance_edges) - 1,
            len(height_bands),
            float(known_contract["depth_front_tolerance_m"]),
            int(known_contract["minimum_passing_prism_probes"]),
        )
        known_count += known.sum(axis=(0, 1))
        positive += (known & (counts >= primary)).sum(axis=(0, 1))
        negative += (known & (counts < primary)).sum(axis=(0, 1))
        for threshold in thresholds:
            sensitivity[threshold]["positive"] += int(
                (known & (counts >= threshold)).sum()
            )
            sensitivity[threshold]["negative"] += int(
                (known & (counts < threshold)).sum()
            )
    denominator = (
        len(rows) * (len(theta_edges) - 1) * (len(distance_edges) - 1)
    )
    known_coverage = {
        name: float(known_count[index] / denominator)
        for index, name in enumerate(LAYERS)
    }
    positive_by_height = {
        name: int(positive[index]) for index, name in enumerate(LAYERS)
    }
    negative_by_height = {
        name: int(negative[index]) for index, name in enumerate(LAYERS)
    }
    sensitivity_output = {
        str(threshold): value
        for threshold, value in sensitivity.items()
    }
    checks = _decision(
        known_coverage,
        positive_by_height,
        negative_by_height,
        sensitivity_output,
        gates,
    )
    checks["authority"] = True
    qualified = all(checks.values())
    result.update(
        {
            "reference_only_assertions": _firewall(
                reference_computed=True
            ),
            "terminal": QUALIFIED if qualified else REJECTED,
            "qualified": qualified,
            "fixed_denominator_per_height": denominator,
            "reference_obstacle": {
                "known_cells_by_height": {
                    name: int(known_count[index])
                    for index, name in enumerate(LAYERS)
                },
                "known_coverage_by_height": known_coverage,
                "primary_threshold": primary,
                "primary_positive_known_cells_by_height": (
                    positive_by_height
                ),
                "primary_negative_known_cells_by_height": (
                    negative_by_height
                ),
                "sensitivity_micro_opportunity": sensitivity_output,
            },
            "checks": checks,
            "arm_outcome_authorized": False,
            "stage_c_execution_authorized": False,
            "student_training_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        }
    )
    return result


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--burn-ledger", type=Path, required=True)
    parser.add_argument("--inventory-plan", type=Path, required=True)
    parser.add_argument("--mechanics-protocol", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = run(
            args.protocol.resolve(),
            args.burn_ledger.resolve(),
            args.inventory_plan.resolve(),
            args.mechanics_protocol.resolve(),
            args.replay_root.resolve(),
            args.authority.resolve(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "qualified": report["qualified"],
                    "output": str(output),
                }
            )
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
