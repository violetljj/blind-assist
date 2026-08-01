#!/usr/bin/env python3
"""Audit E0.1 fresh dev/heldout 0.4 s teacher opportunity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_stage_c_e0_teacher_opportunity as e0_opp  # noqa: E402
import run_stage_c_d0_semantic_independent_label_readiness as d0  # noqa: E402
import run_stage_c_d1_causal_future_label_mechanics as d1  # noqa: E402


SCHEMA = "blindassist_hftf_stage_c_e0_1_teacher_opportunity"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_foot_ground_student_canary_e0_1"
)
PROTOCOL_STATUS = (
    "FROZEN_BEFORE_FRESH_EVALUATION_RGB_DEPTH_OR_LABEL_OUTCOME"
)
TRANSPORT_SCHEMA = "blindassist_hftf_stage_c_e0_1_fresh_transport"


def _load_json(path: Path) -> dict[str, Any]:
    return e0_opp._load_json(path)


def _sha256(path: Path) -> str:
    return e0_opp._sha256(path)


def _anchors(rows: int) -> list[int]:
    return list(range(5, rows - 2, 5))


def _validate(
    protocol: dict[str, Any],
    protocol_path: Path,
    transport: dict[str, Any],
    media_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("Stage C E0.1 protocol is not frozen")
    if (
        transport.get("schema") != TRANSPORT_SCHEMA
        or transport.get("terminal")
        != "E0_1_FRESH_EVALUATION_MEDIA_TRANSPORT_SUPPORTED"
        or not transport.get("all_transport_gates_pass")
    ):
        raise ValueError("E0.1 fresh transport is not supported")
    if transport.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("E0.1 transport protocol binding mismatch")
    if Path(str(transport["media_root"])).resolve() != media_root:
        raise ValueError("E0.1 transport media-root mismatch")
    if not transport.get(
        "fresh_0_4_s_teacher_opportunity_audit_authorized"
    ):
        raise ValueError("E0.1 0.4 s teacher audit is not authorized")
    if transport.get("fresh_evaluation_geometry_label_outcome_read"):
        raise ValueError("E0.1 transport unexpectedly read label outcome")
    e0_path = (
        protocol_path.parent
        / protocol["parent_bindings"]["e0_protocol_path"]
    ).resolve()
    if _sha256(e0_path) != protocol["parent_bindings"][
        "e0_protocol_sha256"
    ]:
        raise ValueError("E0.1 parent E0 protocol mismatch")
    e0_protocol = _load_json(e0_path)
    d1_parent = e0_protocol["parent_bindings"]
    d1_path = (e0_path.parent / d1_parent["d1_protocol_path"]).resolve()
    d1_protocol = _load_json(d1_path)
    _, d0_protocol = d1._validate_protocol(
        d1_protocol, d1_path, repo_root
    )
    return d1_protocol, d0_protocol


def _run_source(
    source: dict[str, Any],
    media_root: Path,
    camera: dict[str, float],
    heights: dict[str, float],
    d1_protocol: dict[str, Any],
    d0_protocol: dict[str, Any],
) -> dict[str, Any]:
    trajectory = source["trajectory"]
    frame = pl.read_parquet(
        media_root / source["files"]["pose"]["path"]
    ).sort("frame")
    x = frame["cart_x"].to_numpy()
    y = frame["cart_y"].to_numpy()
    yaw = d1._yaw_from_frame(frame)
    anchors = _anchors(frame.height)
    needed = {
        index for anchor in anchors for index in (anchor, anchor + 2)
    }
    decoded = d0._decode_selected_depth(
        media_root / source["files"]["depth"]["path"], needed
    )
    if set(decoded) != needed:
        raise ValueError(f"{trajectory}: incomplete 0.4 s depth decode")
    profiles: dict[int, list[dict[str, Any]]] = {}
    plane_known = 0
    for index in sorted(needed):
        points = d0._project_depth(
            decoded[index],
            camera,
            d0_protocol["depth_projection"],
        )
        plane = d0._fit_ground_plane(
            points,
            float(heights[trajectory]),
            trajectory,
            index,
            d0_protocol["ground_plane_reader"],
        )
        plane_known += plane is not None
        profiles[index] = d0._surface_profiles(
            points,
            plane,
            d0_protocol["surface_profile_reader"],
        )
    mapping_gate = d1_protocol["ordered_gates"][
        "odometry_mapping_each_source_each_horizon"
    ]
    mapping = d1._odometry_mapping(
        x,
        y,
        yaw,
        2,
        float(mapping_gate["moving_distance_threshold_m"]),
    )
    contract = d1_protocol["formal_anchor_selection"]
    rebin = d1_protocol["reprojection_and_rebin"]
    distances = rebin["target_distance_m"]
    candidate_cells = 0
    known = 0
    no_risk = 0
    risk = 0
    added = 0
    lost = 0
    unknown_to_safe = 0
    eligible_count = 0
    risk_anchors: set[int] = set()
    risk_directions: set[float] = set()
    for anchor in anchors:
        anchor_position = np.array([x[anchor], y[anchor]])
        history_position = np.array([x[anchor - 2], y[anchor - 2]])
        origin, speed = d1._causal_origin(
            anchor_position,
            history_position,
            float(contract["history_velocity_interval_s"]),
            0.4,
        )
        if speed > float(contract["maximum_history_speed_mps"]):
            continue
        eligible_count += 1
        current = d1._profile_observations(
            profiles[anchor],
            anchor_position,
            float(yaw[anchor]),
            distances,
        )
        future_position = np.array([x[anchor + 2], y[anchor + 2]])
        future = d1._profile_observations(
            profiles[anchor + 2],
            future_position,
            float(yaw[anchor + 2]),
            distances,
        )
        baseline = d1._rebin(
            current,
            origin,
            float(yaw[anchor]),
            rebin,
            d0_protocol["surface_profile_reader"],
        )
        candidate = d1._rebin(
            current + future,
            origin,
            float(yaw[anchor]),
            rebin,
            d0_protocol["surface_profile_reader"],
        )
        comparison = d1._comparison(baseline, candidate)
        candidate_cells += len(candidate)
        known += sum(cell["state"] != "UNKNOWN" for cell in candidate)
        no_risk += sum(
            cell["state"]
            == "KNOWN_NO_GROUND_CONTINUITY_RISK_PROXY"
            for cell in candidate
        )
        risks = [
            cell
            for cell in candidate
            if cell["state"]
            == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
        ]
        risk += len(risks)
        if risks:
            risk_anchors.add(anchor)
            risk_directions.update(
                float(cell["direction_degrees"]) for cell in risks
            )
        added += comparison["candidate_added_known_direction_count"]
        lost += comparison["candidate_lost_known_direction_count"]
        unknown_to_safe += comparison["unknown_to_safe_violation_count"]
    return {
        "role": source["role"],
        "trajectory": trajectory,
        "formal_anchor_count": len(anchors),
        "teacher_depth_frame_count": len(needed),
        "ground_plane_known_fraction": (
            plane_known / len(needed) if needed else 0.0
        ),
        "history_speed_eligible_fraction": (
            eligible_count / len(anchors) if anchors else 0.0
        ),
        "odometry_mapping_0_4_s": mapping,
        "candidate_direction_cell_count": candidate_cells,
        "candidate_known_direction_count": known,
        "candidate_known_direction_fraction_0_4_s": (
            known / candidate_cells if candidate_cells else 0.0
        ),
        "candidate_known_no_risk_count": no_risk,
        "candidate_risk_proxy_cell_count": risk,
        "future_added_known_direction_count": added,
        "candidate_lost_known_direction_count": lost,
        "unknown_to_safe_violation_count": unknown_to_safe,
        "distinct_risk_proxy_anchors": sorted(risk_anchors),
        "distinct_risk_proxy_directions": sorted(risk_directions),
    }


def _source_gate(
    source: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    checks = (
        ("formal_anchor_count", "minimum_formal_anchor_count"),
        (
            "ground_plane_known_fraction",
            "minimum_ground_plane_known_fraction",
        ),
        (
            "history_speed_eligible_fraction",
            "minimum_history_speed_eligible_fraction",
        ),
        (
            "candidate_known_direction_fraction_0_4_s",
            "minimum_candidate_known_direction_fraction_0_4_s",
        ),
    )
    for metric, threshold in checks:
        if float(source[metric]) < float(gate[threshold]):
            failures.append(metric)
    if source["candidate_lost_known_direction_count"] > int(
        gate["maximum_candidate_known_cells_lost_vs_baseline"]
    ):
        failures.append("candidate_lost_known_direction_count")
    if source["unknown_to_safe_violation_count"] > int(
        gate["maximum_unknown_to_safe_violations"]
    ):
        failures.append("unknown_to_safe_violation_count")
    return {
        "role": source["role"],
        "trajectory": source["trajectory"],
        "passed": not failures,
        "failures": failures,
    }


def _role_gate(
    sources: list[dict[str, Any]], gate: dict[str, Any]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for role in ("dev", "heldout"):
        source = next(item for item in sources if item["role"] == role)
        role_gate = gate[role]
        metrics = {
            "risk_proxy_cell_count": source[
                "candidate_risk_proxy_cell_count"
            ],
            "distinct_risk_proxy_anchor_count": len(
                source["distinct_risk_proxy_anchors"]
            ),
            "distinct_sources_with_risk_proxy": int(
                source["candidate_risk_proxy_cell_count"] > 0
            ),
            "distinct_risk_proxy_direction_count": len(
                source["distinct_risk_proxy_directions"]
            ),
            "known_no_risk_cell_count": source[
                "candidate_known_no_risk_count"
            ],
        }
        failures: list[str] = []
        pairs = (
            ("risk_proxy_cell_count", "minimum_risk_proxy_cells"),
            (
                "distinct_risk_proxy_anchor_count",
                "minimum_distinct_risk_proxy_anchors",
            ),
            (
                "distinct_sources_with_risk_proxy",
                "minimum_distinct_sources_with_risk_proxy",
            ),
            (
                "distinct_risk_proxy_direction_count",
                "minimum_distinct_risk_proxy_directions",
            ),
        )
        for metric, threshold in pairs:
            if metrics[metric] < int(role_gate[threshold]):
                failures.append(metric)
        if metrics["known_no_risk_cell_count"] < int(
            gate["minimum_known_no_risk_cells_each_role"]
        ):
            failures.append("known_no_risk_cell_count")
        results.append(
            {
                "role": role,
                "metrics": metrics,
                "passed": not failures,
                "failures": failures,
            }
        )
    return results


def audit(
    protocol_path: Path,
    transport_path: Path,
    media_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    transport = _load_json(transport_path)
    d1_protocol, d0_protocol = _validate(
        protocol,
        protocol_path,
        transport,
        media_root,
        repo_root,
    )
    camera = _load_json(media_root / "meta/camera_rgb.json")
    heights = _load_json(media_root / "meta/heights.json")
    sources = [
        _run_source(
            source,
            media_root,
            camera,
            heights,
            d1_protocol,
            d0_protocol,
        )
        for source in protocol["fresh_evaluation_sources"]
    ]
    gates = protocol["ordered_execution_gates"]
    source_results = [
        _source_gate(
            source, gates["teacher_mechanics_each_fresh_source"]
        )
        for source in sources
    ]
    role_results = _role_gate(
        sources, gates["fresh_role_label_opportunity"]
    )
    teacher_pass = all(item["passed"] for item in source_results)
    opportunity_pass = all(item["passed"] for item in role_results)
    if not teacher_pass:
        terminal = gates["teacher_mechanics_each_fresh_source"][
            "failure_terminal"
        ]
    elif not opportunity_pass:
        terminal = gates["fresh_role_label_opportunity"][
            "failure_terminal"
        ]
    else:
        terminal = (
            "E0_1_FRESH_EVALUATION_TEACHER_AND_OPPORTUNITY_SUPPORTED"
        )
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "transport_path": str(transport_path),
        "transport_sha256": _sha256(transport_path),
        "media_root": str(media_root),
        "source_reports": sources,
        "source_teacher_gate_results": source_results,
        "fresh_role_opportunity_gate_results": role_results,
        "all_teacher_mechanics_gates_pass": teacher_pass,
        "all_fresh_role_opportunity_gates_pass": opportunity_pass,
        "zero_point_eight_second_output_computed": False,
        "risk_proxy_is_not_hazard_truth": True,
        "fresh_evaluation_geometry_label_outcome_read": True,
        "teacher_corpus_generation_authorized": (
            teacher_pass and opportunity_pass
        ),
        "student_training_authorized": (
            teacher_pass and opportunity_pass
        ),
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _new_output(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((repo_root / "artifacts.local").resolve())
    except ValueError as error:
        raise ValueError("E0.1 opportunity output must stay under artifacts.local") from error
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite report: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    protocol = (repo_root / args.protocol).resolve()
    transport = (repo_root / args.transport).resolve()
    media_root = (repo_root / args.media_root).resolve()
    output = _new_output(repo_root / args.output, repo_root)
    first = audit(protocol, transport, media_root, repo_root)
    second = audit(protocol, transport, media_root, repo_root)
    if json.dumps(first, sort_keys=True) != json.dumps(
        second, sort_keys=True
    ):
        raise ValueError("E0.1 opportunity report is not deterministic")
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
                        "role": item["role"],
                        "trajectory": item["trajectory"],
                        "anchors": item["formal_anchor_count"],
                        "known_fraction_0_4_s": item[
                            "candidate_known_direction_fraction_0_4_s"
                        ],
                        "risk_cells": item[
                            "candidate_risk_proxy_cell_count"
                        ],
                        "risk_anchors": len(
                            item["distinct_risk_proxy_anchors"]
                        ),
                    }
                    for item in first["source_reports"]
                ],
                "deterministic": True,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if first["terminal"].endswith("_SUPPORTED") else 3


if __name__ == "__main__":
    raise SystemExit(main())
