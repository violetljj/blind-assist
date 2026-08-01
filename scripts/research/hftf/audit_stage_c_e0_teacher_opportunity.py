#!/usr/bin/env python3
"""Audit fresh HFTF Stage C E0 teacher mechanics and role opportunity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage_c_d0_semantic_independent_label_readiness as d0  # noqa: E402
import run_stage_c_d1_causal_future_label_mechanics as d1  # noqa: E402


SCHEMA = "blindassist_hftf_stage_c_e0_teacher_opportunity"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_fresh_foot_ground_student_canary_e0"
)
PROTOCOL_STATUS = (
    "FROZEN_BEFORE_FRESH_RGB_DEPTH_OR_GEOMETRY_LABEL_OUTCOME"
)
TRANSPORT_SCHEMA = "blindassist_hftf_stage_c_e0_fresh_media_transport"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_inputs(
    protocol: dict[str, Any],
    protocol_path: Path,
    transport: dict[str, Any],
    transport_path: Path,
    media_root: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("Stage C E0 protocol is not frozen")
    if (
        transport.get("schema") != TRANSPORT_SCHEMA
        or transport.get("terminal")
        != "E0_FRESH_MEDIA_TRANSPORT_SUPPORTED"
        or not transport.get("all_transport_gates_pass")
    ):
        raise ValueError("E0 fresh transport is not supported")
    if transport.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("E0 transport protocol binding mismatch")
    if Path(str(transport["media_root"])).resolve() != media_root:
        raise ValueError("E0 transport media-root mismatch")
    if not transport.get(
        "teacher_mechanics_and_label_opportunity_audit_authorized"
    ):
        raise ValueError("E0 teacher opportunity audit is not authorized")
    if transport.get("fresh_geometry_label_outcome_read"):
        raise ValueError("Transport unexpectedly read geometry outcome")

    parents = protocol["parent_bindings"]
    d1_path = (
        protocol_path.parent / parents["d1_protocol_path"]
    ).resolve()
    if _sha256(d1_path) != parents["d1_protocol_sha256"]:
        raise ValueError("E0 D1 protocol binding mismatch")
    d1_runner = (
        protocol_path.parent / parents["d1_runner_path"]
    ).resolve()
    if _sha256(d1_runner) != parents["d1_runner_sha256"]:
        raise ValueError("E0 D1 runner binding mismatch")
    d0_runner = (
        protocol_path.parent / parents["d0_runner_path"]
    ).resolve()
    if _sha256(d0_runner) != parents["d0_runner_sha256"]:
        raise ValueError("E0 D0 runner binding mismatch")
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
    trajectory = str(source["trajectory"])
    pose_path = media_root / source["files"]["pose"]["path"]
    depth_path = media_root / source["files"]["depth"]["path"]
    frame = pl.read_parquet(pose_path).sort("frame")
    x = frame["cart_x"].to_numpy()
    y = frame["cart_y"].to_numpy()
    yaw = d1._yaw_from_frame(frame)
    anchors = d1._formal_anchors(frame.height)
    needed = {
        index
        for anchor in anchors
        for index in (anchor, anchor + 2, anchor + 4)
    }
    decoded = d0._decode_selected_depth(depth_path, needed)
    if set(decoded) != needed:
        raise ValueError(f"{trajectory}: incomplete teacher depth decode")
    profiles: dict[int, list[dict[str, Any]]] = {}
    plane_known = 0
    for frame_index in sorted(needed):
        points = d0._project_depth(
            decoded[frame_index],
            camera,
            d0_protocol["depth_projection"],
        )
        plane = d0._fit_ground_plane(
            points,
            float(heights[trajectory]),
            trajectory,
            frame_index,
            d0_protocol["ground_plane_reader"],
        )
        plane_known += plane is not None
        profiles[frame_index] = d0._surface_profiles(
            points,
            plane,
            d0_protocol["surface_profile_reader"],
        )

    mapping_gate = d1_protocol["ordered_gates"][
        "odometry_mapping_each_source_each_horizon"
    ]
    mapping = {
        "0.4": d1._odometry_mapping(
            x,
            y,
            yaw,
            2,
            float(mapping_gate["moving_distance_threshold_m"]),
        ),
        "0.8": d1._odometry_mapping(
            x,
            y,
            yaw,
            4,
            float(mapping_gate["moving_distance_threshold_m"]),
        ),
    }
    anchor_contract = d1_protocol["formal_anchor_selection"]
    rebin = d1_protocol["reprojection_and_rebin"]
    distances = rebin["target_distance_m"]
    horizon_specs = (("0.4", 2, 0.4), ("0.8", 4, 0.8))
    metrics = {
        name: {
            "eligible_anchor_count": 0,
            "candidate_cell_count": 0,
            "known_count": 0,
            "known_no_risk_count": 0,
            "risk_count": 0,
            "future_added_known_count": 0,
            "known_lost_count": 0,
            "unknown_to_safe_violation_count": 0,
            "risk_anchors": set(),
            "risk_directions": set(),
        }
        for name, _, _ in horizon_specs
    }
    eligible_both = 0
    for anchor in anchors:
        anchor_position = np.array([x[anchor], y[anchor]])
        history_position = np.array([x[anchor - 2], y[anchor - 2]])
        current_observations = d1._profile_observations(
            profiles[anchor],
            anchor_position,
            float(yaw[anchor]),
            distances,
        )
        anchor_eligible = True
        for horizon_name, offset, horizon_s in horizon_specs:
            origin, speed = d1._causal_origin(
                anchor_position,
                history_position,
                float(anchor_contract["history_velocity_interval_s"]),
                horizon_s,
            )
            eligible = speed <= float(
                anchor_contract["maximum_history_speed_mps"]
            )
            anchor_eligible = anchor_eligible and eligible
            if not eligible:
                continue
            future_position = np.array(
                [x[anchor + offset], y[anchor + offset]]
            )
            future_observations = d1._profile_observations(
                profiles[anchor + offset],
                future_position,
                float(yaw[anchor + offset]),
                distances,
            )
            baseline = d1._rebin(
                current_observations,
                origin,
                float(yaw[anchor]),
                rebin,
                d0_protocol["surface_profile_reader"],
            )
            candidate = d1._rebin(
                current_observations + future_observations,
                origin,
                float(yaw[anchor]),
                rebin,
                d0_protocol["surface_profile_reader"],
            )
            comparison = d1._comparison(baseline, candidate)
            item = metrics[horizon_name]
            item["eligible_anchor_count"] += 1
            item["candidate_cell_count"] += len(candidate)
            item["known_count"] += sum(
                cell["state"] != "UNKNOWN" for cell in candidate
            )
            item["known_no_risk_count"] += sum(
                cell["state"]
                == "KNOWN_NO_GROUND_CONTINUITY_RISK_PROXY"
                for cell in candidate
            )
            risk_cells = [
                cell
                for cell in candidate
                if cell["state"]
                == "KNOWN_GROUND_CONTINUITY_RISK_PROXY"
            ]
            item["risk_count"] += len(risk_cells)
            if risk_cells:
                item["risk_anchors"].add(anchor)
                item["risk_directions"].update(
                    float(cell["direction_degrees"])
                    for cell in risk_cells
                )
            item["future_added_known_count"] += comparison[
                "candidate_added_known_direction_count"
            ]
            item["known_lost_count"] += comparison[
                "candidate_lost_known_direction_count"
            ]
            item["unknown_to_safe_violation_count"] += comparison[
                "unknown_to_safe_violation_count"
            ]
        eligible_both += anchor_eligible

    summary: dict[str, Any] = {}
    for name, item in metrics.items():
        cells = int(item["candidate_cell_count"])
        summary[name] = {
            "eligible_anchor_count": int(item["eligible_anchor_count"]),
            "candidate_direction_cell_count": cells,
            "candidate_known_direction_count": int(item["known_count"]),
            "candidate_known_direction_fraction": (
                float(item["known_count"]) / cells if cells else 0.0
            ),
            "candidate_known_no_risk_count": int(
                item["known_no_risk_count"]
            ),
            "candidate_risk_proxy_cell_count": int(item["risk_count"]),
            "future_added_known_direction_count": int(
                item["future_added_known_count"]
            ),
            "candidate_lost_known_direction_count": int(
                item["known_lost_count"]
            ),
            "unknown_to_safe_violation_count": int(
                item["unknown_to_safe_violation_count"]
            ),
            "distinct_risk_proxy_anchors": sorted(
                int(value) for value in item["risk_anchors"]
            ),
            "distinct_risk_proxy_directions": sorted(
                float(value) for value in item["risk_directions"]
            ),
        }
    return {
        "role": source["role"],
        "trajectory": trajectory,
        "formal_anchor_count": len(anchors),
        "teacher_depth_frame_count": len(needed),
        "ground_plane_known_count": int(plane_known),
        "ground_plane_known_fraction": (
            float(plane_known) / len(needed) if needed else 0.0
        ),
        "history_speed_eligible_anchor_count": int(eligible_both),
        "history_speed_eligible_fraction": (
            float(eligible_both) / len(anchors) if anchors else 0.0
        ),
        "odometry_mapping": mapping,
        "summary_by_horizon": summary,
    }


def _source_gate(
    source: dict[str, Any], gates: dict[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    if source["formal_anchor_count"] < int(
        gates["minimum_formal_anchor_count"]
    ):
        failures.append("formal_anchor_count")
    if source["ground_plane_known_fraction"] < float(
        gates["minimum_ground_plane_known_fraction"]
    ):
        failures.append("ground_plane_known_fraction")
    if source["history_speed_eligible_fraction"] < float(
        gates["minimum_history_speed_eligible_fraction"]
    ):
        failures.append("history_speed_eligible_fraction")
    for horizon, metrics in source["summary_by_horizon"].items():
        if metrics["candidate_known_direction_fraction"] < float(
            gates[
                "minimum_candidate_known_direction_fraction_each_future_horizon"
            ]
        ):
            failures.append(f"{horizon}:candidate_known_fraction")
        if metrics["candidate_lost_known_direction_count"] > int(
            gates["maximum_candidate_known_cells_lost_vs_baseline"]
        ):
            failures.append(f"{horizon}:known_lost")
        if metrics["unknown_to_safe_violation_count"] > int(
            gates["maximum_unknown_to_safe_violations"]
        ):
            failures.append(f"{horizon}:unknown_to_safe")
    return {
        "role": source["role"],
        "trajectory": source["trajectory"],
        "passed": not failures,
        "failures": failures,
    }


def _role_metrics(
    sources: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for role in ("train", "dev", "heldout"):
        selected = [item for item in sources if item["role"] == role]
        risk_cells = 0
        no_risk_cells = 0
        risk_anchors: set[tuple[str, int]] = set()
        risk_sources: set[str] = set()
        risk_directions: set[float] = set()
        for source in selected:
            source_has_risk = False
            for metrics in source["summary_by_horizon"].values():
                risk_cells += metrics["candidate_risk_proxy_cell_count"]
                no_risk_cells += metrics[
                    "candidate_known_no_risk_count"
                ]
                if metrics["candidate_risk_proxy_cell_count"]:
                    source_has_risk = True
                risk_anchors.update(
                    (source["trajectory"], int(anchor))
                    for anchor in metrics[
                        "distinct_risk_proxy_anchors"
                    ]
                )
                risk_directions.update(
                    float(direction)
                    for direction in metrics[
                        "distinct_risk_proxy_directions"
                    ]
                )
            if source_has_risk:
                risk_sources.add(source["trajectory"])
        output[role] = {
            "source_count": len(selected),
            "risk_proxy_cell_count": risk_cells,
            "known_no_risk_cell_count": no_risk_cells,
            "distinct_risk_proxy_anchor_count": len(risk_anchors),
            "distinct_sources_with_risk_proxy": len(risk_sources),
            "distinct_risk_proxy_direction_count": len(risk_directions),
            "distinct_risk_proxy_directions": sorted(risk_directions),
        }
    return output


def _role_gate(
    metrics: dict[str, dict[str, Any]],
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role in ("train", "dev", "heldout"):
        item = metrics[role]
        gate = gates[role]
        failures: list[str] = []
        checks = (
            (
                "risk_proxy_cell_count",
                "minimum_risk_proxy_cells",
            ),
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
        for metric, threshold in checks:
            if int(item[metric]) < int(gate[threshold]):
                failures.append(metric)
        if item["known_no_risk_cell_count"] < int(
            gates["minimum_known_no_risk_cells_each_role"]
        ):
            failures.append("known_no_risk_cell_count")
        output.append(
            {
                "role": role,
                "passed": not failures,
                "failures": failures,
            }
        )
    return output


def audit(
    protocol_path: Path,
    transport_path: Path,
    media_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    transport = _load_json(transport_path)
    d1_protocol, d0_protocol = _validate_inputs(
        protocol,
        protocol_path,
        transport,
        transport_path,
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
        for source in protocol["frozen_sources"]
    ]
    ordered = protocol["ordered_execution_gates"]
    source_results = [
        _source_gate(item, ordered["teacher_mechanics_each_source"])
        for item in sources
    ]
    roles = _role_metrics(sources)
    role_results = _role_gate(
        roles, ordered["role_label_opportunity"]
    )
    teacher_pass = all(item["passed"] for item in source_results)
    opportunity_pass = all(item["passed"] for item in role_results)
    if not teacher_pass:
        terminal = ordered["teacher_mechanics_each_source"][
            "failure_terminal"
        ]
    elif not opportunity_pass:
        terminal = ordered["role_label_opportunity"]["failure_terminal"]
    else:
        terminal = "E0_FRESH_TEACHER_AND_ROLE_OPPORTUNITY_SUPPORTED"
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
        "role_opportunity_metrics": roles,
        "role_opportunity_gate_results": role_results,
        "all_teacher_mechanics_gates_pass": teacher_pass,
        "all_role_opportunity_gates_pass": opportunity_pass,
        "risk_proxy_is_not_hazard_truth": True,
        "fresh_geometry_label_outcome_read": True,
        "full_teacher_corpus_persisted": False,
        "teacher_corpus_generation_authorized": (
            teacher_pass and opportunity_pass
        ),
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _require_new_output(path: Path, repo_root: Path) -> Path:
    resolved = path.resolve()
    artifacts = (repo_root / "artifacts.local").resolve()
    try:
        resolved.relative_to(artifacts)
    except ValueError as error:
        raise ValueError("E0 opportunity output must stay under artifacts.local") from error
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
    output = _require_new_output(repo_root / args.output, repo_root)
    first = audit(
        protocol, transport, media_root, repo_root
    )
    second = audit(
        protocol, transport, media_root, repo_root
    )
    deterministic = (
        json.dumps(first, sort_keys=True, separators=(",", ":"))
        == json.dumps(second, sort_keys=True, separators=(",", ":"))
    )
    if not deterministic:
        raise ValueError("E0 teacher-opportunity result is not deterministic")
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
                "teacher_gates_pass": first[
                    "all_teacher_mechanics_gates_pass"
                ],
                "role_opportunity_gates_pass": first[
                    "all_role_opportunity_gates_pass"
                ],
                "roles": first["role_opportunity_metrics"],
                "sources": [
                    {
                        "role": item["role"],
                        "trajectory": item["trajectory"],
                        "anchors": item["formal_anchor_count"],
                        "plane_known_fraction": item[
                            "ground_plane_known_fraction"
                        ],
                        "history_speed_eligible_fraction": item[
                            "history_speed_eligible_fraction"
                        ],
                        "known_fraction": {
                            horizon: metrics[
                                "candidate_known_direction_fraction"
                            ]
                            for horizon, metrics in item[
                                "summary_by_horizon"
                            ].items()
                        },
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
