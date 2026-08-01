#!/usr/bin/env python3
"""Run the fixed HFTF Stage C E0.2 multi-source qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acquire_stage_c_e0_fresh_media as acquire  # noqa: E402
import audit_stage_c_e0_1_teacher_opportunity as e01_opp  # noqa: E402
import audit_stage_c_e0_fresh_media_transport as transport  # noqa: E402
import run_stage_c_d1_causal_future_label_mechanics as d1  # noqa: E402


SCHEMA = "blindassist_hftf_stage_c_e0_2_fixed_batch_qualification"
PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_multi_source_evaluation_qualification_e0_2"
)
PROTOCOL_STATUS = "FROZEN_BEFORE_FIXED_BATCH_RGB_DEPTH_OR_LABEL_OUTCOME"
LOCK_SCHEMA = "blindassist_hftf_stage_c_e0_2_fixed_batch_source_lock"


def _load_json(path: Path) -> dict[str, Any]:
    return acquire._load_json(path)


def _sha256(path: Path) -> str:
    return acquire._sha256(path)


def _validate_lock(
    protocol: dict[str, Any],
    protocol_path: Path,
    lock: dict[str, Any],
) -> None:
    if lock.get("schema") != LOCK_SCHEMA:
        raise ValueError("Unexpected E0.2 source-lock schema")
    if lock.get("terminal") != "E0_2_FIXED_BATCH_SOURCE_LOCK_VALIDATED":
        raise ValueError("E0.2 fixed batch is not source-locked")
    if lock.get("protocol_sha256") != _sha256(protocol_path):
        raise ValueError("E0.2 source-lock protocol mismatch")
    if (
        lock.get("frozen_evaluation_sources")
        != protocol["frozen_evaluation_sources"]
    ):
        raise ValueError("E0.2 source-lock cohort mismatch")
    if lock.get("rgb_or_depth_read") or lock.get(
        "geometry_label_outcome_read"
    ):
        raise ValueError("E0.2 source lock unexpectedly read outcome")
    if not lock.get("exact_fixed_batch_acquisition_authorized"):
        raise ValueError("E0.2 exact acquisition is not authorized")


def _compatible_acquisition_protocol(
    protocol: dict[str, Any],
    protocol_path: Path,
) -> dict[str, Any]:
    e01 = _load_json(
        protocol_path.parent
        / protocol["parent_bindings"]["e0_1_protocol_path"]
    )
    e0 = _load_json(
        protocol_path.parent
        / e01["parent_bindings"]["e0_protocol_path"]
    )
    selection = protocol["fixed_batch_selection"]
    return {
        "dataset_binding": {
            "dataset_repo": selection["dataset_repo"],
            "dataset_revision": selection["dataset_revision"],
            "metadata_files": e0["dataset_binding"]["metadata_files"],
        },
        "frozen_sources": protocol["frozen_evaluation_sources"],
    }


def _acquire(
    protocol: dict[str, Any],
    protocol_path: Path,
    media_root: Path,
) -> dict[str, Any]:
    compatible = _compatible_acquisition_protocol(
        protocol, protocol_path
    )
    patterns = acquire._allow_patterns(compatible)
    snapshot_download(
        repo_id=compatible["dataset_binding"]["dataset_repo"],
        repo_type="dataset",
        revision=compatible["dataset_binding"]["dataset_revision"],
        allow_patterns=patterns,
        local_dir=media_root,
    )
    files = acquire._validate_download(compatible, media_root)
    return {
        "allow_patterns": patterns,
        "downloaded_file_count": len(files),
        "downloaded_files": files,
        "fixed_batch_media_total_bytes": sum(
            item["size_bytes"]
            for item in files
            if item["kind"] in {"pose", "rgb", "depth"}
        ),
        "new_sources_burned": True,
    }


def _transport(
    protocol: dict[str, Any], media_root: Path
) -> tuple[list[dict[str, Any]], bool]:
    meta = _load_json(media_root / "meta/info.json")
    fps = float(meta["fps"])
    reports = [
        transport._audit_source(source, media_root, fps)
        for source in protocol["frozen_evaluation_sources"]
    ]
    return reports, fps == 5.0 and all(
        item["transport_pass"] for item in reports
    )


def _teacher_configs(
    protocol: dict[str, Any],
    protocol_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    e01_path = (
        protocol_path.parent
        / protocol["parent_bindings"]["e0_1_protocol_path"]
    ).resolve()
    e01 = _load_json(e01_path)
    e0_path = (
        e01_path.parent / e01["parent_bindings"]["e0_protocol_path"]
    ).resolve()
    e0 = _load_json(e0_path)
    d1_path = (
        e0_path.parent
        / e0["parent_bindings"]["d1_protocol_path"]
    ).resolve()
    d1_protocol = _load_json(d1_path)
    _, d0_protocol = d1._validate_protocol(
        d1_protocol, d1_path, repo_root
    )
    return d1_protocol, d0_protocol


def _teacher(
    protocol: dict[str, Any],
    protocol_path: Path,
    media_root: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    d1_protocol, d0_protocol = _teacher_configs(
        protocol, protocol_path, repo_root
    )
    camera = _load_json(media_root / "meta/camera_rgb.json")
    heights = _load_json(media_root / "meta/heights.json")
    return [
        e01_opp._run_source(
            source,
            media_root,
            camera,
            heights,
            d1_protocol,
            d0_protocol,
        )
        for source in protocol["frozen_evaluation_sources"]
    ]


def _role_metrics(
    sources: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for role in ("dev", "heldout"):
        selected = [item for item in sources if item["role"] == role]
        risk_anchors = {
            (item["trajectory"], int(anchor))
            for item in selected
            for anchor in item["distinct_risk_proxy_anchors"]
        }
        risk_sources = {
            item["trajectory"]
            for item in selected
            if item["candidate_risk_proxy_cell_count"] > 0
        }
        directions = {
            float(direction)
            for item in selected
            for direction in item["distinct_risk_proxy_directions"]
        }
        output[role] = {
            "source_count": len(selected),
            "risk_proxy_cell_count": sum(
                item["candidate_risk_proxy_cell_count"]
                for item in selected
            ),
            "distinct_risk_proxy_anchor_count": len(risk_anchors),
            "distinct_sources_with_risk_proxy": len(risk_sources),
            "distinct_risk_proxy_direction_count": len(directions),
            "distinct_risk_proxy_directions": sorted(directions),
            "known_no_risk_cell_count": sum(
                item["candidate_known_no_risk_count"]
                for item in selected
            ),
        }
    return output


def _role_gate(
    metrics: dict[str, dict[str, Any]],
    gate: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
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
        ("known_no_risk_cell_count", "minimum_known_no_risk_cells"),
    )
    for role in ("dev", "heldout"):
        failures = [
            metric
            for metric, threshold in pairs
            if metrics[role][metric] < int(gate[role][threshold])
        ]
        results.append(
            {
                "role": role,
                "passed": not failures,
                "failures": failures,
            }
        )
    return results


def run(
    protocol_path: Path,
    source_lock_path: Path,
    media_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    protocol = _load_json(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("Stage C E0.2 protocol is not frozen")
    lock = _load_json(source_lock_path)
    _validate_lock(protocol, protocol_path, lock)
    acquisition = _acquire(protocol, protocol_path, media_root)
    transport_reports, transport_pass = _transport(
        protocol, media_root
    )
    if not transport_pass:
        return {
            "schema": SCHEMA,
            "terminal": protocol["ordered_gates"]["transport_each_source"][
                "failure_terminal"
            ],
            "protocol_path": str(protocol_path),
            "protocol_sha256": _sha256(protocol_path),
            "source_lock_path": str(source_lock_path),
            "source_lock_sha256": _sha256(source_lock_path),
            "media_root": str(media_root),
            "acquisition": acquisition,
            "transport_reports": transport_reports,
            "all_transport_gates_pass": False,
            "geometry_label_outcome_read": False,
            "student_training_authorized": False,
            "research_mainline_changed": False,
            "default_app_changed": False,
        }
    sources = _teacher(
        protocol, protocol_path, media_root, repo_root
    )
    source_gate = protocol["ordered_gates"][
        "teacher_mechanics_each_source"
    ]
    source_results = [
        e01_opp._source_gate(source, source_gate)
        for source in sources
    ]
    role_metrics = _role_metrics(sources)
    role_results = _role_gate(
        role_metrics, protocol["ordered_gates"]["role_opportunity"]
    )
    teacher_pass = all(item["passed"] for item in source_results)
    opportunity_pass = all(item["passed"] for item in role_results)
    if not teacher_pass:
        terminal = source_gate["failure_terminal"]
    elif not opportunity_pass:
        terminal = protocol["ordered_gates"]["role_opportunity"][
            "failure_terminal"
        ]
    else:
        terminal = protocol["ordered_gates"]["role_opportunity"][
            "success_terminal"
        ]
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "source_lock_path": str(source_lock_path),
        "source_lock_sha256": _sha256(source_lock_path),
        "media_root": str(media_root),
        "acquisition": acquisition,
        "transport_reports": transport_reports,
        "all_transport_gates_pass": True,
        "source_reports": sources,
        "source_teacher_gate_results": source_results,
        "role_opportunity_metrics": role_metrics,
        "role_opportunity_gate_results": role_results,
        "all_teacher_mechanics_gates_pass": teacher_pass,
        "all_role_opportunity_gates_pass": opportunity_pass,
        "zero_point_eight_second_output_computed": False,
        "geometry_label_outcome_read": True,
        "risk_proxy_is_not_hazard_truth": True,
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
        raise ValueError("E0.2 output must stay under artifacts.local") from error
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite report: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    protocol = (repo_root / args.protocol).resolve()
    source_lock = (repo_root / args.source_lock).resolve()
    media_root = (repo_root / args.media_root).resolve()
    output = _new_output(repo_root / args.output, repo_root)
    first = run(
        protocol, source_lock, media_root, repo_root
    )
    if first.get("all_transport_gates_pass"):
        second_sources = _teacher(
            _load_json(protocol), protocol, media_root, repo_root
        )
        deterministic = (
            json.dumps(
                first["source_reports"],
                sort_keys=True,
                separators=(",", ":"),
            )
            == json.dumps(
                second_sources,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        deterministic = True
    if not deterministic:
        raise ValueError("E0.2 teacher output is not deterministic")
    first["determinism_check"] = {
        "second_teacher_payload_byte_exact": True
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
                "transport_pass": first["all_transport_gates_pass"],
                "teacher_pass": first.get(
                    "all_teacher_mechanics_gates_pass", False
                ),
                "opportunity_pass": first.get(
                    "all_role_opportunity_gates_pass", False
                ),
                "role_metrics": first.get(
                    "role_opportunity_metrics", {}
                ),
                "sources": [
                    {
                        "role": item["role"],
                        "trajectory": item["trajectory"],
                        "known_fraction_0_4_s": item[
                            "candidate_known_direction_fraction_0_4_s"
                        ],
                        "risk_cells": item[
                            "candidate_risk_proxy_cell_count"
                        ],
                    }
                    for item in first.get("source_reports", [])
                ],
                "deterministic": True,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0 if first["terminal"].endswith("_QUALIFIED") else 3


if __name__ == "__main__":
    raise SystemExit(main())
