#!/usr/bin/env python3
"""Validate the fail-closed RCLE RGB Segment Confirmation R1 terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require(condition: bool, label: str, checks: dict[str, bool]) -> None:
    checks[label] = bool(condition)
    if not condition:
        raise ValueError(label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--result-doc", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    terminal_path = args.terminal.resolve()
    result_doc_path = args.result_doc.resolve()
    terminal = read_json(terminal_path)
    checks: dict[str, bool] = {}

    require(
        terminal.get("protocol_id") == "RCLE_RGB_SEGMENT_CONFIRMATION_R1",
        "protocol_id",
        checks,
    )
    require(
        terminal.get("decision") == "RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE",
        "terminal_decision",
        checks,
    )
    require(
        terminal.get("validity") == "VALID_FAIL_CLOSED_TERMINAL",
        "terminal_validity",
        checks,
    )

    bindings = terminal.get("bindings", {})
    for name, binding in bindings.items():
        bound_path = (repo / binding["path"]).resolve()
        require(bound_path.is_file(), f"binding_{name}_exists", checks)
        require(
            sha256_file(bound_path) == binding["sha256"],
            f"binding_{name}_sha256",
            checks,
        )

    attempts = terminal.get("identity_attempts", [])
    require(len(attempts) == 2, "two_identity_attempts", checks)
    by_source = {item["source_family_id"]: item for item in attempts}
    openloris = by_source.get("OPENLORIS_CORRIDOR", {})
    dlr = by_source.get("DLR_RGBD_VICON", {})
    require(
        openloris.get("segment_id") == "corridor1-1:w004"
        and openloris.get("role") == "positive"
        and openloris.get("terminal")
        == "INVALID_IDENTITY_EXTRACTION_CLOSE_ATTEMPT",
        "openloris_exact_terminal",
        checks,
    )
    require(
        dlr.get("segment_id") == "extreme_geometry/hexagon_01:w001"
        and dlr.get("role") == "below"
        and dlr.get("terminal") == "SEGMENT_IDENTITY_NOT_EVALUABLE",
        "dlr_exact_terminal",
        checks,
    )
    require(
        all(
            item.get("claim_state") == "CONSUMED"
            and item.get("retry_state") == "FORBIDDEN"
            and item.get("rgb_identity_closed") is False
            and item.get("selected_rgb_frames") == 0
            and item.get("payload_files") == 0
            and item.get("pixel_decode_calls") == 0
            and item.get("rgb_algorithm_calls") == 0
            for item in attempts
        ),
        "attempt_fail_closed_counts",
        checks,
    )

    frame_rows = read_jsonl(repo / bindings["frame_ledger"]["path"])
    abstention_rows = read_jsonl(repo / bindings["abstention_ledger"]["path"])
    metrics = read_json(repo / bindings["alignment_metrics"]["path"])
    require(len(frame_rows) == 2, "frame_ledger_two_terminal_rows", checks)
    require(
        all(
            row.get("record_type") == "SEGMENT_ZERO_FRAME_TERMINAL"
            and row.get("eligible_rgb_frame_count") == 0
            and row.get("frame_rows_emitted") == 0
            for row in frame_rows
        ),
        "frame_ledger_zero_frame_semantics",
        checks,
    )
    require(len(abstention_rows) == 3, "three_abstention_rows", checks)
    require(
        metrics.get("status") == "NOT_EVALUABLE"
        and all(
            segment.get("pair_denominator") == 0
            and segment.get("aligned_pair_count") == 0
            and segment.get("pair_coverage") is None
            and segment.get("maximum_timestamp_delta_seconds") is None
            and segment.get("median_timestamp_delta_seconds") is None
            and segment.get("rgb_geometry_consistency") is None
            for segment in metrics.get("segments", [])
        ),
        "alignment_null_zero_denominator_semantics",
        checks,
    )
    require(
        metrics.get("cross_segment", {}).get("source_role_confounding") is True
        and metrics.get("cross_segment", {}).get("comparison_performed") is False,
        "source_role_confounding_preserved",
        checks,
    )
    require(
        metrics.get("execution_counts")
        == {
            "eligible_rgb_frames": 0,
            "frame_rows": 0,
            "aligned_pairs": 0,
            "pixel_decode_calls": 0,
            "rgb_algorithm_calls": 0,
        },
        "zero_execution_counts",
        checks,
    )

    rgb_execution = terminal.get("rgb_execution", {})
    require(
        rgb_execution.get("final_rgb_identity_and_synchronization_lock_created")
        is False
        and rgb_execution.get("independent_rgb_execution_signature_created")
        is False
        and rgb_execution.get("rgb_execution_authority") is False
        and rgb_execution.get("algorithm_run_performed") is False,
        "rgb_execution_closed",
        checks,
    )
    ceiling = terminal.get("claim_ceiling", {})
    require(
        ceiling.get("source_role_confounding")
        == "OpenLORIS-positive and DLR-below remain source-role confounded."
        and all(
            ceiling.get(field) is False
            for field in (
                "mechanism_evidence_granted",
                "positive_below_discrimination_granted",
                "performance_or_generalization_granted",
                "host_replay_authority",
                "android_authority",
                "product_authority",
                "safety_authority",
            )
        ),
        "claim_ceiling_closed",
        checks,
    )
    require(
        terminal.get("mvsec", {}).get("supplemental_control_accessed") is False
        and terminal.get("mvsec", {}).get("exact_rgb_capture_identity_confirmed")
        is False,
        "mvsec_not_accessed",
        checks,
    )

    terminal_sha256 = sha256_file(terminal_path)
    result_text = result_doc_path.read_text(encoding="utf-8")
    require(
        terminal_sha256 in result_text
        and "RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE"
        in result_text
        and "VALID_FAIL_CLOSED_TERMINAL" in result_text,
        "result_doc_binds_terminal",
        checks,
    )

    output = {
        "schema_version": "rcle.rgb_segment_confirmation.protocol_terminal_validation.v1",
        "protocol_id": "RCLE_RGB_SEGMENT_CONFIRMATION_R1",
        "decision": "PASS",
        "terminal": {
            "path": terminal_path.relative_to(repo).as_posix(),
            "sha256": terminal_sha256,
        },
        "result_doc": {
            "path": result_doc_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(result_doc_path),
        },
        "checks": checks,
        "check_count": len(checks),
        "errors": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": "PASS", "check_count": len(checks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
