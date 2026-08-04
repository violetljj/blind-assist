"""Independently recompute the Bonn RGB-D R1 result from its frame ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


DIRECTIONS = {
    "RELATIVELY_OPEN_LEFT": "left",
    "RELATIVELY_OPEN_CENTER": "center",
    "RELATIVELY_OPEN_RIGHT": "right",
}
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "docs/research/hftf/"
    "SCALE_FREE_TRAVERSABILITY_R1_BONN_RGBD_PROTOCOL_2026-08-04.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def ratio(numerator: int, denominator: int) -> float:
    return numerator / max(1, denominator)


def recompute(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sequence_id"]), []).append(row)
    summaries = []
    for sequence_id, sequence_rows in grouped.items():
        indices = [int(row["frame_index"]) for row in sequence_rows]
        if indices != list(range(len(sequence_rows))):
            raise ValueError(f"{sequence_id}: frame ledger is not contiguous")
        truth_valid = sum(
            row["truth_score"].get("status") == "VALID" for row in sequence_rows
        )
        candidate_valid = sum(
            row["candidate_score"].get("status") == "VALID" for row in sequence_rows
        )
        truth_directional = [
            row
            for row in sequence_rows
            if row["truth_decision"].get("label") in DIRECTIONS
        ]
        recommended = [
            row
            for row in truth_directional
            if row["candidate_decision"].get("label") in DIRECTIONS
        ]
        correct = sum(
            row["candidate_decision"]["label"] == row["truth_decision"]["label"]
            for row in recommended
        )
        opposite = sum(
            {
                DIRECTIONS[row["candidate_decision"]["label"]],
                DIRECTIONS[row["truth_decision"]["label"]],
            }
            == {"left", "right"}
            for row in recommended
        )
        common = [
            row
            for row in sequence_rows
            if row["truth_decision"].get("status") == "VALID"
            and row["candidate_decision"].get("status") == "VALID"
        ]
        exact = sum(
            row["candidate_decision"].get("label")
            == row["truth_decision"].get("label")
            for row in common
        )
        count = len(sequence_rows)
        summaries.append(
            {
                "sequence_id": sequence_id,
                "frame_count": count,
                "truth_score_valid_count": truth_valid,
                "truth_score_coverage": ratio(truth_valid, count),
                "candidate_score_valid_count": candidate_valid,
                "candidate_execution_coverage": ratio(candidate_valid, count),
                "truth_directional_support": len(truth_directional),
                "candidate_recommendation_count": len(recommended),
                "recommendation_coverage": ratio(len(recommended), len(truth_directional)),
                "directional_correct_count": correct,
                "directional_accuracy": ratio(correct, len(recommended)),
                "opposite_direction_count": opposite,
                "opposite_direction_rate": ratio(opposite, len(recommended)),
                "common_decision_count": len(common),
                "exact_decision_agreement": ratio(exact, len(common)),
            }
        )
        for metadata_key in ("video_id", "role"):
            values = {row.get(metadata_key) for row in sequence_rows}
            if len(values) == 1 and None not in values:
                summaries[-1][metadata_key] = values.pop()
    return summaries


def terminal(
    summaries: list[dict[str, Any]],
    gates: dict[str, Any],
    names: dict[str, str] | None = None,
) -> str:
    names = names or {
        "not_evaluable": "SCALE_FREE_TRAVERSABILITY_R1_NOT_EVALUABLE_SOURCE_SUPPORT",
        "not_supported": "SCALE_FREE_TRAVERSABILITY_R1_EXTERNAL_RGBD_REPLICATION_NOT_SUPPORTED",
        "supported": "SCALE_FREE_TRAVERSABILITY_R1_EXTERNAL_RGBD_REPLICATION_SUPPORTED_DEVELOPMENT_ONLY",
    }
    source_evaluable = all(
        row["truth_score_coverage"]
        >= gates["minimum_truth_score_coverage_each_sequence"]
        and row["truth_directional_support"]
        >= gates["minimum_directional_truth_support_each_sequence"]
        for row in summaries
    )
    if not source_evaluable:
        return names["not_evaluable"]
    accuracy_macro = sum(row["directional_accuracy"] for row in summaries) / len(summaries)
    opposite_macro = sum(row["opposite_direction_rate"] for row in summaries) / len(summaries)
    supported = all(
        row["candidate_execution_coverage"]
        >= gates["minimum_candidate_execution_coverage_each_sequence"]
        and row["recommendation_coverage"]
        >= gates["minimum_recommendation_coverage_each_sequence"]
        and row["directional_accuracy"]
        >= gates["minimum_directional_accuracy_worst_sequence"]
        for row in summaries
    ) and accuracy_macro >= gates["minimum_directional_accuracy_macro"] \
        and opposite_macro <= gates["maximum_macro_opposite_direction_rate"]
    return (
        names["supported"]
        if supported
        else names["not_supported"]
    )


def validate(result_path: Path, frames_path: Path, protocol_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures: list[str] = []
    if result.get("frames_jsonl_sha256") != sha256(frames_path):
        failures.append("frames_sha256_mismatch")
    if result.get("protocol_sha256") != sha256(protocol_path):
        failures.append("protocol_sha256_mismatch")
    if result.get("gates") != protocol.get("gates"):
        failures.append("gate_binding_mismatch")
    actual = recompute(rows)
    if result.get("sequence_results") != actual:
        failures.append("sequence_summary_mismatch")
    if result.get("frame_count") != len(rows):
        failures.append("frame_count_mismatch")
    expected_terminal = terminal(actual, protocol["gates"], protocol.get("terminals"))
    if result.get("terminal") != expected_terminal:
        failures.append("terminal_mismatch")
    accuracy_macro = sum(row["directional_accuracy"] for row in actual) / len(actual)
    opposite_macro = sum(row["opposite_direction_rate"] for row in actual) / len(actual)
    if not math.isclose(
        float(result.get("directional_accuracy_sequence_macro", float("nan"))),
        accuracy_macro,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        failures.append("accuracy_macro_mismatch")
    if not math.isclose(
        float(result.get("opposite_direction_rate_sequence_macro", float("nan"))),
        opposite_macro,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        failures.append("opposite_macro_mismatch")
    return {
        "schema": "blindassist_scale_free_traversability_external_rgbd_validation_v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "recomputed_terminal": expected_terminal,
        "frame_count": len(rows),
        "sequence_count": len(actual),
        "result_sha256": sha256(result_path),
        "frames_sha256": sha256(frames_path),
        "protocol_sha256": sha256(protocol_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args.result, args.frames, args.protocol)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
