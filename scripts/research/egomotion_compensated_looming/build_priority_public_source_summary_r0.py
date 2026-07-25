#!/usr/bin/env python3
"""Build a non-terminal summary of the three priority public-source audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = (
    ("ARIA_DIGITAL_TWIN", "adt_geometry_cell_prescreen_terminal_r0.json", "ADT_CELL_PRESCREEN_INSUFFICIENT"),
    ("ARGOVERSE_2_SENSOR", "av2_join_and_cell_mechanics_terminal_r0.json", "AV2_REQUIRED_PURE_ROTATION_CELL_STRUCTURALLY_ABSENT"),
    ("UT_CODA", "coda_tiny_continuity_and_binding_terminal_r0.json", "HOLD_CODA_BOUNDED_PRESCREEN"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_summary(audit_dir: Path) -> dict:
    sources = []
    for source_id, filename, expected_terminal in EXPECTED:
        path = audit_dir / filename
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt["source_id"] != source_id:
            raise AssertionError(f"source drift: {filename}")
        if receipt["terminal"] != expected_terminal or receipt["status"] != "VALID":
            raise AssertionError(f"terminal drift: {filename}")
        sources.append(
            {
                "source_id": source_id,
                "receipt": filename,
                "sha256": _sha256(path),
                "source_boundary_terminal": expected_terminal,
                "source_boundary_status": "VALID",
            }
        )
    return {
        "schema_version": "egomotion_compensated_looming_priority_public_source_summary_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0",
        "artifact_role": "NON_TERMINAL_SOURCE_AUDIT_BOUNDARY_SUMMARY",
        "parent_r0_execution_status": "NOT_EXECUTED",
        "boundary_summary_code": (
            "PRIORITY_PUBLIC_SOURCE_PATHS_INSUFFICIENT_"
            "NEW_HEAD_MOUNTED_SOURCE_OR_CONTROLLED_CAPTURE_REQUIRED"
        ),
        "source_boundary_receipts": sources,
        "admitted_real_source_count": 0,
        "candidate_rgb_payload_decoded": False,
        "candidate_signal_computed": False,
        "role_split_frozen": False,
        "authority": {
            "may_treat_summary_as_parent_r0_terminal": False,
            "may_run_signal_comparison": False,
            "may_audit_new_head_mounted_source_metadata": True,
            "may_preregister_new_controlled_capture": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.audit_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
