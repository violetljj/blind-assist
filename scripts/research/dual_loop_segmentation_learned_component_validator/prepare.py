"""Prepare the frozen causal component table for learned validator R0."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .core import (
    PREPARE_SCHEMA_VERSION,
    PROTOCOL_ID,
    atomic_output_directory,
    build_component_table,
    load_bound_inputs,
    read_json,
    sha256_file,
    validate_static_config,
    validate_table_contract,
    verify_output_scope,
    write_json,
    write_jsonl,
)


def current_git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_prepare(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path,
    preflight_only: bool,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    verify_output_scope(repo_root, output_root)
    config = read_json(config_path)
    validate_static_config(config)
    inputs = load_bound_inputs(repo_root, config)
    table_rows = build_component_table(config=config, inputs=inputs)
    table_summary = validate_table_contract(config, table_rows)
    summary = {
        "schema_version": PREPARE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "PREFLIGHT_VALID" if preflight_only else "COMPLETE",
        "preflight_only": preflight_only,
        "stage": config["stage"],
        "evidence_instance": config["evidence_instance"],
        "claim_ceiling": config["claim_ceiling"],
        "git_head": current_git_head(repo_root),
        "config": {
            "path": str(config_path.relative_to(repo_root)).replace("\\", "/"),
            "sha256": sha256_file(config_path),
        },
        "input": inputs.provenance,
        "table": table_summary,
        "feature_names": config["feature_contract"]["feature_names"],
        "truth_firewall": {
            "feature_namespace_exact_allowlist": True,
            "target_namespace_separate": True,
            "diagnostic_namespace_not_model_input": True,
            "future_fields_used_as_features": False,
            "truth_fields_used_as_features": False,
            "session_scene_role_used_as_features": False,
            "entropy_status": config["feature_contract"]["entropy_status"],
            "yolo_overlap_status": config["feature_contract"][
                "yolo_overlap_status"
            ],
            "same_class_yolo_status": config["feature_contract"][
                "same_class_yolo_status"
            ],
        },
        "grouping": {
            "outer_method": config["grouped_evaluation"]["outer_method"],
            "session_count": len(
                config["input_contract"]["expected_session_frame_counts"]
            ),
            "participant_route_parent_capture_independence": (
                "NOT_EVALUABLE_MISSING_IDENTIFIERS"
            ),
            "claim": config["grouped_evaluation"]["claim"],
        },
        "fresh_holdout_accessed": False,
        "model_fit_executed": False,
        "threshold_selection_executed": False,
    }
    if preflight_only:
        return summary
    temporary, finalize = atomic_output_directory(output_root)
    try:
        table_path = temporary / "component_table.jsonl"
        write_jsonl(table_path, table_rows)
        summary["output_files"] = {
            "component_table.jsonl": {
                "sha256": sha256_file(table_path),
                "row_count": len(table_rows),
            }
        }
        write_json(temporary / "prepare_receipt.json", summary)
        finalize(True)
    except BaseException:
        finalize(False)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_prepare(
        repo_root=args.repo_root,
        config_path=args.config,
        output_root=args.output_root,
        preflight_only=args.preflight_only,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "preflight_only": result["preflight_only"],
                "row_count": result["table"]["row_count"],
                "feature_count": result["table"]["feature_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
