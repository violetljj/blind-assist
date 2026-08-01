#!/usr/bin/env python3
"""Aggregate the frozen D3-Q0 selector/failure receipt prefix only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage_c_d3_q0_common import (
    AGGREGATE_ATTEMPT_SCHEMA,
    AGGREGATE_ATTEMPT_STATUS,
    BUDGET_TERMINAL,
    QUALIFICATION_TERMINAL,
    SELECTION_SCHEMA,
    aggregate_paths,
    load_json,
    preserve_temporary_artifact,
    scan_screening_state,
    sha256,
    validate_aggregate_attempt,
    validate_execution_contract,
    write_json_exclusive_fsync,
)


IMPLEMENTATION_KEY = "screening_aggregator"
EXHAUSTED_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_1_screening_budget_exhausted"
)
INVALID_SCHEMA = "blindassist_hftf_stage_c_d3_q0_1_screening_invalid"


def _selected_sources(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "slot_index": row["slot_index"],
            "session_id": row["session_id"],
            "selector_path": row["selector_path"],
            "selector_sha256": row["selector_sha256"],
            "source_authority_and_content_hashes": row[
                "source_authority_and_content_hashes"
            ],
        }
        for row in state["qualified_rows"]
    ]


def _terminal_payload(
    context: dict[str, Any],
    state: dict[str, Any],
    aggregate_attempt_sha256: str,
) -> dict[str, Any]:
    terminal = state["terminal"]
    if terminal not in {QUALIFICATION_TERMINAL, BUDGET_TERMINAL}:
        raise ValueError("D3-Q0 screening prefix has no aggregate terminal")
    selected = _selected_sources(state)
    if (
        terminal == QUALIFICATION_TERMINAL
        and len(selected) != 6
    ) or (
        terminal == BUDGET_TERMINAL
        and (
            state["consumed_count"] != 40
            or state["newly_opened_count"] != 39
            or len(selected) >= 6
        )
    ):
        raise ValueError("D3-Q0 aggregate terminal state is inconsistent")
    return {
        "schema": (
            SELECTION_SCHEMA
            if terminal == QUALIFICATION_TERMINAL
            else EXHAUSTED_SCHEMA
        ),
        "terminal": terminal,
        "workflow_profile": "THESIS_DEVELOPMENT",
        "execution_contract_sha256": context["contract_sha256"],
        "metadata_roster_sha256": context["roster_sha256"],
        "aggregate_attempt_sha256": aggregate_attempt_sha256,
        "consumed_slot_count": state["consumed_count"],
        "newly_opened_slot_count": state["newly_opened_count"],
        "carry_forward_burned_slot_count": len(
            state["carry_forward_rows"]
        ),
        "carry_forward_burn_receipt": state["carry_forward_rows"][0],
        "qualified_source_count": len(selected),
        "selected_sources": selected,
        "failure_receipt_count": len(state["failure_rows"]),
        "screening_receipts_only_read": True,
        "sealed_payload_read": False,
        "source_replacement_authorized": False,
        "budget_expansion_authorized": False,
        "screening_rerun_authorized": False,
    }


def _freeze_invalid(
    context: dict[str, Any],
    paths: dict[str, Path],
    error: BaseException,
) -> None:
    root = Path(context["root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if paths["invalid"].exists():
        return
    preserve_temporary_artifact(paths["invalid"])
    write_json_exclusive_fsync(
        paths["invalid"],
        {
            "schema": INVALID_SCHEMA,
            "terminal": "D3_QUALIFICATION_INVALID_STOP",
            "workflow_profile": "THESIS_DEVELOPMENT",
            "execution_contract_sha256": context["contract_sha256"],
            "metadata_roster_sha256": context["roster_sha256"],
            "aggregate_attempt_sha256": (
                sha256(paths["aggregate_attempt"])
                if paths["aggregate_attempt"].is_file()
                else None
            ),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "partial_artifacts_preserved": True,
            "sealed_payload_read": False,
            "rerun_authorized": False,
        },
    )


def aggregate_screening(
    context: dict[str, Any],
) -> dict[str, Any]:
    root = Path(context["root"]).resolve()
    paths = aggregate_paths(root)
    existing = [
        label
        for label in ("selection", "exhausted", "invalid")
        if paths[label].exists()
    ]
    if existing:
        raise FileExistsError(
            "D3-Q0 aggregate artifact already exists: "
            + ",".join(existing)
        )
    if paths["aggregate_attempt"].exists():
        validate_aggregate_attempt(
            load_json(paths["aggregate_attempt"]),
            context["contract_sha256"],
            context["roster_sha256"],
        )
        error = ValueError(
            "prior durable aggregate attempt is incomplete; "
            "screening frozen invalid without rereading receipts"
        )
        _freeze_invalid(context, paths, error)
        raise error
    aggregate_temporary = paths["aggregate_attempt"].with_name(
        paths["aggregate_attempt"].name + ".tmp"
    )
    aggregate_orphan = paths["aggregate_attempt"].with_name(
        paths["aggregate_attempt"].name + ".orphan"
    )
    if aggregate_temporary.exists() or aggregate_orphan.exists():
        preserve_temporary_artifact(paths["aggregate_attempt"])
        error = ValueError(
            "incomplete aggregate-attempt temporary/orphan artifact; "
            "screening frozen invalid without reading receipts"
        )
        _freeze_invalid(context, paths, error)
        raise error
    root.mkdir(parents=True, exist_ok=True)
    write_json_exclusive_fsync(
        paths["aggregate_attempt"],
        {
            "schema": AGGREGATE_ATTEMPT_SCHEMA,
            "status": AGGREGATE_ATTEMPT_STATUS,
            "workflow_profile": "THESIS_DEVELOPMENT",
            "execution_contract_sha256": context["contract_sha256"],
            "metadata_roster_sha256": context["roster_sha256"],
            "selector_or_failure_receipts_read_before_attempt": False,
            "sealed_payload_read": False,
            "rerun_authorized": False,
        },
    )
    try:
        state = scan_screening_state(
            root,
            context["slots"],
            context["contract_sha256"],
            context["roster_sha256"],
            context["carry_forward_authority"],
        )
        if state["terminal"] is None:
            if state["interrupted_slot"] is not None:
                raise ValueError(
                    "D3-Q0 current prefix ends in an interrupted slot"
                )
            raise ValueError(
                "D3-Q0 screening is not terminal yet"
            )
        payload = _terminal_payload(
            context,
            state,
            sha256(paths["aggregate_attempt"]),
        )
        destination = (
            paths["selection"]
            if payload["terminal"] == QUALIFICATION_TERMINAL
            else paths["exhausted"]
        )
        write_json_exclusive_fsync(destination, payload)
        return payload
    except (KeyError, OSError, TypeError, ValueError) as error:
        _freeze_invalid(context, paths, error)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    try:
        context = validate_execution_contract(
            args.contract,
            IMPLEMENTATION_KEY,
            Path(__file__),
            verify_git=True,
        )
        result = aggregate_screening(context)
        print(json.dumps({"terminal": result["terminal"]}))
        return 0 if result["terminal"] == QUALIFICATION_TERMINAL else 2
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
