#!/usr/bin/env python3
"""Lock train-reuse, fresh-dev, and reserved-test sources for HFTF G0."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from build_sanpo_sequence_evalset import (
    GCS_PREFIX,
    fetch_text,
    get_gcs_object,
    media_url,
    object_inventory,
)
from plan_stage_c_f0_1_sanpo_cross_split_inventory import (
    _test_candidate,
    _validate_test_split,
)
from plan_stage_c_f0_sanpo_inventory import _load_json, _sha256


PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_c_signed_clearance_current_bridge_g0"
)
PROTOCOL_STATUS = (
    "FROZEN_AFTER_F0_1_STOP_BEFORE_G0_CLEARANCE_OR_SOURCE_SCAN_OUTCOME"
)
SCHEMA = "blindassist_hftf_stage_c_g0_signed_clearance_source_plan"
READY = "G0_SIGNED_CLEARANCE_SOURCE_PLAN_READY"
NOT_EVALUABLE = "G0_SIGNED_CLEARANCE_SOURCE_PLAN_NOT_EVALUABLE"


def _resolve_parent(
    protocol_path: Path, receipt: dict[str, Any]
) -> Path:
    raw = Path(str(receipt["path"]))
    if raw.parts and raw.parts[0] == "artifacts.local":
        return (protocol_path.parents[3] / raw).resolve()
    return (protocol_path.parent / raw).resolve()


def _parent(
    protocol_path: Path,
    protocol: dict[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    receipt = protocol["parents"][key]
    path = _resolve_parent(protocol_path, receipt)
    if _sha256(path) != str(receipt["sha256"]):
        raise ValueError(f"G0 source-plan parent hash mismatch: {key}")
    return path, _load_json(path)


def _burned_ids(ledger_path: Path, ledger: dict[str, Any]) -> set[str]:
    parent_path = (
        ledger_path.parent
        / str(ledger["parent_r4_burn_ledger"]["path"])
    ).resolve()
    parent = _load_json(parent_path)
    if _sha256(parent_path) != str(
        ledger["parent_r4_burn_ledger"]["sha256"]
    ):
        raise ValueError("G0 parent burn-ledger hash mismatch")
    values = [
        *(str(item) for item in parent["burned_session_ids"]),
        *(
            str(item)
            for item in ledger["additional_r4_outcome_open_session_ids"]
        ),
    ]
    if len(values) != int(ledger["effective_burned_session_count"]):
        raise ValueError("G0 effective historical burn count mismatch")
    if len(values) != len(set(values)):
        raise ValueError("G0 historical burn union contains duplicates")
    return set(values)


def _source_ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["session_id"]) for record in records]


def _require_source_order(
    label: str,
    expected: list[str],
    records: list[dict[str, Any]],
) -> None:
    if _source_ids(records) != expected:
        raise ValueError(f"G0 {label} source order mismatch")


def _validate_outcome_and_freshness_chain(
    f0_plan: dict[str, Any],
    f0_1_plan: dict[str, Any],
    source_lock: dict[str, Any],
    acquisition: dict[str, Any],
    authority_cohort: dict[str, Any],
    teacher_opportunity: dict[str, Any],
    result: dict[str, Any],
    historical_burned: set[str],
) -> tuple[list[str], list[str]]:
    f0_candidates = f0_plan["inventory_candidates"]
    f0_1_sources = f0_1_plan["sources"]
    f0_1_ids = _source_ids(f0_1_sources)
    _require_source_order("source-lock", f0_1_ids, source_lock["sources"])
    _require_source_order("acquisition", f0_1_ids, acquisition["sources"])
    _require_source_order(
        "authority-cohort", f0_1_ids, authority_cohort["sources"]
    )
    _require_source_order(
        "teacher-opportunity",
        f0_1_ids,
        teacher_opportunity["source_results"],
    )
    if _source_ids(f0_candidates[:9]) != f0_1_ids[:9]:
        raise ValueError("G0 F0/F0.1 official-train reuse drifted")
    expected_roles = ["train"] * 6 + ["dev"] * 3 + ["heldout"] * 3
    expected_splits = ["train"] * 9 + ["test"] * 3
    if (
        [str(item.get("role")) for item in f0_1_sources]
        != expected_roles
        or [str(item.get("official_split")) for item in f0_1_sources]
        != expected_splits
    ):
        raise ValueError("G0 F0.1 source roles or official splits drifted")
    consumed_test = [
        str(item)
        for item in result["burn_and_authorization"][
            "official_test_parent_sessions_consumed_for_f0_1_effect"
        ]
    ]
    if consumed_test != f0_1_ids[9:12]:
        raise ValueError("G0 F0.1 consumed-test source set drifted")
    outcome_flags = (
        "geometry_outcome_read",
        "teacher_outcome_read",
        "student_outcome_read",
    )
    if any(
        plan.get(flag) is not False
        for plan in (f0_plan, f0_1_plan)
        for flag in outcome_flags
    ):
        raise ValueError("G0 metadata-plan outcome firewall was open")
    fresh_eval = f0_candidates[9:12]
    if len(fresh_eval) != 3:
        raise ValueError("G0 fixed fresh-evaluation source count mismatch")
    for expected_rank, item in enumerate(fresh_eval, start=10):
        if (
            item.get("inventory_eligible") is not True
            or int(item.get("inventory_eligible_rank", -1)) != expected_rank
            or str(item.get("role")) != "heldout"
        ):
            raise ValueError(
                "G0 fixed fresh-evaluation F0 eligibility drifted"
            )
    fresh_ids = _source_ids(fresh_eval)
    outcome_open = (
        set(f0_1_ids)
        | set(_source_ids(acquisition["sources"]))
        | set(_source_ids(authority_cohort["sources"]))
        | set(_source_ids(teacher_opportunity["source_results"]))
        | set(consumed_test)
        | historical_burned
    )
    if set(fresh_ids) & outcome_open:
        raise ValueError(
            "G0 fresh-evaluation source is outcome-open or historically burned"
        )
    return f0_1_ids, fresh_ids


def _validate_role_sets(
    train: list[dict[str, Any]],
    fresh_eval: list[dict[str, Any]],
    heldout: list[dict[str, Any]],
) -> None:
    if len(train) != 9 or len(fresh_eval) != 3 or len(heldout) != 3:
        raise ValueError("G0 source role count mismatch")
    groups = [
        set(_source_ids(records))
        for records in (train, fresh_eval, heldout)
    ]
    if any(len(group) != size for group, size in zip(groups, (9, 3, 3))):
        raise ValueError("G0 duplicate parent session within role")
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("G0 parent session appears in multiple roles")


def _validate_parents(
    protocol_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    set[str],
]:
    protocol = _load_json(protocol_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != PROTOCOL_STATUS
    ):
        raise ValueError("G0 protocol identity mismatch")
    implementation = protocol.get("implementations", {}).get(
        "source_planner", {}
    )
    if (
        Path(str(implementation.get("path", ""))).as_posix()
        != "scripts/research/hftf/plan_stage_c_g0_signed_clearance_sources.py"
        or implementation.get("sha256")
        != _sha256(Path(__file__).resolve())
        or implementation.get("execution_authorized") is not True
    ):
        raise ValueError("G0 source planner implementation receipt mismatch")
    _, result = _parent(
        protocol_path, protocol, "f0_1_heldout_effect_result"
    )
    _, f0_plan = _parent(protocol_path, protocol, "f0_inventory_plan")
    _, f0_1_plan = _parent(
        protocol_path, protocol, "f0_1_cross_split_inventory_plan"
    )
    _, source_lock = _parent(
        protocol_path, protocol, "f0_1_source_lock"
    )
    _, acquisition = _parent(
        protocol_path, protocol, "f0_1_acquisition_audit"
    )
    _, authority_cohort = _parent(
        protocol_path, protocol, "f0_1_authority_cohort"
    )
    _, teacher_opportunity = _parent(
        protocol_path, protocol, "f0_1_teacher_opportunity"
    )
    _, heldout_contract = _parent(
        protocol_path, protocol, "f0_1_heldout_execution_contract"
    )
    _, mechanics = _parent(
        protocol_path, protocol, "swept_envelope_mechanics"
    )
    _, teacher_contract = _parent(
        protocol_path, protocol, "teacher_execution_contract"
    )
    ledger_path, ledger = _parent(
        protocol_path, protocol, "f0_source_pool_burn_ledger"
    )
    if (
        result.get("terminal")
        != "F0_1_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_"
        "STUDENT_SIGNAL_NOT_SUPPORTED_STOP"
        or f0_plan.get("terminal") != "F0_SANPO_FIXED_SOURCE_INVENTORY_READY"
        or f0_1_plan.get("terminal")
        != "F0_1_SANPO_CROSS_SPLIT_SOURCE_INVENTORY_READY"
        or source_lock.get("terminal")
        != "F0_1_SANPO_CROSS_SPLIT_SOURCE_LOCK_VALIDATED"
        or acquisition.get("terminal")
        != "F0_1_SANPO_ACQUISITION_AND_TRANSPORT_READY"
        or acquisition.get("all_sources_ok") is not True
        or authority_cohort.get("terminal")
        != "F0_1_SANPO_SOURCE_AUTHORITY_COHORT_READY"
        or authority_cohort.get("all_sources_authority_ready") is not True
        or teacher_opportunity.get("terminal")
        != "F0_1_SANPO_TEACHER_OPPORTUNITY_READY_FOR_CORPUS"
        or heldout_contract.get("status")
        != "FROZEN_AFTER_NINE_CHECKPOINT_GATE_BEFORE_HELDOUT_TARGET_"
        "MATERIALIZATION_OR_STUDENT_OUTPUT"
        or mechanics.get("status")
        != "FROZEN_DEVELOPMENT_CANARY_RESULT_NOT_RUN"
        or teacher_contract.get("status")
        != "FROZEN_BEFORE_FIRST_F0_1_TEACHER_GEOMETRY_OUTCOME"
    ):
        raise ValueError("G0 source-plan parent terminal mismatch")
    if (
        len(f0_plan.get("inventory_candidates", [])) != 12
        or len(f0_1_plan.get("sources", [])) != 12
        or len(source_lock.get("sources", [])) != 12
        or len(acquisition.get("sources", [])) != 12
        or len(authority_cohort.get("sources", [])) != 12
        or len(teacher_opportunity.get("source_results", [])) != 12
    ):
        raise ValueError("G0 source-plan parent source count mismatch")
    historical_burned = _burned_ids(ledger_path, ledger)
    _validate_outcome_and_freshness_chain(
        f0_plan,
        f0_1_plan,
        source_lock,
        acquisition,
        authority_cohort,
        teacher_opportunity,
        result,
        historical_burned,
    )
    return (
        protocol,
        result,
        f0_plan,
        f0_1_plan,
        source_lock,
        acquisition,
        authority_cohort,
        teacher_opportunity,
        historical_burned,
    )


def plan(protocol_path: Path, retries: int) -> dict[str, Any]:
    if retries <= 0:
        raise ValueError("Retries must be positive")
    (
        protocol,
        result,
        f0_plan,
        f0_1_plan,
        source_lock,
        acquisition,
        authority_cohort,
        teacher_opportunity,
        historical_burned,
    ) = _validate_parents(protocol_path)
    train: list[dict[str, Any]] = []
    for item in f0_1_plan["sources"][:9]:
        copied = dict(item)
        copied["g0_source_role"] = (
            "development_reuse_outcome_open_train"
            if str(item["role"]) == "train"
            else "development_reuse_outcome_open_model_selection"
        )
        copied["fresh_evidence_credit"] = False
        train.append(copied)
    fresh_eval: list[dict[str, Any]] = []
    for item in f0_plan["inventory_candidates"][9:12]:
        copied = dict(item)
        copied["role"] = "fresh_evaluation"
        copied["official_split"] = "train"
        copied["g0_source_role"] = (
            "one_shot_fresh_evaluation_metadata_planned_only"
        )
        copied["media_geometry_teacher_or_student_outcome_open"] = False
        copied["fresh_outcome_eligibility_if_role_preserved"] = True
        copied["fresh_evidence_obtained"] = False
        fresh_eval.append(copied)
    expected_fresh_eval = protocol["source_role_contract"][
        "one_shot_fresh_evaluation"
    ][
        "session_ids"
    ]
    if _source_ids(fresh_eval) != expected_fresh_eval:
        raise ValueError("G0 fixed fresh-evaluation sessions drifted")
    acquired_ids = set(_source_ids(acquisition["sources"]))
    outcome_open_ids = (
        acquired_ids
        | set(_source_ids(source_lock["sources"]))
        | set(_source_ids(authority_cohort["sources"]))
        | set(_source_ids(teacher_opportunity["source_results"]))
    )
    if outcome_open_ids & set(expected_fresh_eval):
        raise ValueError(
            "G0 fresh-evaluation session was previously outcome-open"
        )
    consumed_test = set(
        result["burn_and_authorization"][
            "official_test_parent_sessions_consumed_for_f0_1_effect"
        ]
    )
    excluded = (
        historical_burned
        | outcome_open_ids
        | set(_source_ids(train))
        | set(_source_ids(fresh_eval))
        | consumed_test
    )
    selection = protocol["source_role_contract"]["reserved_fresh_heldout"]
    split_name = (
        f"{GCS_PREFIX}/sanpo-synthetic/splits/test_session_ids.txt"
    )
    split_object = get_gcs_object(split_name, retries)
    split_text = fetch_text(
        media_url(split_name, split_object.get("generation")), retries
    )
    _validate_test_split(
        {
            "split_object_generation": selection[
                "official_test_split_generation"
            ],
            "split_text_sha256": selection[
                "official_test_split_text_sha256"
            ],
            "split_session_count": selection[
                "official_test_split_session_count"
            ],
        },
        str(split_object.get("generation")),
        split_text,
    )
    heldout: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    for session_id in (
        line.strip() for line in split_text.splitlines() if line.strip()
    ):
        if session_id in excluded:
            scanned.append(
                {
                    "session_id": session_id,
                    "inventory_eligible": False,
                    "reason": "historical_burn_or_g0_role_or_f0_1_consumed",
                }
            )
            continue
        try:
            item = _test_candidate(
                session_id, len(heldout) + 1, retries
            )
            item["role"] = "reserved_heldout"
            item["g0_source_role"] = (
                "metadata_only_future_heldout_reservation"
            )
            item["media_geometry_teacher_or_student_outcome_open"] = False
            item["fresh_outcome_eligibility_if_role_preserved"] = True
            item["fresh_evidence_obtained"] = False
            heldout.append(item)
            scanned.append(item)
            if len(heldout) == int(selection["count"]):
                break
        except (KeyError, OSError, TypeError, ValueError) as error:
            scanned.append(
                {
                    "session_id": session_id,
                    "inventory_eligible": False,
                    "reason": str(error),
                }
            )
    terminal = (
        READY if len(heldout) == int(selection["count"]) else NOT_EVALUABLE
    )
    if terminal == READY:
        _validate_role_sets(train, fresh_eval, heldout)
    return {
        "schema": SCHEMA,
        "terminal": terminal,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "source_planner_sha256": _sha256(Path(__file__).resolve()),
        "roles": {
            "development_reuse": train,
            "one_shot_fresh_evaluation": fresh_eval,
            "reserved_fresh_heldout": heldout,
        },
        "role_counts": {
            "development_reuse": len(train),
            "one_shot_fresh_evaluation": len(fresh_eval),
            "reserved_fresh_heldout": len(heldout),
        },
        "test_split_object": object_inventory(split_object),
        "test_split_text_sha256": selection[
            "official_test_split_text_sha256"
        ],
        "test_scan_ledger": scanned,
        "excluded_parent_session_count": len(excluded),
        "all_roles_parent_session_disjoint": terminal == READY,
        "firewall": {
            "new_rgb_depth_mask_or_pose_opened": False,
            "new_geometry_teacher_outcome_opened": False,
            "new_student_outcome_opened": False,
            "fresh_evaluation_acquisition_authorized": False,
            "reserved_heldout_acquisition_authorized": False,
        },
        "authorization": {
            "g0_d0_consumed_mechanics_may_execute": terminal == READY,
            "fresh_evaluation_acquisition_requires_g0_d0_support": True,
            "student_training_requires_separate_frozen_contract": True,
            "future_or_temporal_experiment_authorized": False,
            "mainline_promotion_authorized": False,
        },
    }


def _require_artifacts_output(path: Path) -> Path:
    expected = (
        Path(__file__).resolve().parents[3]
        / "artifacts.local/evidence/hftf/"
        "stage-c-g0-signed-clearance-source-plan-20260801/"
        "source_plan.json"
    ).resolve()
    if path.resolve() != expected:
        raise ValueError("G0 source plan output path is not canonical")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError("Refusing to overwrite G0 source plan")
        report = plan(args.protocol.resolve(), args.retries)
        output.parent.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(
            tempfile.mkdtemp(
                prefix=f"{output.parent.name}.partial-",
                dir=output.parent.parent,
            )
        )
        with (partial / output.name).open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        if output.parent.exists():
            raise FileExistsError("G0 source plan output root appeared")
        partial.replace(output.parent)
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "role_counts": report["role_counts"],
                    "output": str(output),
                }
            )
        )
        return 0 if report["terminal"] == READY else 2
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"terminal": NOT_EVALUABLE, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
