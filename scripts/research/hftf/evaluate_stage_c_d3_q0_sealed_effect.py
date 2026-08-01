#!/usr/bin/env python3
"""Evaluate the frozen D3-Q0 selected-six effect from sealed payloads.

All 42 future-blind prediction records must be durable before the evaluator
creates its one-shot sealed-payload-open receipt.  The evaluator opens only the
six selected payloads, verifies exact basis/support equality with the formal
predictions, recomputes the four qualification strata per source, and only
then invokes the unchanged D2 effect aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d2_transport_effect import (
    summarize_observations,
)
from preprocess_stage_c_d3_q0_selected_future_blind import (
    COMPLETION_SCHEMA,
    COMPLETION_TERMINAL,
    PREDICTION_SCHEMA,
)
from run_stage_c_d3_q0_next_slot import (
    ADVECTED,
    ARM_NAMES,
    PERSISTENCE,
    SEALED_SCHEMA,
    SEALED_STATUS,
    summarize_qualification,
)
from stage_c_d2_mechanics_common import (
    ANCHORS,
    EXPECTED_ANCHOR_COUNT,
    EXPECTED_HORIZON_RECORD_COUNT,
    HEIGHTS,
    HORIZONS,
    arrays_from_arm,
)
from stage_c_d3_q0_common import (
    SCREENING_ROOT_RELATIVE,
    aggregate_paths,
    canonical_json_sha256,
    load_json,
    preserve_temporary_artifact,
    sha256,
    slot_layout,
    validate_execution_contract,
    validate_selection,
    validate_selector,
    write_json_exclusive_fsync,
)


IMPLEMENTATION_KEY = "sealed_effect_evaluator"
ATTEMPT_SCHEMA = "blindassist_hftf_stage_c_d3_q0_effect_attempt"
ATTEMPT_STATUS = (
    "D3_Q0_EFFECT_ATTEMPT_FSYNCED_BEFORE_SEALED_PAYLOAD_OPEN"
)
TRUTH_OPEN_SCHEMA = (
    "blindassist_hftf_stage_c_d3_q0_sealed_payload_open_once_receipt"
)
TRUTH_OPEN_STATUS = (
    "D3_Q0_SELECTED_SIX_SEALED_PAYLOAD_OPEN_STARTED_NO_SECOND_OPEN"
)
RESULT_SCHEMA = "blindassist_hftf_stage_c_d3_q0_sealed_effect_result"
SUPPORTED = (
    "CAUSAL_SIGNED_CLEARANCE_TRANSPORT_SUPPORTED_FOR_RGB_STUDENT_PROTOCOL"
)
STOP = "CAUSAL_SIGNED_CLEARANCE_TRANSPORT_NOT_SUPPORTED_STOP"
MISMATCH = (
    "D3_NOT_EVALUABLE_QUALIFICATION_RECOMPUTE_MISMATCH_NO_REPLACEMENT"
)
FAILURE_SCHEMA = "blindassist_hftf_stage_c_d3_q0_effect_failure"
PRETRUTH_FAILURE = (
    "D3_Q0_EFFECT_PRETRUTH_VALIDATION_FAILED_NO_RERUN_NO_REPLACEMENT"
)
TRUTH_INTERRUPTED = (
    "D3_Q0_SEALED_PAYLOAD_EFFECT_INTERRUPTED_NO_SECOND_OPEN_NO_REPLACEMENT"
)


class EffectError(ValueError):
    """D3 formal effect inputs or one-shot authority are invalid."""


def _formal_paths(root: Path) -> dict[str, Path]:
    prediction_root = root / "formal" / "predictions"
    effect_root = root / "formal" / "effect"
    return {
        "prediction_root": prediction_root,
        "completion": prediction_root / "completion.json",
        "effect_root": effect_root,
        "attempt": effect_root / "attempt.json",
        "truth_open": effect_root / "sealed-payload-open-once.json",
        "result": effect_root / "result.json",
        "failure": effect_root / "failure.json",
    }


def _within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise EffectError(f"{label} escaped canonical root") from error
    return resolved


def _load_payload_once(
    path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise EffectError("D3 sealed payload hash mismatch")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise EffectError("D3 sealed payload is not a JSON object")
    return value


def load_prediction_completion(
    completion_path: Path,
    contract_sha256: str,
    roster_sha256: str,
    selection_sha256: str,
    selected_ids: list[str],
    prediction_root: Path,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, int], dict[str, Any]],
]:
    completion = load_json(completion_path)
    if (
        completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("terminal") != COMPLETION_TERMINAL
        or completion.get("contract_sha256") != contract_sha256
        or completion.get("roster_sha256") != roster_sha256
        or completion.get("selection_sha256") != selection_sha256
        or completion.get("prediction_record_count")
        != EXPECTED_ANCHOR_COUNT
        or completion.get("anchor_horizon_record_count")
        != EXPECTED_HORIZON_RECORD_COUNT
        or completion.get("all_records_durable_before_sealed_truth_open")
        is not True
        or completion.get("sealed_payload_read") is not False
        or completion.get("future_pose_depth_or_mask_read") is not False
        or completion.get("effect_evaluator_authorized") is not True
    ):
        raise EffectError("D3 prediction completion identity mismatch")
    expected = [
        (source_id, anchor)
        for source_id in selected_ids
        for anchor in ANCHORS
    ]
    receipts = completion.get("records", [])
    actual = [
        (
            str(item["session_id"]),
            int(item["anchor_normalized_index"]),
        )
        for item in receipts
    ]
    if actual != expected:
        raise EffectError("D3 prediction record order mismatch")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for key, receipt in zip(expected, receipts):
        path = _within(
            Path(str(receipt["path"])),
            prediction_root,
            "prediction record",
        )
        if sha256(path) != str(receipt["sha256"]):
            raise EffectError("D3 prediction record hash mismatch")
        record = load_json(path)
        if (
            record.get("schema") != PREDICTION_SCHEMA
            or str(record.get("session_id")) != key[0]
            or int(record.get("anchor_normalized_index", -1)) != key[1]
            or record.get("future_depth_mask_or_pose_read") is not False
            or record.get("future_pose_depth_or_mask_read") is not False
            or record.get("sealed_payload_read") is not False
            or int(record.get("unknown_to_safe_violations", -1)) != 0
        ):
            raise EffectError("D3 prediction identity/firewall mismatch")
        points_path = _within(
            Path(str(record["points"]["path"])),
            prediction_root,
            "prediction points",
        )
        if (
            sha256(points_path) != str(receipt["points_sha256"])
            or sha256(points_path) != str(record["points"]["sha256"])
        ):
            raise EffectError("D3 prediction points hash mismatch")
        records[key] = record
    return completion, records


def _truth_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    truth = record["truth"]
    known = np.asarray(truth["known"], dtype=bool)
    nullable = np.asarray(truth["signed_clearance_m"], dtype=object)
    counts = np.asarray(truth["probe_pass_counts"])
    if (
        known.shape != (2, 6, 6)
        or nullable.shape != (2, 6, 6)
        or counts.shape != (2, 6, 6)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or np.any(counts > 9)
        or not np.array_equal(known, counts >= 5)
    ):
        raise EffectError("D3 sealed truth support shape mismatch")
    clearance = np.full((2, 6, 6), np.nan, dtype=np.float64)
    for index in np.ndindex((2, 6, 6)):
        if known[index]:
            if nullable[index] is None:
                raise EffectError("D3 known truth is null")
            clearance[index] = float(nullable[index])
            if not math.isfinite(clearance[index]):
                raise EffectError("D3 truth clearance is non-finite")
        elif nullable[index] is not None:
            raise EffectError("D3 UNKNOWN truth became numeric")
    return known, clearance


def _support_matches_prediction(
    sealed_support: dict[str, Any],
    prediction_arm: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    known, clearance = arrays_from_arm(prediction_arm)
    counts = np.asarray(sealed_support["probe_pass_counts"])
    sealed_known = np.asarray(sealed_support["known"], dtype=bool)
    prediction_counts = np.asarray(prediction_arm["probe_pass_counts"])
    if (
        counts.shape != (2, 6, 6)
        or not np.issubdtype(counts.dtype, np.integer)
        or not np.array_equal(counts, prediction_counts)
        or not np.array_equal(sealed_known, known)
        or not np.array_equal(sealed_known, counts >= 5)
    ):
        raise EffectError(
            "D3 qualification/formal prediction support mismatch"
        )
    return known, clearance


def _selector_summary(selector: dict[str, Any]) -> dict[str, Any]:
    return {
        "strata": selector["strata"],
        "qualified": selector["qualified"],
    }


def _recomputed_selector_summary(
    summary: dict[str, Any],
) -> dict[str, Any]:
    strata = []
    for row in summary["strata"]:
        gates = {
            "coverage": row["gates"][
                "common_known_coverage_at_least_0_10"
            ],
            "risk": row["gates"]["known_risk_count_at_least_5"],
            "safe": row["gates"]["known_safe_count_at_least_20"],
            "unknown_to_safe": (
                summary["unknown_to_safe_violations"] == 0
            ),
        }
        strata.append(
            {
                "height": row["height"],
                "horizon_s": row["horizon_s"],
                "denominator": row["denominator"],
                "common_known_count": row["common_known_count"],
                "common_known_coverage": row[
                    "common_known_coverage"
                ],
                "truth_risk_count": row["known_risk_count"],
                "truth_safe_count": row["known_safe_count"],
                "unknown_to_safe_violation_count": (
                    summary["unknown_to_safe_violations"]
                ),
                "gates": gates,
                "passed": all(gates.values()),
            }
        )
    return {"strata": strata, "qualified": summary["qualified"]}


def result_authorization(terminal: str) -> dict[str, bool]:
    return {
        "freeze_rgb_student_contract_authorized": terminal == SUPPORTED,
        "rgb_student_protocol_execution_authorized": False,
        "rgb_student_training_authorized": False,
        "rgb_student_execution_authorized": False,
        "reserved_official_test_open_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
        "android_changed": False,
        "production_authorized": False,
        "safety_claim_authorized": False,
    }


def _prepare_effect_inputs(
    context: dict[str, Any],
    root: Path,
    paths: dict[str, Path],
    selection_path: Path,
) -> tuple[
    dict[str, Any],
    list[str],
    dict[tuple[str, int], dict[str, Any]],
    list[tuple[dict[str, Any], Path, dict[str, Any]]],
]:
    selection = validate_selection(
        selection_path,
        context["slots"],
        sha256(context["contract_path"]),
        context["roster_sha256"],
    )
    selected = selection["selected_sources"]
    selected_ids = [str(item["session_id"]) for item in selected]
    if len(selected_ids) != 6 or len(set(selected_ids)) != 6:
        raise EffectError("D3 effect selection must be exactly six parents")
    _completion, predictions = load_prediction_completion(
        paths["completion"],
        sha256(context["contract_path"]),
        context["roster_sha256"],
        sha256(selection_path),
        selected_ids,
        paths["prediction_root"],
    )
    selector_rows: list[
        tuple[dict[str, Any], Path, dict[str, Any]]
    ] = []
    for row in selected:
        slot_index = int(row["slot_index"])
        source = context["slots"][slot_index - 1]
        layout = slot_layout(root, source)
        selector_path = Path(layout["selector"]).resolve()
        if sha256(selector_path) != str(row["selector_sha256"]):
            raise EffectError("D3 selected selector hash mismatch")
        selector = validate_selector(
            load_json(selector_path),
            source,
            sha256(context["contract_path"]),
            context["roster_sha256"],
        )
        payload_path = _within(
            Path(layout["sealed_payload"]),
            root,
            "sealed payload",
        )
        if not payload_path.is_file():
            raise EffectError("D3 sealed payload is absent")
        selector_rows.append((source, payload_path, selector))
    return selection, selected_ids, predictions, selector_rows


def _freeze_effect_failure(
    paths: dict[str, Path],
    context: dict[str, Any],
    selection_path: Path,
    error: BaseException,
    terminal: str,
    *,
    include_parent_hashes: bool = True,
) -> None:
    paths["effect_root"].mkdir(parents=True, exist_ok=True)
    if paths["result"].exists() or paths["failure"].exists():
        return
    preserve_temporary_artifact(paths["failure"])
    write_json_exclusive_fsync(
        paths["failure"],
        {
            "schema": FAILURE_SCHEMA,
            "terminal": terminal,
            "workflow_profile": "THESIS_DEVELOPMENT",
            "contract_sha256": sha256(context["contract_path"]),
            "selection_sha256": (
                sha256(selection_path)
                if include_parent_hashes and selection_path.is_file()
                else None
            ),
            "prediction_completion_sha256": (
                sha256(paths["completion"])
                if include_parent_hashes and paths["completion"].is_file()
                else None
            ),
            "effect_attempt_sha256": (
                sha256(paths["attempt"])
                if paths["attempt"].is_file()
                else None
            ),
            "sealed_payload_open_once_receipt_sha256": (
                sha256(paths["truth_open"])
                if paths["truth_open"].is_file()
                else None
            ),
            "sealed_payload_open_started": paths["truth_open"].is_file(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "partial_artifacts_preserved": True,
            "effect_rerun_authorized": False,
            "second_sealed_payload_open_authorized": False,
            "future_media_open_authorized": False,
            "source_replacement_authorized": False,
            "same_cohort_retuning_authorized": False,
            "freeze_rgb_student_contract_authorized": False,
        },
    )


def run_evaluator(contract_path: Path) -> dict[str, Any]:
    context = validate_execution_contract(
        contract_path,
        IMPLEMENTATION_KEY,
        Path(__file__),
        verify_git=True,
    )
    root = Path(context["root"]).resolve()
    if root != (
        Path(__file__).resolve().parents[3] / SCREENING_ROOT_RELATIVE
    ).resolve():
        raise EffectError("D3 screening root is noncanonical")
    paths = _formal_paths(root)
    selection_path = Path(aggregate_paths(root)["selection"]).resolve()
    if paths["result"].exists() or paths["failure"].exists():
        raise FileExistsError("D3 effect terminal artifact already exists")
    if paths["effect_root"].exists() and any(
        paths["effect_root"].iterdir()
    ):
        terminal = (
            TRUTH_INTERRUPTED
            if paths["truth_open"].is_file()
            else PRETRUTH_FAILURE
        )
        error = EffectError(
            "prior effect attempt is incomplete; failure frozen "
            "without reopening predictions, selectors, or sealed payloads"
        )
        _freeze_effect_failure(
            paths,
            context,
            selection_path,
            error,
            terminal,
            include_parent_hashes=False,
        )
        raise error
    if not selection_path.is_file() or not paths["completion"].is_file():
        raise EffectError(
            "D3 effect prerequisites are not durably complete"
        )
    try:
        (
            _selection,
            selected_ids,
            predictions,
            selector_rows,
        ) = _prepare_effect_inputs(
            context,
            root,
            paths,
            selection_path,
        )
    except BaseException as error:
        _freeze_effect_failure(
            paths,
            context,
            selection_path,
            error,
            PRETRUTH_FAILURE,
        )
        raise
    try:
        write_json_exclusive_fsync(
            paths["attempt"],
            {
                "schema": ATTEMPT_SCHEMA,
                "status": ATTEMPT_STATUS,
                "workflow_profile": "THESIS_DEVELOPMENT",
                "contract_sha256": sha256(context["contract_path"]),
                "roster_sha256": context["roster_sha256"],
                "selection_sha256": sha256(selection_path),
                "prediction_completion_sha256": sha256(
                    paths["completion"]
                ),
                "all_predictions_validated_before_attempt": True,
                "sealed_payload_opened_before_attempt": False,
                "second_sealed_payload_open_authorized": False,
                "future_media_open_authorized": False,
            },
        )
        write_json_exclusive_fsync(
            paths["truth_open"],
            {
                "schema": TRUTH_OPEN_SCHEMA,
                "status": TRUTH_OPEN_STATUS,
                "workflow_profile": "THESIS_DEVELOPMENT",
                "contract_sha256": sha256(context["contract_path"]),
                "selection_sha256": sha256(selection_path),
                "prediction_completion_sha256": sha256(
                    paths["completion"]
                ),
                "effect_attempt_sha256": sha256(paths["attempt"]),
                "all_predictions_durable_before_receipt": True,
                "sealed_payload_opened_before_receipt": False,
                "payload_count": 6,
                "future_media_open_authorized": False,
                "second_sealed_payload_open_authorized": False,
            },
        )
    except BaseException as error:
        _freeze_effect_failure(
            paths,
            context,
            selection_path,
            error,
            PRETRUTH_FAILURE,
        )
        raise
    observations: list[dict[str, Any]] = []
    payload_receipts: list[dict[str, Any]] = []
    recompute_mismatch: list[dict[str, Any]] = []
    try:
        for source, payload_path, selector in selector_rows:
            expected_payload_sha = selector[
                "source_authority_and_content_hashes"
            ]["sealed_payload_sha256"]
            payload = _load_payload_once(
                payload_path,
                expected_payload_sha,
            )
            payload_receipts.append(
                {
                    "slot_index": int(
                        source["d3_roster_slot_index"]
                    ),
                    "session_id": str(source["session_id"]),
                    "path": str(payload_path),
                    "sha256": expected_payload_sha,
                }
            )
            if (
                payload.get("schema") != SEALED_SCHEMA
                or payload.get("status") != SEALED_STATUS
                or int(payload.get("slot_index", -1))
                != int(source["d3_roster_slot_index"])
                or str(payload.get("session_id"))
                != str(source["session_id"])
                or payload.get("contract_sha256")
                != sha256(context["contract_path"])
                or payload.get("roster_sha256")
                != context["roster_sha256"]
                or payload.get("content_index_file_sha256")
                != selector["source_authority_and_content_hashes"][
                    "content_index_sha256"
                ]
                or payload.get("observation_count") != 14
                or payload.get(
                    "candidate_arm_clearance_computed_or_written"
                )
                is not False
                or payload.get(
                    "mae_f1_confusion_delta_or_improvement_computed"
                )
                is not False
                or payload.get("future_media_may_be_opened_again")
                is not False
            ):
                raise EffectError("D3 sealed payload identity mismatch")
            recomputed = summarize_qualification(payload["observations"])
            selector_recomputed = _recomputed_selector_summary(recomputed)
            if canonical_json_sha256(
                selector_recomputed
            ) != canonical_json_sha256(
                _selector_summary(selector)
            ):
                recompute_mismatch.append(
                    {
                        "slot_index": int(source["d3_roster_slot_index"]),
                        "session_id": str(source["session_id"]),
                        "selector_sha256": canonical_json_sha256(
                            _selector_summary(selector)
                        ),
                        "recomputed_sha256": canonical_json_sha256(
                            selector_recomputed
                        ),
                    }
                )
                continue
            by_prediction = {
                anchor: {
                    float(item["horizon_s"]): item
                    for item in predictions[
                        (str(source["session_id"]), anchor)
                    ]["horizons"]
                }
                for anchor in ANCHORS
            }
            for sealed_record in payload["observations"]:
                anchor = int(sealed_record["anchor_normalized_index"])
                horizon = float(sealed_record["horizon_s"])
                prediction = by_prediction[anchor][horizon]
                if canonical_json_sha256(
                    sealed_record["predicted_basis"]
                ) != canonical_json_sha256(prediction["predicted_basis"]):
                    recompute_mismatch.append(
                        {
                            "slot_index": int(
                                source["d3_roster_slot_index"]
                            ),
                            "session_id": str(source["session_id"]),
                            "anchor": anchor,
                            "horizon_s": horizon,
                            "reason": "predicted_basis_mismatch",
                        }
                    )
                    continue
                try:
                    arms = {
                        arm: _support_matches_prediction(
                            sealed_record["support"][arm],
                            prediction["arms"][arm],
                        )
                        for arm in ARM_NAMES
                    }
                except EffectError as error:
                    recompute_mismatch.append(
                        {
                            "slot_index": int(
                                source["d3_roster_slot_index"]
                            ),
                            "session_id": str(source["session_id"]),
                            "anchor": anchor,
                            "horizon_s": horizon,
                            "reason": str(error),
                        }
                    )
                    continue
                truth_known, truth_clearance = _truth_arrays(
                    sealed_record
                )
                observations.append(
                    {
                        "session_id": str(source["session_id"]),
                        "anchor_normalized_index": anchor,
                        "horizon_s": horizon,
                        "truth_known": truth_known,
                        "truth_clearance": truth_clearance,
                        "arms": arms,
                        "unknown_to_safe_violations": int(
                            sealed_record.get(
                                "unknown_to_safe_violations",
                                0,
                            )
                        ),
                    }
                )
        if recompute_mismatch:
            terminal = MISMATCH
            result = {
                "schema": RESULT_SCHEMA,
                "terminal": terminal,
                "workflow_profile": "THESIS_DEVELOPMENT",
                "contract_sha256": sha256(context["contract_path"]),
                "selection_sha256": sha256(selection_path),
                "prediction_completion_sha256": sha256(
                    paths["completion"]
                ),
                "sealed_payload_open_once_receipt_sha256": sha256(
                    paths["truth_open"]
                ),
                "sealed_payload_count_opened": len(payload_receipts),
                "qualification_recompute_match": False,
                "mismatches": recompute_mismatch,
                "effect_metrics_computed": False,
                "source_replacement_authorized": False,
                "authorization": result_authorization(terminal),
            }
        else:
            if len(observations) != EXPECTED_HORIZON_RECORD_COUNT:
                raise EffectError("D3 formal observation count mismatch")
            metrics = summarize_observations(selected_ids, observations)
            if not metrics["opportunity_adequate"]:
                terminal = MISMATCH
                result = {
                    "schema": RESULT_SCHEMA,
                    "terminal": terminal,
                    "workflow_profile": "THESIS_DEVELOPMENT",
                    "contract_sha256": sha256(context["contract_path"]),
                    "selection_sha256": sha256(selection_path),
                    "prediction_completion_sha256": sha256(
                        paths["completion"]
                    ),
                    "sealed_payload_open_once_receipt_sha256": sha256(
                        paths["truth_open"]
                    ),
                    "sealed_payload_count_opened": 6,
                    "qualification_recompute_match": False,
                    "mismatches": [
                        {"reason": "formal_opportunity_not_adequate"}
                    ],
                    "effect_metrics_computed": False,
                    "source_replacement_authorized": False,
                    "authorization": result_authorization(terminal),
                }
            else:
                terminal = (
                    SUPPORTED
                    if metrics["all_effect_gates_passed"]
                    else STOP
                )
                result = {
                    "schema": RESULT_SCHEMA,
                    "terminal": terminal,
                    "workflow_profile": "THESIS_DEVELOPMENT",
                    "contract_sha256": sha256(context["contract_path"]),
                    "selection_sha256": sha256(selection_path),
                    "prediction_completion_sha256": sha256(
                        paths["completion"]
                    ),
                    "sealed_payload_open_once_receipt_sha256": sha256(
                        paths["truth_open"]
                    ),
                    "sealed_payload_count_opened": 6,
                    "qualification_recompute_match": True,
                    "source_replacement_authorized": False,
                    "same_cohort_retuning_authorized": False,
                    "metrics": metrics,
                    "payload_receipts": payload_receipts,
                    "authorization": result_authorization(terminal),
                }
        write_json_exclusive_fsync(paths["result"], result)
        return result
    except BaseException as error:
        _freeze_effect_failure(
            paths,
            context,
            selection_path,
            error,
            TRUTH_INTERRUPTED,
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = run_evaluator(args.contract)
    print(json.dumps({"terminal": result["terminal"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
