#!/usr/bin/env python3
"""Seal the R11 source-only 48-to-24 parent selection before any FARO access."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import inspect
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psutil

from scripts.research.taro_o0r_factor_headroom_runtime.evidence import (
    FactorEvidenceWriter,
)
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import (
    positive_occupancy_factor as r7_positive,
)
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r9_clear_runtime import clear_enrichment_fit
from scripts.research.taro_o1r_r11_abstention_runtime import (
    abstention_candidate,
    fresh_pool,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_top24_selection_execution_lock.v1"
LOCK_ID = "TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_ONE_SHOT_EXECUTION_LOCK"
LOCK_RELATIVE = (
    "docs/research/taro/"
    "TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_ONE_SHOT_EXECUTION_LOCK_2026-08-13.json"
)
PHASE_A_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-phase-a-r0"
PHASE_A_TERMINAL_RELATIVE = f"{PHASE_A_ROOT}/terminal.json"
PHASE_A_COMPLETION_RELATIVE = f"{PHASE_A_ROOT}/phase-a-completion.json"
PHASE_A_AUDIT_ROOT = (
    "artifacts.local/evidence/taro/"
    "o1r-r11-fresh-pool-phase-a-validator-round12-repair-r0"
)
PHASE_A_AUDIT_RELATIVE = f"{PHASE_A_AUDIT_ROOT}/post-result-audit.json"
INVENTORY_RELATIVE = (
    "artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0/exact-frame-plan.json"
)
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-top24-selection-r0"

PASS_TERMINAL = "TARO_O1R_R11_FRESH_POOL_TOP24_SOURCE_ONLY_SELECTION_SEALED_PASS"
FAIL_TERMINAL = "TARO_O1R_R11_FRESH_POOL_TOP24_SELECTION_EXECUTION_INVALID"
PHASE_A_PASS_TERMINAL = "TARO_O1R_R11_FRESH_POOL_PHASE_A_SOURCE_ONLY_SEALED_PASS"
PHASE_A_AUDIT_PASS = "TARO_O1R_R11_PHASE_A_OFFLINE_VALIDATOR_ROUND12_REPAIR_PASS"

PARENT_COUNT = 48
SELECTED_PARENT_COUNT = 24
FRAME_COUNT = 1043
QUERY_COUNT = 9387
PHASE_A_PRE_TERMINAL_FILE_COUNT = 5218
PHASE_A_FINAL_FILE_COUNT = 5219
SUCCESS_PRE_TERMINAL_FILE_COUNT = 3
SUCCESS_FINAL_FILE_COUNT = 4
TERMINAL_RESERVE_BYTES = 4_194_304
TERMINAL_WALL_RESERVE_SECONDS = 60
TERMINAL_RESERVE_NAME = ".terminal-reserve.bin"
FAILURE_MESSAGE_MAX_CHARS = 4096

FROZEN_FRAME_COUNTS = [
    20, 14, 23, 24, 29, 7, 12, 14, 10, 21, 28, 15, 11, 28, 29, 72,
    36, 14, 18, 4, 54, 32, 83, 17, 15, 16, 29, 10, 12, 34, 7, 14,
    11, 6, 9, 1, 46, 6, 27, 26, 50, 9, 11, 27, 12, 9, 28, 13,
]
EXPECTED_PARENT_IDENTITIES = tuple((visit, video) for visit, video, _rank in fresh_pool.EXPECTED_POOL)

FROZEN_SELECTOR_ID = "TARO_R9_SOURCE_ONLY_CLEAR_ENRICHMENT_GRID_SEARCH_V1"
FROZEN_SELECTOR_CONTENT_SHA256 = "67FD8430418E23E4C974EBA4D7F49DCBD4DE66164A16491DE76F05AC974796CC"
FROZEN_RULE_ID = "02CE016D6B0011F0"
FROZEN_RULE = {
    "state_policy": "UNKNOWN_ONLY",
    "minimum_far_valid_anchor_count": 6,
    "maximum_far_valid_anchor_count": 1_000_000,
    "far_fraction_index": 0,
    "maximum_far_fraction": 0.0,
    "minimum_observed_support_points": 0,
    "require_query_receipt": True,
    "require_positive_obstacle_veto_false": True,
    "require_all_occupied_hits_false": True,
    "rule_id": FROZEN_RULE_ID,
}
FROZEN_SELECTOR = {
    "selector_id": FROZEN_SELECTOR_ID,
    "selector_content_sha256": FROZEN_SELECTOR_CONTENT_SHA256,
    "rule_id": FROZEN_RULE_ID,
}

PHASE_A_TERMINAL_FILE_SHA256 = "C4084BDBD00ECCA5735753E20893348505AB2A19650FB334850ADCD6D5173186"
PHASE_A_TERMINAL_CONTENT_SHA256 = "596DF5960A483F45A8777B914C50412B6A4A29CBCF39581FE0984DC658F77B71"
PHASE_A_COMPLETION_FILE_SHA256 = "739F3796925B88FCE6F8443DB1979D6644F4937AC913CF60C4BFB618003CF046"
PHASE_A_COMPLETION_CONTENT_SHA256 = "ADB3D8317BC40A09266A5B5FB08D6554598AFC2D038B4E37C3C308BD7CB5EB7A"
PHASE_A_AUDIT_FILE_SHA256 = "2D80268D3E6D928F928F7C3B8AF8B14C7396D16A305521293C052DF6A378D19C"
PHASE_A_AUDIT_CONTENT_SHA256 = "1717F1601A037E43A31EDF5861D496A9AEFE0C4F561F5A8DA0886E3E21EB0824"
INVENTORY_CONTENT_SHA256 = "35156C2901A4CBEEDB6D611A56ABE3D711CEB68EF932480C21428BA4FF741600"

EXPECTED_ARGV = [
    "-m",
    "scripts.research.taro_o1r_r11_abstention_runtime.run_top24_selection",
    "--execution-lock",
    LOCK_RELATIVE,
]
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "授权",
    "scope": (
        "Exact frozen R11 48-parent Training pool and 144-URL plan: zero-body HEAD, bounded source download "
        "and integrity validation, all-48 source-only Phase A, source-only top-24 selection, then FARO only "
        "for the sealed top 24; no training, device, deployment, product, safety, or redistribution authority."
    ),
}
EXPECTED_AUTHORITY = {
    "sealed_phase_a_reload": True,
    "source_only_parent_scoring": True,
    "top24_selection": True,
    "source_zip_member_payload_read": False,
    "highres_depth_member_payload_read": False,
    "faro_read": False,
    "truth_read": False,
    "label_read": False,
    "outcome_read": False,
    "model_execution": False,
    "candidate_rerun": False,
    "threshold_fit": False,
    "training": False,
    "network": False,
    "device": False,
    "deployment": False,
    "product": False,
    "safety": False,
    "redistribution": False,
}
EXPECTED_RESOURCE_BUDGET = {
    "maximum_wall_seconds": 7200,
    "maximum_peak_rss_bytes": 8_589_934_592,
    "maximum_evidence_bytes": 16_777_216,
}
EXPECTED_ONE_SHOT_POLICY = {
    "consumed_on_output_root_creation": True,
    "failure_does_not_restore_authority": True,
    "atomic_terminal_bundle": True,
    "success_pre_terminal_file_count": SUCCESS_PRE_TERMINAL_FILE_COUNT,
    "success_final_file_count": SUCCESS_FINAL_FILE_COUNT,
    "terminal_reserve_bytes": TERMINAL_RESERVE_BYTES,
    "terminal_wall_reserve_seconds": TERMINAL_WALL_RESERVE_SECONDS,
}
EXPECTED_BINDINGS = {
    "R11_PROTOCOL": "docs/research/taro/TARO_O1R_R11_POSITIVE_OCCUPANCY_ABSTENTION_AND_FRESH_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json",
    "R11_DATA_USE_AUTHORIZATION": "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-12.json",
    "R11_INVENTORY": INVENTORY_RELATIVE,
    "R11_PHASE_A_EXECUTION_LOCK": "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json",
    "R11_PHASE_A_TERMINAL": PHASE_A_TERMINAL_RELATIVE,
    "R11_PHASE_A_COMPLETION": PHASE_A_COMPLETION_RELATIVE,
    "R11_PHASE_A_REPAIRED_AUDIT": PHASE_A_AUDIT_RELATIVE,
    "R9_DEVELOPMENT_RESULT": "docs/research/taro/TARO_O1R_R9_CLEAR_ENRICHMENT_DEVELOPMENT_RESULT_2026-08-12.json",
    "R9_SELECTOR_ARTIFACT": "artifacts.local/evidence/taro/o1r-r9-clear-enrichment-development-r0/selector.json",
    "R9_SELECTOR_RUNTIME": "scripts/research/taro_o1r_r9_clear_runtime/clear_enrichment_fit.py",
    "SOURCE_ADAPTER_RUNTIME": "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py",
    "TRUTH_MATERIALIZER_RUNTIME": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "CANDIDATE_SCALE_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py",
    "REDUCER_INTEGRATION_RUNTIME": "scripts/research/taro_o1r_reducer_integration_runtime/reducer_integration.py",
    "R7_SOURCE_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R7_POSITIVE_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/positive_occupancy_factor.py",
    "R7_FRESH_COHORT_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/fresh_confirmation_cohort.py",
    "R10_FRESH_POOL_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/fresh_pool.py",
    "R11_ABSTENTION_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/abstention_candidate.py",
    "R11_FRESH_POOL_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/fresh_pool.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "R11_TOP24_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_top24_selection.py",
    "R11_TOP24_RUNNER_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_run_top24_selection.py",
    "R11_TOP24_INDEPENDENT_VALIDATOR": "scripts/research/taro_o1r_r11_abstention_runtime/validate_top24_selection.py",
    "R11_TOP24_VALIDATOR_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_validate_top24_selection.py",
    "R11_TOP24_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_IMPLEMENTATION_LOCK_2026-08-13.md",
}
ARTIFACT_BINDING_ROLES = {
    "R11_INVENTORY",
    "R11_PHASE_A_TERMINAL",
    "R11_PHASE_A_COMPLETION",
    "R11_PHASE_A_REPAIRED_AUDIT",
    "R9_SELECTOR_ARTIFACT",
}

PARENT_SCORES_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_source_only_parent_scores.v1"
SELECTION_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_top24_source_only_selection.v1"
RESULT_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_top24_selection_result.v1"
TERMINAL_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_top24_selection_terminal.v1"
FAILURE_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_top24_selection_failure.v1"


class FreshTop24SelectionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise FreshTop24SelectionError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "R11_TOP24_JSON_OBJECT", f"JSON object required: {path}")
    return value


def _load_json_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "R11_TOP24_JSON_OBJECT", f"gzip JSON object required: {path}")
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R11_TOP24_SEAL_COLLISION", "caller supplied content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str, code: str = "R11_TOP24_SEAL") -> dict[str, Any]:
    require(isinstance(value, Mapping), code, "sealed record must be an object")
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema
        and isinstance(observed, str)
        and adapter.canonical_sha256(record) == observed,
        code,
        f"record schema/content seal drift: {schema}",
    )
    record["content_sha256"] = observed
    return record


def validate_frozen_rule(value: Mapping[str, Any]) -> dict[str, Any]:
    rule = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require(rule == FROZEN_RULE, "R11_TOP24_RULE_DRIFT", "frozen R9 rule drift")
    payload = {key: item for key, item in rule.items() if key != "rule_id"}
    require(adapter.canonical_sha256(payload)[:16] == FROZEN_RULE_ID, "R11_TOP24_RULE_ID", "rule id drift")
    return rule


def validate_frozen_selector(value: Mapping[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(
        observed == FROZEN_SELECTOR_CONTENT_SHA256
        and adapter.canonical_sha256(record) == FROZEN_SELECTOR_CONTENT_SHA256,
        "R11_TOP24_SELECTOR_SEAL",
        "frozen R9 selector content drift",
    )
    record["content_sha256"] = observed
    validated = clear_enrichment_fit.validate_selector(record)
    require(
        validated.get("selector_id") == FROZEN_SELECTOR_ID
        and validate_frozen_rule(validated.get("chosen_rule", {})) == FROZEN_RULE,
        "R11_TOP24_SELECTOR_IDENTITY",
        "frozen R9 selector identity/rule drift",
    )
    return validated


def _tie_break(parent_id: str, video_id: str) -> str:
    return adapter.canonical_sha256([parent_id, video_id])


def _canonical_fraction(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(float(numerator / denominator), adapter.FLOAT_DECIMALS)


def score_parent(source_frame_records: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    """Score one parent from sealed source records only; no result-side input is accepted."""

    frozen_rule = validate_frozen_rule(rule)
    require(bool(source_frame_records), "R11_TOP24_PARENT_EMPTY", "parent source records are empty")
    rows = [copy.deepcopy(dict(row)) for row in source_frame_records]
    parent_id, video_id = str(rows[0].get("parent_id")), str(rows[0].get("video_id"))
    require(
        all(str(row.get("parent_id")) == parent_id and str(row.get("video_id")) == video_id for row in rows),
        "R11_TOP24_PARENT_IDENTITY",
        "source records mix parent identities",
    )
    available = 0
    eligible = 0
    source_hashes: list[str] = []
    for row in rows:
        require(
            row.get("source_phase_has_label_input") is False
            and row.get("training_steps") == row.get("network_requests") == 0,
            "R11_TOP24_SOURCE_FIREWALL",
            "source record crosses source-only selection firewall",
        )
        features = row.get("query_features")
        source_hash = row.get("content_sha256")
        require(
            isinstance(features, list) and len(features) == 9 and isinstance(source_hash, str) and len(source_hash) == 64,
            "R11_TOP24_SOURCE_RECORD",
            "source record hash/query cardinality drift",
        )
        source_hashes.append(source_hash)
        for feature in features:
            require(isinstance(feature, Mapping), "R11_TOP24_QUERY_FEATURE", "query feature must be an object")
            available += feature.get("query_receipt") is not None
            eligible += clear_enrichment_fit.eligible(feature, frozen_rule)
    return {
        **FROZEN_SELECTOR,
        "parent_id": parent_id,
        "video_id": video_id,
        "frame_count": len(rows),
        "query_count": len(rows) * 9,
        "available_query_count": int(available),
        "eligible_query_count": int(eligible),
        "eligible_fraction_of_available": _canonical_fraction(eligible, available),
        "source_frame_hash_sequence_sha256": adapter.canonical_sha256(source_hashes),
        "tie_break_sha256": _tie_break(parent_id, video_id),
        "source_zip_member_payload_reads": 0,
        "highres_depth_member_payload_reads": 0,
        "faro_reads": 0,
        "truth_reads": 0,
        "label_reads": 0,
        "outcome_reads": 0,
        "model_executions": 0,
        "clear_output_emitted": False,
        "unknown_is_negative": False,
        "training_steps": 0,
        "network_requests": 0,
    }


SCORE_FIELDS = {
    *FROZEN_SELECTOR,
    "parent_id", "video_id", "frame_count", "query_count", "available_query_count",
    "eligible_query_count", "eligible_fraction_of_available", "source_frame_hash_sequence_sha256",
    "tie_break_sha256", "source_zip_member_payload_reads", "highres_depth_member_payload_reads",
    "faro_reads", "truth_reads", "label_reads", "outcome_reads", "model_executions",
    "clear_output_emitted", "unknown_is_negative", "training_steps", "network_requests",
}


def rank_parent_scores(parent_scores: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(parent_scores) == PARENT_COUNT, "R11_TOP24_PARENT_COUNT", "selection requires exactly 48 scores")
    scores = [copy.deepcopy(dict(row)) for row in parent_scores]
    identities = [(str(row.get("parent_id")), str(row.get("video_id"))) for row in scores]
    require(len(set(identities)) == PARENT_COUNT, "R11_TOP24_PARENT_DUPLICATE", "parent identities are not unique")
    for row, identity in zip(scores, identities, strict=True):
        expected_fraction = _canonical_fraction(int(row.get("eligible_query_count", -1)), int(row.get("available_query_count", -1)))
        require(
            set(row) == SCORE_FIELDS
            and {key: row[key] for key in FROZEN_SELECTOR} == FROZEN_SELECTOR
            and row["query_count"] == 9 * row["frame_count"]
            and 0 <= row["eligible_query_count"] <= row["available_query_count"] <= row["query_count"]
            and row["eligible_fraction_of_available"] == expected_fraction
            and isinstance(row["source_frame_hash_sequence_sha256"], str)
            and len(row["source_frame_hash_sequence_sha256"]) == 64
            and row["tie_break_sha256"] == _tie_break(*identity)
            and row["source_zip_member_payload_reads"] == row["highres_depth_member_payload_reads"] == 0
            and row["faro_reads"] == row["truth_reads"] == row["label_reads"] == row["outcome_reads"] == 0
            and row["model_executions"] == row["training_steps"] == row["network_requests"] == 0
            and row["clear_output_emitted"] is False
            and row["unknown_is_negative"] is False,
            "R11_TOP24_PARENT_SCORE",
            "parent score fields, selector, or firewall drift",
        )
    return sorted(scores, key=lambda row: (-int(row["eligible_query_count"]), str(row["tie_break_sha256"])))


def validate_parent_scores(value: Mapping[str, Any]) -> dict[str, Any]:
    record = _validate_seal(value, PARENT_SCORES_SCHEMA, "R11_TOP24_SCORES_SEAL")
    scores = record.get("parent_scores")
    require(isinstance(scores, list), "R11_TOP24_SCORES", "parent score list missing")
    ranked = rank_parent_scores(scores)
    require(
        record.get("selector") == FROZEN_SELECTOR
        and record.get("parent_count") == PARENT_COUNT
        and record.get("frame_count") == FRAME_COUNT
        and record.get("query_count") == QUERY_COUNT
        and [(row["parent_id"], row["video_id"]) for row in scores] == list(EXPECTED_PARENT_IDENTITIES)
        and [row["frame_count"] for row in scores] == FROZEN_FRAME_COUNTS
        and record.get("ranked_parent_identities") == [[row["parent_id"], row["video_id"]] for row in ranked]
        and record.get("all_48_source_records_sealed_before_scoring") is True
        and record.get("all_48_parent_scores_sealed_before_faro") is True
        and _zero_read_record(record),
        "R11_TOP24_SCORES",
        "sealed parent scores drift",
    )
    return record


def validate_selection(value: Mapping[str, Any], parent_scores: Mapping[str, Any]) -> dict[str, Any]:
    scores_record = validate_parent_scores(parent_scores)
    record = _validate_seal(value, SELECTION_SCHEMA, "R11_TOP24_SELECTION_SEAL")
    selected = rank_parent_scores(scores_record["parent_scores"])[:SELECTED_PARENT_COUNT]
    require(
        record.get("selector") == FROZEN_SELECTOR
        and record.get("parent_scores_sha256") == scores_record["content_sha256"]
        and record.get("selected_parent_count") == SELECTED_PARENT_COUNT
        and record.get("selected_parent_identities") == [[row["parent_id"], row["video_id"]] for row in selected]
        and record.get("selected_parent_scores") == selected
        and record.get("selection_sealed_before_faro") is True
        and record.get("read_unselected_faro") is False
        and record.get("source_reselection_after_faro") is False
        and record.get("parent_reselection_after_faro") is False
        and record.get("candidate_or_threshold_reselection_after_faro") is False
        and record.get("unknown_is_negative") is False
        and _zero_read_record(record),
        "R11_TOP24_SELECTION",
        "sealed top-24 selection drift",
    )
    return record


def _zero_read_record(record: Mapping[str, Any]) -> bool:
    return (
        record.get("source_zip_member_payload_reads") == 0
        and record.get("highres_depth_member_payload_reads") == 0
        and record.get("faro_reads") == record.get("truth_reads") == 0
        and record.get("label_reads") == record.get("outcome_reads") == 0
        and record.get("model_executions") == record.get("training_steps") == record.get("network_requests") == 0
    )


def build_selection(
    completion: Mapping[str, Any],
    source_frame_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(len(source_frame_records) == FRAME_COUNT, "R11_TOP24_SOURCE_COUNT", "source frame count drift")
    require(
        adapter.canonical_sha256([row["content_sha256"] for row in source_frame_records])
        == completion.get("source_frame_hash_sequence_sha256"),
        "R11_TOP24_SOURCE_SEQUENCE",
        "source hash sequence differs from Phase A completion",
    )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for source in source_frame_records:
        grouped[(str(source["parent_id"]), str(source["video_id"]))].append(source)
    require(set(grouped) == set(EXPECTED_PARENT_IDENTITIES), "R11_TOP24_ROSTER", "source roster drift")
    scores = [score_parent(grouped[identity], FROZEN_RULE) for identity in EXPECTED_PARENT_IDENTITIES]
    require(
        [row["frame_count"] for row in scores] == FROZEN_FRAME_COUNTS
        and sum(row["query_count"] for row in scores) == QUERY_COUNT,
        "R11_TOP24_FRAME_PLAN",
        "source frame plan drift",
    )
    ranked = rank_parent_scores(scores)
    common = {
        "source_zip_member_payload_reads": 0,
        "highres_depth_member_payload_reads": 0,
        "faro_reads": 0,
        "truth_reads": 0,
        "label_reads": 0,
        "outcome_reads": 0,
        "model_executions": 0,
        "training_steps": 0,
        "network_requests": 0,
    }
    parent_scores = validate_parent_scores(
        _seal(
            {
                "schema": PARENT_SCORES_SCHEMA,
                "selector": FROZEN_SELECTOR,
                "phase_a_completion_sha256": completion["content_sha256"],
                "source_frame_hash_sequence_sha256": completion["source_frame_hash_sequence_sha256"],
                "parent_count": PARENT_COUNT,
                "frame_count": FRAME_COUNT,
                "query_count": QUERY_COUNT,
                "parent_scores": scores,
                "ranked_parent_identities": [[row["parent_id"], row["video_id"]] for row in ranked],
                "all_48_source_records_sealed_before_scoring": True,
                "all_48_parent_scores_sealed_before_faro": True,
                **common,
            }
        )
    )
    selected = ranked[:SELECTED_PARENT_COUNT]
    selection = validate_selection(
        _seal(
            {
                "schema": SELECTION_SCHEMA,
                "selector": FROZEN_SELECTOR,
                "parent_scores_sha256": parent_scores["content_sha256"],
                "selected_parent_count": SELECTED_PARENT_COUNT,
                "selected_parent_identities": [[row["parent_id"], row["video_id"]] for row in selected],
                "selected_parent_scores": selected,
                "selection_sealed_before_faro": True,
                "read_unselected_faro": False,
                "source_reselection_after_faro": False,
                "parent_reselection_after_faro": False,
                "candidate_or_threshold_reselection_after_faro": False,
                "unknown_is_negative": False,
                **common,
            }
        ),
        parent_scores,
    )
    return parent_scores, selection


def _validate_phase_a_audit() -> dict[str, Any]:
    path = _repo_path(PHASE_A_AUDIT_RELATIVE)
    require(
        path.is_file() and materializer.sha256_file(path) == PHASE_A_AUDIT_FILE_SHA256,
        "R11_TOP24_PHASE_A_AUDIT_FILE",
        "repaired Phase A audit file drift",
    )
    require(
        {item.name for item in path.parent.iterdir()} == {path.name},
        "R11_TOP24_PHASE_A_AUDIT_ROOT",
        "repaired Phase A audit root file set drift",
    )
    audit = _validate_seal(
        _load_json(path),
        "blindassist.taro.o1r.r11_phase_a_validator_round12_audit.v1",
        "R11_TOP24_PHASE_A_AUDIT_SEAL",
    )
    validation = audit.get("original_validator_result", {})
    require(
        audit["content_sha256"] == PHASE_A_AUDIT_CONTENT_SHA256
        and audit.get("status") == PHASE_A_AUDIT_PASS
        and audit.get("execution_validity") == "VALID_WITH_POST_TERMINAL_NUMERIC_REPRESENTATION_REPAIR"
        and audit.get("scientific_terminal") == PHASE_A_PASS_TERMINAL
        and audit.get("same_sealed_phase_a_root") == PHASE_A_ROOT
        and audit.get("phase_a_root_modified") is False
        and audit.get("model_rerun") is False
        and audit.get("parent_scoring_performed") is False
        and audit.get("top24_selection_performed") is False
        and audit.get("next_gate") == "R11_SOURCE_ONLY_TOP24_IMPLEMENTATION_LOCK"
        and validation.get("passed") is True
        and validation.get("parent_count") == PARENT_COUNT
        and validation.get("frame_count") == FRAME_COUNT
        and validation.get("query_count") == QUERY_COUNT
        and validation.get("root_file_count") == PHASE_A_FINAL_FILE_COUNT
        and validation.get("terminal") == PHASE_A_PASS_TERMINAL
        and validation.get("highres_depth_member_payload_reads") == validation.get("faro_reads") == 0
        and validation.get("truth_reads") == 0
        and audit.get("highres_depth_member_payload_reads") == audit.get("faro_reads") == 0
        and audit.get("truth_reads") == audit.get("label_reads") == audit.get("outcome_reads") == 0
        and audit.get("model_rerun") is False
        and audit.get("training_steps") == audit.get("network_requests") == 0,
        "R11_TOP24_PHASE_A_AUDIT",
        "repaired Phase A independent PASS is not admitted",
    )
    return audit


def _validate_phase_a_terminal_light() -> tuple[dict[str, Any], dict[str, Any]]:
    terminal_path = _repo_path(PHASE_A_TERMINAL_RELATIVE)
    completion_path = _repo_path(PHASE_A_COMPLETION_RELATIVE)
    require(
        materializer.sha256_file(terminal_path) == PHASE_A_TERMINAL_FILE_SHA256
        and materializer.sha256_file(completion_path) == PHASE_A_COMPLETION_FILE_SHA256,
        "R11_TOP24_PHASE_A_FILE",
        "Phase A terminal/completion file drift",
    )
    terminal = _validate_seal(
        _load_json(terminal_path),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_terminal.v1",
        "R11_TOP24_PHASE_A_TERMINAL_SEAL",
    )
    completion = _validate_seal(
        _load_json(completion_path),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_completion.v1",
        "R11_TOP24_PHASE_A_COMPLETION_SEAL",
    )
    files = terminal.get("files")
    result = terminal.get("result", {})
    require(
        terminal["content_sha256"] == PHASE_A_TERMINAL_CONTENT_SHA256
        and completion["content_sha256"] == PHASE_A_COMPLETION_CONTENT_SHA256
        and terminal.get("terminal") == PHASE_A_PASS_TERMINAL
        and terminal.get("passed") is terminal.get("execution_valid") is True
        and terminal.get("file_count_before_terminal") == PHASE_A_PRE_TERMINAL_FILE_COUNT
        and isinstance(files, dict) and len(files) == PHASE_A_PRE_TERMINAL_FILE_COUNT
        and files.get("phase-a-completion.json", {}).get("sha256") == PHASE_A_COMPLETION_FILE_SHA256
        and result.get("phase_a_completion_sha256") == completion["content_sha256"]
        and result.get("r9_parent_scoring_performed") is False
        and result.get("top24_selection_performed") is False
        and completion.get("r9_parent_scoring_performed") is False
        and completion.get("top24_selection_performed") is False
        and completion.get("all_source_records_sealed_before_faro") is True
        and completion.get("all_r7_and_r11_records_sealed_before_parent_scoring") is True
        and completion.get("parent_count") == PARENT_COUNT
        and completion.get("frame_count") == FRAME_COUNT
        and completion.get("query_count") == QUERY_COUNT
        and completion.get("faro_reads") == completion.get("truth_reads") == 0
        and completion.get("highres_depth_member_payload_reads") == 0
        and completion.get("unknown_is_negative") is False
        and completion.get("training_steps") == completion.get("network_requests") == 0,
        "R11_TOP24_PHASE_A_TERMINAL",
        "Phase A terminal/completion is not admitted",
    )
    return terminal, completion


def _frame_rows() -> list[tuple[str, str, str]]:
    inventory = _validate_seal(
        _load_json(_repo_path(INVENTORY_RELATIVE)),
        "blindassist.taro.o1r.r11_fresh_pool_inventory.v1",
        "R11_TOP24_INVENTORY_SEAL",
    )
    parents = inventory.get("parents")
    require(
        inventory["content_sha256"] == INVENTORY_CONTENT_SHA256
        and inventory.get("parent_count") == PARENT_COUNT
        and inventory.get("exact_pose_bounded_frame_count") == FRAME_COUNT
        and isinstance(parents, list) and len(parents) == PARENT_COUNT,
        "R11_TOP24_INVENTORY",
        "frozen inventory identity/count drift",
    )
    rows: list[tuple[str, str, str]] = []
    counts: list[int] = []
    identities: list[tuple[str, str]] = []
    for parent in parents:
        identity = (str(parent.get("visit_id")), str(parent.get("video_id")))
        tokens = parent.get("frame_plan", {}).get("exact_timestamp_tokens")
        require(isinstance(tokens, list) and len(tokens) > 0, "R11_TOP24_FRAME_PLAN", "timestamp plan missing")
        identities.append(identity)
        counts.append(len(tokens))
        rows.extend((identity[0], identity[1], str(token)) for token in tokens)
    require(
        identities == list(EXPECTED_PARENT_IDENTITIES)
        and counts == FROZEN_FRAME_COUNTS
        and len(rows) == FRAME_COUNT,
        "R11_TOP24_FRAME_PLAN",
        "frozen parent/frame plan drift",
    )
    return rows


def _verify_phase_a_files(root: Path, terminal: Mapping[str, Any]) -> dict[str, int]:
    files = terminal.get("files")
    require(isinstance(files, dict) and len(files) == PHASE_A_PRE_TERMINAL_FILE_COUNT, "R11_TOP24_PHASE_A_FILES", "prior receipt count drift")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    require(actual == set(files) | {"terminal.json"}, "R11_TOP24_PHASE_A_ROOT_SET", "Phase A exact root set drift")
    total = 0
    for relative, receipt in files.items():
        path = materializer.safe_join(root, relative)
        require(
            isinstance(receipt, dict)
            and receipt.get("path") == relative
            and path.is_file()
            and path.stat().st_size == receipt.get("bytes")
            and materializer.sha256_file(path) == receipt.get("sha256"),
            "R11_TOP24_PHASE_A_PRIOR_FILE",
            f"Phase A prior file drift: {relative}",
        )
        total += path.stat().st_size
    require(total == terminal.get("bytes_before_terminal"), "R11_TOP24_PHASE_A_BYTES", "Phase A prior byte total drift")
    return {"phase_a_prior_file_validations": len(files), "phase_a_prior_bytes_validated": total}


def _source_relative(parent_id: str, video_id: str, token: str) -> str:
    return f"phase-a-sources/{parent_id}/{video_id}/{token}.json"


def _lineage_relative(parent_id: str, video_id: str, token: str) -> str:
    return f"phase-a-lineage/{parent_id}/{video_id}/{token}.json.gz"


def _load_phase_a_sources(
    root: Path,
    rows: Sequence[tuple[str, str, str]],
    completion: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sources: list[dict[str, Any]] = []
    for parent_id, video_id, token in rows:
        physical_frame_id = f"{video_id}:{token}"
        receipt = _validate_seal(
            _load_json(root / _source_relative(parent_id, video_id, token)),
            "blindassist.taro.o1r.r11_fresh_pool_source_frame_receipt.v1",
            "R11_TOP24_SOURCE_RECEIPT_SEAL",
        )
        lineage = _validate_seal(
            _load_json_gzip(root / _lineage_relative(parent_id, video_id, token)),
            "blindassist.taro.o1r.r11_fresh_pool_phase_a_lineage.v1",
            "R11_TOP24_LINEAGE_SEAL",
        )
        source = r7_canary.validate_source_frame_record(lineage.get("r7_source_frame_record"))
        base = r7_positive.validate_positive_occupancy_factor(lineage.get("r7_positive_factor_bundle"))
        candidate = abstention_candidate.validate_abstention_bundle(lineage.get("r11_abstention_bundle"))
        base_rows, candidate_rows = base["query_results"], candidate["query_results"]
        require(
            receipt.get("parent_id") == source.get("parent_id") == base.get("parent_id") == candidate.get("parent_id") == parent_id
            and receipt.get("video_id") == source.get("video_id") == base.get("video_id") == candidate.get("video_id") == video_id
            and receipt.get("timestamp_token") == source.get("timestamp_token") == base.get("timestamp_token") == candidate.get("timestamp_token") == token
            and receipt.get("physical_frame_id") == source.get("physical_frame_id") == base.get("physical_frame_id") == candidate.get("physical_frame_id") == physical_frame_id
            and lineage.get("physical_frame_id") == physical_frame_id
            and lineage.get("source_frame_receipt_sha256") == receipt["content_sha256"]
            and source.get("source_frame_receipt_sha256") == receipt["content_sha256"]
            and base.get("source_frame_record_sha256") == candidate.get("source_frame_record_sha256") == source["content_sha256"]
            and receipt.get("highres_depth_member_payload_read") is False
            and receipt.get("faro_payload_read") is False
            and receipt.get("truth_payload_read") is False
            and lineage.get("highres_depth_member_payload_read") is False
            and lineage.get("faro_payload_read") is False
            and lineage.get("truth_inputs") == 0,
            "R11_TOP24_SOURCE_LINEAGE",
            "source receipt/R7/R11 lineage drift",
        )
        require(
            all(
                base_row.get("query_id") == candidate_row.get("query_id")
                and base_row.get("grid_index") == candidate_row.get("grid_index") == index
                and (candidate_row.get("state") != "OCCUPIED_OBSERVED" or base_row.get("state") == "OCCUPIED_OBSERVED")
                for index, (base_row, candidate_row) in enumerate(zip(base_rows, candidate_rows, strict=True))
            ),
            "R11_TOP24_FACTOR_PAIR",
            "R11 factor is not an ordered subset of R7 positive",
        )
        sources.append(source)
    require(
        len(sources) == FRAME_COUNT
        and adapter.canonical_sha256([row["content_sha256"] for row in sources])
        == completion.get("source_frame_hash_sequence_sha256"),
        "R11_TOP24_SOURCE_SEQUENCE",
        "reloaded source record sequence drift",
    )
    return sources, {
        "phase_a_lineage_decodes": len(sources),
        "phase_a_source_receipt_decodes": len(sources),
        "source_frame_records_validated": len(sources),
        "query_features_scored": len(sources) * 9,
        "source_zip_member_payload_reads": 0,
        "highres_depth_member_payload_reads": 0,
        "faro_reads": 0,
        "truth_reads": 0,
        "label_reads": 0,
        "outcome_reads": 0,
        "model_executions": 0,
        "training_steps": 0,
        "network_requests": 0,
    }


def _git_bytes(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    require(
        completed.returncode == 0,
        "R11_TOP24_IMPLEMENTATION_BINDING",
        f"implementation commit lacks binding: {relative}",
    )
    return completed.stdout


def _verify_binding_rows(lock: Mapping[str, Any], implementation_commit: str) -> None:
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R11_TOP24_BINDINGS", "binding count drift")
    seen: set[str] = set()
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(
            set(row) == {"role", "path", "bytes", "sha256"}
            and isinstance(role, str) and role not in seen
            and EXPECTED_BINDINGS.get(role) == relative,
            "R11_TOP24_BINDING_ROW",
            "binding role/path drift",
        )
        path = _repo_path(relative)
        require(
            path.is_file() and path.stat().st_size == row["bytes"] and materializer.sha256_file(path) == row["sha256"],
            "R11_TOP24_BINDING_HASH",
            f"binding bytes/SHA drift: {relative}",
        )
        if role not in ARTIFACT_BINDING_ROLES:
            require(
                path.read_bytes() == _git_bytes(implementation_commit, relative),
                "R11_TOP24_IMPLEMENTATION_BINDING",
                f"implementation-commit binding drift: {relative}",
            )
        seen.add(role)
    require(seen == set(EXPECTED_BINDINGS), "R11_TOP24_BINDINGS", "binding role set drift")


def _validate_protocol_and_r9() -> None:
    protocol = _validate_seal(
        _load_json(_repo_path(EXPECTED_BINDINGS["R11_PROTOCOL"])),
        "blindassist.taro.o1r.r11_positive_occupancy_abstention_protocol_lock.v1",
        "R11_TOP24_PROTOCOL_SEAL",
    )
    selection = protocol.get("fresh_frontdoor", {}).get("selection")
    firewall = protocol.get("phase_firewall")
    require(
        protocol.get("lock_id")
        == "TARO_O1R_R11_POSITIVE_OCCUPANCY_ABSTENTION_AND_FRESH_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK"
        and protocol.get("fresh_frontdoor", {}).get("pool_parent_count") == PARENT_COUNT
        and protocol.get("fresh_frontdoor", {}).get("selected_parent_count") == SELECTED_PARENT_COUNT
        and selection
        == {
            "selector_id": FROZEN_SELECTOR_ID,
            "rule_id": FROZEN_RULE_ID,
            "score": "ELIGIBLE_QUERY_COUNT_DESCENDING",
            "tie_break": "CANONICAL_SHA256_PARENT_VIDEO_ASCENDING",
            "use": "PARENT_RANKING_ONLY",
        }
        and firewall
        == {
            "phase_a_faro_reads": 0,
            "seal_all_48_source_records_before_faro": True,
            "seal_all_48_parent_scores_before_faro": True,
            "seal_top24_before_faro": True,
            "read_unselected_faro": False,
            "source_reselection_after_faro": False,
            "parent_reselection_after_faro": False,
            "candidate_or_threshold_reselection_after_faro": False,
            "unknown_is_negative": False,
        },
        "R11_TOP24_PROTOCOL",
        "R11 protocol selector/firewall drift",
    )
    development = _load_json(_repo_path(EXPECTED_BINDINGS["R9_DEVELOPMENT_RESULT"]))
    require(
        development.get("schema") == "blindassist.taro.o1r.r9_clear_enrichment_development_result_record.v1"
        and development.get("execution_valid") is True
        and development.get("passed_old_development_target") is False
        and development.get("selector", {}).get("selector_id") == FROZEN_SELECTOR_ID
        and development.get("selector", {}).get("selector_sha256") == FROZEN_SELECTOR_CONTENT_SHA256
        and development.get("selector", {}).get("rule_id") == FROZEN_RULE_ID
        and development.get("promotion", {}).get("selector_to_fresh_cohort_selection") is True
        and development.get("promotion", {}).get("effectiveness") is False,
        "R11_TOP24_R9_DEVELOPMENT",
        "R9 development evidence does not admit selector-only ranking",
    )


def _commit_is_on_master(commit: str) -> bool:
    if not isinstance(commit, str) or len(commit) != 40:
        return False
    for ancestor in ("HEAD", "refs/remotes/origin/master"):
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, ancestor],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return False
    return True


def _validate_actual_argv() -> None:
    original_argv = [str(value) for value in getattr(sys, "orig_argv", [])]
    require("-m" in original_argv, "R11_TOP24_ARGV", "selection must use the frozen module-form argv")
    module_index = original_argv.index("-m")
    require(
        original_argv[module_index:] == EXPECTED_ARGV,
        "R11_TOP24_ARGV",
        "actual selection argv drift",
    )


def validate_execution_lock(path: Path) -> dict[str, Any]:
    expected_path = _repo_path(LOCK_RELATIVE)
    lock_path = path.resolve()
    require(lock_path == expected_path, "R11_TOP24_LOCK_PATH", "execution lock path drift")
    lock = _validate_seal(_load_json(lock_path), LOCK_SCHEMA, "R11_TOP24_LOCK_SEAL")
    require(
        lock.get("lock_id") == LOCK_ID
        and lock.get("status") == "AUTHORIZED_UNCONSUMED"
        and lock.get("consumed") is False
        and lock.get("argv") == EXPECTED_ARGV
        and lock.get("phase_a_root") == PHASE_A_ROOT
        and lock.get("phase_a_repaired_audit") == PHASE_A_AUDIT_RELATIVE
        and lock.get("output_root") == OUTPUT_ROOT
        and lock.get("overwrite") is False
        and lock.get("rerun") is False
        and lock.get("frozen_selector") == FROZEN_SELECTOR
        and lock.get("user_authority") == EXPECTED_USER_AUTHORITY
        and lock.get("execution_authority") == EXPECTED_AUTHORITY
        and lock.get("resource_budget") == EXPECTED_RESOURCE_BUDGET
        and lock.get("one_shot_policy") == EXPECTED_ONE_SHOT_POLICY
        and lock.get("phase_a_terminal_content_sha256") == PHASE_A_TERMINAL_CONTENT_SHA256
        and lock.get("phase_a_audit_content_sha256") == PHASE_A_AUDIT_CONTENT_SHA256
        and lock.get("implementation_on_origin_master") is True
        and _commit_is_on_master(lock.get("implementation_commit")),
        "R11_TOP24_LOCK_POLICY",
        "execution lock identity/policy/authority drift",
    )
    _validate_actual_argv()
    _verify_binding_rows(lock, lock["implementation_commit"])
    _validate_protocol_and_r9()
    _validate_phase_a_audit()
    _validate_phase_a_terminal_light()
    validate_frozen_selector(_load_json(_repo_path(EXPECTED_BINDINGS["R9_SELECTOR_ARTIFACT"])))
    require(not _repo_path(OUTPUT_ROOT).exists(), "R11_TOP24_ROOT_COLLISION", "formal selection root exists")
    lock["_lock_path"] = lock_path
    return lock


def _resource_snapshot(process: psutil.Process, started: float, budget: Mapping[str, int], reserve_seconds: int = 0) -> dict[str, Any]:
    elapsed = time.monotonic() - started
    peak_rss = getattr(process.memory_info(), "peak_wset", None)
    require(isinstance(peak_rss, int) and peak_rss > 0, "R11_TOP24_PEAK_RSS", "OS peak RSS unavailable")
    require(elapsed + reserve_seconds <= budget["maximum_wall_seconds"], "R11_TOP24_TIMEOUT", "wall budget exceeded")
    require(peak_rss <= budget["maximum_peak_rss_bytes"], "R11_TOP24_RSS", "peak RSS budget exceeded")
    return {"elapsed_seconds": round(float(elapsed), 6), "peak_rss_bytes": peak_rss}


def _allocate_terminal_reserve(writer: FactorEvidenceWriter) -> None:
    reserve = writer.root / TERMINAL_RESERVE_NAME
    require(not reserve.exists(), "R11_TOP24_RESERVE_COLLISION", "terminal reserve exists")
    with reserve.open("xb") as stream:
        stream.write(bytes(TERMINAL_RESERVE_BYTES))
        stream.flush()
        os.fsync(stream.fileno())
    require(reserve.stat().st_size == TERMINAL_RESERVE_BYTES, "R11_TOP24_RESERVE", "terminal reserve size drift")


def _release_terminal_reserve(writer: FactorEvidenceWriter) -> None:
    reserve = writer.root / TERMINAL_RESERVE_NAME
    reserve.unlink(missing_ok=True)
    require(not reserve.exists(), "R11_TOP24_RESERVE_RELEASE", "terminal reserve release failed")


def _adopt_partials(writer: FactorEvidenceWriter) -> None:
    for partial in sorted(writer.root.rglob("*.partial")):
        target = partial
        if partial == writer.root / "terminal.json.partial":
            target = writer.root / "invalid-terminal-write.partial"
            require(not target.exists(), "R11_TOP24_TERMINAL_PARTIAL", "terminal partial collision")
            partial.replace(target)
        relative = target.relative_to(writer.root).as_posix()
        require(relative not in writer.file_receipts, "R11_TOP24_PARTIAL_RECEIPT", "partial receipt collision")
        payload = target.read_bytes()
        writer.file_receipts[relative] = {
            "path": relative,
            "bytes": len(payload),
            "sha256": materializer.sha256_bytes(payload),
        }
        writer.bytes_written += len(payload)


def _failure_terminal(writer: FactorEvidenceWriter, error: BaseException) -> dict[str, Any]:
    failure = _seal(
        {
            "schema": FAILURE_SCHEMA,
            "terminal": FAIL_TERMINAL,
            "passed": False,
            "execution_valid": False,
            "failure_code": str(getattr(error, "code", type(error).__name__))[:256],
            "message": str(error)[:FAILURE_MESSAGE_MAX_CHARS],
            "one_shot_consumed": True,
        }
    )
    return _seal(
        {
            "schema": TERMINAL_SCHEMA,
            "terminal": FAIL_TERMINAL,
            "passed": False,
            "execution_valid": False,
            "result": failure,
            "files": dict(sorted(writer.file_receipts.items())),
            "file_count_before_terminal": len(writer.file_receipts),
            "bytes_before_terminal": writer.bytes_written,
            "one_shot_consumed": True,
        }
    )


def _write_failure(writer: FactorEvidenceWriter, error: BaseException) -> None:
    if not writer.activated:
        return
    _release_terminal_reserve(writer)
    _adopt_partials(writer)
    require(not (writer.root / "terminal.json").exists(), "R11_TOP24_TERMINAL_COLLISION", "terminal already exists")
    writer.maximum_bytes = EXPECTED_RESOURCE_BUDGET["maximum_evidence_bytes"]
    writer.write_json("terminal.json", _failure_terminal(writer, error))


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    budget = lock["resource_budget"]
    process = psutil.Process(os.getpid())
    started = time.monotonic()
    writer = FactorEvidenceWriter(
        _repo_path(OUTPUT_ROOT),
        budget["maximum_evidence_bytes"] - TERMINAL_RESERVE_BYTES,
    )
    try:
        writer.activate(
            _seal(
                {
                    "schema": "blindassist.taro.o1r.r11_fresh_pool_top24_selection_execution_receipt.v1",
                    "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]),
                    "execution_lock_content_sha256": lock["content_sha256"],
                    "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "phase_a_root": PHASE_A_ROOT,
                    "phase_a_audit_content_sha256": PHASE_A_AUDIT_CONTENT_SHA256,
                    "frozen_selector": FROZEN_SELECTOR,
                    "one_shot_consumed_on_root_creation": True,
                    **{key: 0 for key in (
                        "source_zip_member_payload_reads", "highres_depth_member_payload_reads", "faro_reads",
                        "truth_reads", "label_reads", "outcome_reads", "model_executions", "training_steps",
                        "network_requests",
                    )},
                }
            )
        )
        _allocate_terminal_reserve(writer)
        terminal, completion = _validate_phase_a_terminal_light()
        root = _repo_path(PHASE_A_ROOT)
        ledger = _verify_phase_a_files(root, terminal)
        rows = _frame_rows()
        sources, source_ledger = _load_phase_a_sources(root, rows, completion)
        parent_scores, selection = build_selection(completion, sources)
        writer.write_json("parent-scores.json", parent_scores)
        writer.write_json("selection.json", selection)
        require(len(writer.file_receipts) == SUCCESS_PRE_TERMINAL_FILE_COUNT, "R11_TOP24_FILE_COUNT", "success pre-terminal count drift")
        resource = _resource_snapshot(process, started, budget, TERMINAL_WALL_RESERVE_SECONDS)
        result = _seal(
            {
                "schema": RESULT_SCHEMA,
                "terminal": PASS_TERMINAL,
                "passed": True,
                "execution_valid": True,
                "parent_count": PARENT_COUNT,
                "frame_count": FRAME_COUNT,
                "query_count": QUERY_COUNT,
                "selected_parent_count": SELECTED_PARENT_COUNT,
                "selected_parent_identities": selection["selected_parent_identities"],
                "phase_a_terminal_content_sha256": PHASE_A_TERMINAL_CONTENT_SHA256,
                "phase_a_completion_sha256": completion["content_sha256"],
                "phase_a_audit_content_sha256": PHASE_A_AUDIT_CONTENT_SHA256,
                "parent_scores_sha256": parent_scores["content_sha256"],
                "selection_sha256": selection["content_sha256"],
                "all_48_source_records_sealed_before_scoring": True,
                "all_48_parent_scores_sealed_before_faro": True,
                "selection_sealed_before_faro": True,
                "unknown_is_negative": False,
                **ledger,
                **source_ledger,
                **resource,
                "resource_budget": dict(budget),
                "one_shot_consumed": True,
                "unique_successor": "TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK",
                "claim_ceiling": (
                    "Source-only R11 scores and a sealed top-24 parent identity; no FARO label, task "
                    "effectiveness, training, deployment, product or safety evidence."
                ),
            }
        )
        terminal_record = _seal(
            {
                "schema": TERMINAL_SCHEMA,
                "terminal": PASS_TERMINAL,
                "passed": True,
                "execution_valid": True,
                "result": result,
                "files": dict(sorted(writer.file_receipts.items())),
                "file_count_before_terminal": len(writer.file_receipts),
                "bytes_before_terminal": writer.bytes_written,
                "one_shot_consumed": True,
            }
        )
        terminal_bytes = len(adapter.canonical_json_bytes(terminal_record)) + 1
        projected_terminal_receipt = {
            "path": "invalid-terminal-write.partial",
            "bytes": terminal_bytes,
            "sha256": "F" * 64,
        }
        projected_files = dict(writer.file_receipts)
        projected_files[projected_terminal_receipt["path"]] = projected_terminal_receipt
        projected_writer = copy.copy(writer)
        projected_writer.file_receipts = projected_files
        projected_writer.bytes_written = writer.bytes_written + terminal_bytes
        projected = _failure_terminal(
            projected_writer,
            FreshTop24SelectionError("X" * 256, "X" * FAILURE_MESSAGE_MAX_CHARS),
        )
        require(
            terminal_bytes + len(adapter.canonical_json_bytes(projected)) + 1 <= TERMINAL_RESERVE_BYTES
            and writer.bytes_written + terminal_bytes <= budget["maximum_evidence_bytes"],
            "R11_TOP24_TERMINAL_RESERVE",
            "terminal exceeds reserved budget",
        )
        _resource_snapshot(process, started, budget, TERMINAL_WALL_RESERVE_SECONDS)
        _release_terminal_reserve(writer)
        writer.maximum_bytes = budget["maximum_evidence_bytes"]
        require(
            set(writer.file_receipts) == {"execution-receipt.json", "parent-scores.json", "selection.json"},
            "R11_TOP24_FINAL_ROOT",
            "success pre-terminal exact file set drift",
        )
        writer.write_json("terminal.json", terminal_record)
        return result
    except Exception as error:
        try:
            _write_failure(writer, error)
        except Exception as failure_error:  # noqa: BLE001 - seal every failure
            raise FreshTop24SelectionError(
                "R11_TOP24_FAILURE_SEAL_FAILED",
                f"selection failed and failure terminal could not be sealed: {failure_error}",
            ) from error
        raise


def assert_public_api_source_only() -> None:
    for function in (score_parent, rank_parent_scores, build_selection):
        names = inspect.signature(function).parameters
        require(
            not any(token in name.lower() for name in names for token in ("faro", "highres", "truth", "label", "outcome", "model")),
            "R11_TOP24_RESULT_SIDE_API",
            "selector public API has a result-side parameter",
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:  # noqa: BLE001 - render every fail-closed error
        print(
            json.dumps(
                {
                    "terminal": FAIL_TERMINAL,
                    "failure_code": str(getattr(error, "code", type(error).__name__)),
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
