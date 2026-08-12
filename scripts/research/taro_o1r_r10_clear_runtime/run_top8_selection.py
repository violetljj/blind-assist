#!/usr/bin/env python3
"""Seal the R10 source-only 32-to-8 selection before any FARO access."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import gzip
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r9_clear_runtime import clear_enrichment_fit
from scripts.research.taro_o1r_r10_clear_runtime import fresh_pool
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_phase_a as phase_a
from scripts.research.taro_o1r_r10_clear_runtime import run_pool_phase_a_r1 as phase_a_r1


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r10_fresh_pool_top8_selection_execution_lock.v1"
LOCK_ID = "TARO_O1R_R10_FRESH_POOL_TOP8_SOURCE_ONLY_SELECTION_ONE_SHOT_EXECUTION_LOCK"
PHASE_A_ROOT = phase_a_r1.OUTPUT_ROOT
INVENTORY_PATH = phase_a.INVENTORY_PATH
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r10-fresh-pool-top8-selection-r0"
PASS_TERMINAL = "TARO_O1R_R10_FRESH_POOL_TOP8_SOURCE_ONLY_SELECTION_SEALED_PASS"
FAIL_TERMINAL = "TARO_O1R_R10_FRESH_POOL_TOP8_SELECTION_EXECUTION_INVALID"

PARENT_COUNT = phase_a.PARENT_COUNT
SELECTED_PARENT_COUNT = clear_enrichment_fit.SELECTED_PARENT_COUNT
FRAME_COUNT = phase_a.FRAME_COUNT
QUERY_COUNT = phase_a.QUERY_COUNT
PHASE_A_FILE_COUNT = 5 * FRAME_COUNT + 4
SELECTION_FILE_COUNT = 4

FROZEN_SELECTOR_ID = "TARO_R9_SOURCE_ONLY_CLEAR_ENRICHMENT_GRID_SEARCH_V1"
FROZEN_SELECTOR_CONTENT_SHA256 = "67FD8430418E23E4C974EBA4D7F49DCBD4DE66164A16491DE76F05AC974796CC"
FROZEN_RULE_ID = "02CE016D6B0011F0"
FROZEN_RULE = {
    "state_policy": "UNKNOWN_ONLY",
    "minimum_far_valid_anchor_count": 6,
    "maximum_far_valid_anchor_count": 1000000,
    "far_fraction_index": 0,
    "maximum_far_fraction": 0.0,
    "minimum_observed_support_points": 0,
    "require_query_receipt": True,
    "require_positive_obstacle_veto_false": True,
    "require_all_occupied_hits_false": True,
    "rule_id": FROZEN_RULE_ID,
}
FROZEN_PROTOCOL_SELECTOR = {
    "selector_id": FROZEN_SELECTOR_ID,
    "selector_sha256": FROZEN_SELECTOR_CONTENT_SHA256,
    "rule_id": FROZEN_RULE_ID,
    **{key: value for key, value in FROZEN_RULE.items() if key != "rule_id"},
}
FROZEN_LOCK_SELECTOR = {
    "selector_id": FROZEN_SELECTOR_ID,
    "selector_content_sha256": FROZEN_SELECTOR_CONTENT_SHA256,
    "rule_id": FROZEN_RULE_ID,
}

EXPECTED_PARENT_IDENTITIES = tuple((visit, video) for visit, video, _ in fresh_pool.EXPECTED_POOL)
EXPECTED_BINDINGS = {
    "R10_PROTOCOL": "docs/research/taro/TARO_O1R_R10_FRESH_PARENT_SOURCE_ONLY_CLEAR_ENRICHED_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json",
    "R9_DEVELOPMENT_RESULT": "docs/research/taro/TARO_O1R_R9_CLEAR_ENRICHMENT_DEVELOPMENT_RESULT_2026-08-12.json",
    "R9_FROZEN_SELECTOR": "artifacts.local/evidence/taro/o1r-r9-clear-enrichment-development-r0/selector.json",
    "R9_SELECTOR_RUNTIME": "scripts/research/taro_o1r_r9_clear_runtime/clear_enrichment_fit.py",
    "R10_POOL_PLANNER": "scripts/research/taro_o1r_r10_clear_runtime/fresh_pool.py",
    "R10_INVENTORY_PLAN": INVENTORY_PATH,
    "R10_PHASE_A_R1_LOCK": "docs/research/taro/TARO_O1R_R10_FRESH_POOL_SOURCE_ONLY_PHASE_A_R1_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json",
    "R10_PHASE_A_R1_RESULT": f"{PHASE_A_ROOT}/result.json",
    "R10_PHASE_A_R1_COMPLETION": f"{PHASE_A_ROOT}/phase-a-completion.json",
    "R10_PHASE_A_R1_MANIFEST": f"{PHASE_A_ROOT}/manifest.json",
    "R10_PHASE_A_BASE_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_phase_a.py",
    "R10_PHASE_A_R1_RUNTIME": "scripts/research/taro_o1r_r10_clear_runtime/run_pool_phase_a_r1.py",
    "SOURCE_RECORD_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "R10_SELECTION_RUNNER": "scripts/research/taro_o1r_r10_clear_runtime/run_top8_selection.py",
}
EXPECTED_AUTHORITY = {
    "sealed_phase_a_reload": True,
    "source_only_parent_scoring": True,
    "top8_selection": True,
    "faro_read": False,
    "truth_read": False,
    "label_read": False,
    "outcome_read": False,
    "candidate_rerun": False,
    "threshold_fit": False,
    "training": False,
    "network": False,
    "device": False,
    "product": False,
    "safety": False,
}
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "先推动，我授权",
    "scope": phase_a.AUTHORITY_SCOPE,
}

PARENT_SCORES_SCHEMA = "blindassist.taro.o1r.r10_fresh_pool_source_only_parent_scores.v1"
SELECTION_SCHEMA = "blindassist.taro.o1r.r10_fresh_pool_top8_source_only_selection.v1"


class FreshTop8SelectionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise FreshTop8SelectionError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R10_SELECTION_SEAL_COLLISION", "selection caller supplied a content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Mapping[str, Any], schema: str, code: str) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema
        and isinstance(observed, str)
        and adapter.canonical_sha256(record) == observed,
        code,
        "sealed selection record hash/schema drift",
    )
    record["content_sha256"] = observed
    return record


def validate_frozen_rule(value: Mapping[str, Any]) -> dict[str, Any]:
    rule = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require(rule == FROZEN_RULE, "R10_SELECTION_RULE_DRIFT", "R10 selector rule differs from the frozen R9 rule")
    payload = {key: item for key, item in rule.items() if key != "rule_id"}
    require(adapter.canonical_sha256(payload)[:16] == FROZEN_RULE_ID, "R10_SELECTION_RULE_ID_DRIFT", "R10 selector rule id drift")
    return rule


def validate_frozen_selector(value: Mapping[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(
        observed == FROZEN_SELECTOR_CONTENT_SHA256
        and adapter.canonical_sha256(record) == FROZEN_SELECTOR_CONTENT_SHA256,
        "R10_SELECTION_SELECTOR_SEAL_DRIFT",
        "frozen R9 selector content seal drift",
    )
    record["content_sha256"] = observed
    validated = clear_enrichment_fit.validate_selector(record)
    require(
        validated.get("selector_id") == FROZEN_SELECTOR_ID
        and validate_frozen_rule(validated.get("chosen_rule", {})) == FROZEN_RULE,
        "R10_SELECTION_SELECTOR_IDENTITY_DRIFT",
        "frozen R9 selector identity/rule drift",
    )
    return validated


def validate_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    protocol = copy.deepcopy(dict(value))
    firewall = protocol.get("phase_firewall")
    require(
        protocol.get("schema") == "blindassist.taro.o1r.r10_fresh_clear_enriched_confirmation_protocol_lock.v1"
        and protocol.get("lock_id") == "TARO_O1R_R10_FRESH_PARENT_SOURCE_ONLY_CLEAR_ENRICHED_CONFIRMATION_PROTOCOL_LOCK"
        and protocol.get("status") == "LOCKED_PRE_NETWORK_PRE_SOURCE_PRE_OUTCOME"
        and protocol.get("fresh_pool", {}).get("parent_count") == PARENT_COUNT
        and protocol.get("frozen_selector") == FROZEN_PROTOCOL_SELECTOR,
        "R10_SELECTION_PROTOCOL_DRIFT",
        "R10 protocol/selector binding drift",
    )
    require(
        firewall == {
            "seal_all_32_source_scores_before_faro": True,
            "seal_top8_before_faro": True,
            "read_unselected_faro": False,
            "source_reselection_after_faro": False,
            "threshold_reselection_after_faro": False,
            "unknown_is_negative": False,
        },
        "R10_SELECTION_PROTOCOL_FIREWALL_DRIFT",
        "R10 protocol selection firewall drift",
    )
    return protocol


def _score_tie(parent_id: str, video_id: str) -> str:
    """The exact tie break used by the frozen R9 fitting runtime."""

    return adapter.canonical_sha256([parent_id, video_id])


def score_parent(
    source_frame_records: Sequence[Mapping[str, Any]],
    rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one parent from source records only; this API accepts no result-side input."""

    frozen_rule = validate_frozen_rule(rule)
    require(bool(source_frame_records), "R10_SELECTION_PARENT_EMPTY", "parent source records are empty")
    rows = sorted(
        (dict(row) for row in source_frame_records),
        key=lambda row: (str(row.get("timestamp_token")), str(row.get("physical_frame_id"))),
    )
    parent_id = str(rows[0].get("parent_id"))
    video_id = str(rows[0].get("video_id"))
    require(
        all(str(row.get("parent_id")) == parent_id and str(row.get("video_id")) == video_id for row in rows),
        "R10_SELECTION_PARENT_IDENTITY_DRIFT",
        "source records mix parent identities",
    )
    available = 0
    eligible = 0
    source_hashes: list[str] = []
    for row in rows:
        require(
            row.get("source_phase_has_label_input") in {None, False}
            and row.get("training_steps", 0) == 0
            and row.get("network_requests", 0) == 0,
            "R10_SELECTION_SOURCE_FIREWALL_DRIFT",
            "source record crosses the selection firewall",
        )
        source_hash = row.get("content_sha256")
        features = row.get("query_features")
        require(
            isinstance(source_hash, str) and isinstance(features, list) and len(features) == 9,
            "R10_SELECTION_SOURCE_RECORD_INVALID",
            "source record hash/query cardinality drift",
        )
        source_hashes.append(source_hash)
        for feature in features:
            available += feature.get("query_receipt") is not None
            eligible += clear_enrichment_fit.eligible(feature, frozen_rule)
    return {
        "selector_id": FROZEN_SELECTOR_ID,
        "selector_content_sha256": FROZEN_SELECTOR_CONTENT_SHA256,
        "rule_id": FROZEN_RULE_ID,
        "parent_id": parent_id,
        "video_id": video_id,
        "frame_count": len(rows),
        "query_count": len(rows) * 9,
        "available_query_count": int(available),
        "eligible_query_count": int(eligible),
        "eligible_fraction_of_available": float(eligible / available) if available else 0.0,
        "source_frame_hash_sequence_sha256": adapter.canonical_sha256(source_hashes),
        "tie_break_sha256": _score_tie(parent_id, video_id),
        "faro_reads": 0,
        "truth_reads": 0,
        "label_reads": 0,
        "outcome_reads": 0,
        "clear_output_emitted": False,
        "training_steps": 0,
        "network_requests": 0,
    }


_SCORE_FIELDS = {
    "selector_id",
    "selector_content_sha256",
    "rule_id",
    "parent_id",
    "video_id",
    "frame_count",
    "query_count",
    "available_query_count",
    "eligible_query_count",
    "eligible_fraction_of_available",
    "source_frame_hash_sequence_sha256",
    "tie_break_sha256",
    "faro_reads",
    "truth_reads",
    "label_reads",
    "outcome_reads",
    "clear_output_emitted",
    "training_steps",
    "network_requests",
}


def rank_parent_scores(parent_scores: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(parent_scores) == PARENT_COUNT, "R10_SELECTION_PARENT_COUNT", "R10 selection requires exactly 32 parent scores")
    scores = [copy.deepcopy(dict(row)) for row in parent_scores]
    identities = [(str(row.get("parent_id")), str(row.get("video_id"))) for row in scores]
    require(len(set(identities)) == PARENT_COUNT, "R10_SELECTION_PARENT_DUPLICATE", "R10 parent score identities are not unique")
    for row, identity in zip(scores, identities, strict=True):
        expected_fraction = (
            float(row["eligible_query_count"] / row["available_query_count"])
            if row.get("available_query_count")
            else 0.0
        )
        require(
            set(row) == _SCORE_FIELDS
            and row["selector_id"] == FROZEN_SELECTOR_ID
            and row["selector_content_sha256"] == FROZEN_SELECTOR_CONTENT_SHA256
            and row["rule_id"] == FROZEN_RULE_ID
            and row["query_count"] == 9 * row["frame_count"]
            and 0 <= row["eligible_query_count"] <= row["available_query_count"] <= row["query_count"]
            and row["eligible_fraction_of_available"] == expected_fraction
            and isinstance(row["source_frame_hash_sequence_sha256"], str)
            and len(row["source_frame_hash_sequence_sha256"]) == 64
            and row["tie_break_sha256"] == _score_tie(*identity)
            and row["faro_reads"] == row["truth_reads"] == row["label_reads"] == row["outcome_reads"] == 0
            and row["clear_output_emitted"] is False
            and row["training_steps"] == row["network_requests"] == 0,
            "R10_SELECTION_PARENT_SCORE_INVALID",
            "R10 parent score fields, selector binding, or firewall drift",
        )
    return sorted(scores, key=lambda row: (-int(row["eligible_query_count"]), str(row["tie_break_sha256"])))


def validate_parent_scores(value: Mapping[str, Any]) -> dict[str, Any]:
    record = _validate_seal(value, PARENT_SCORES_SCHEMA, "R10_SELECTION_SCORES_HASH_DRIFT")
    parent_scores = record.get("parent_scores", [])
    ranked = rank_parent_scores(parent_scores)
    expected_ranked = [[row["parent_id"], row["video_id"]] for row in ranked]
    require(
        record.get("selector") == FROZEN_LOCK_SELECTOR
        and record.get("parent_count") == PARENT_COUNT
        and record.get("frame_count") == FRAME_COUNT
        and record.get("query_count") == QUERY_COUNT
        and [(row["parent_id"], row["video_id"]) for row in parent_scores]
        == list(EXPECTED_PARENT_IDENTITIES)
        and [row["frame_count"] for row in parent_scores] == phase_a.FROZEN_FRAME_COUNTS
        and record.get("ranked_parent_identities") == expected_ranked
        and record.get("all_32_scores_sealed_before_faro") is True
        and record.get("faro_reads") == record.get("truth_reads") == record.get("label_reads") == record.get("outcome_reads") == 0
        and record.get("clear_output_emitted") is False
        and record.get("training_steps") == record.get("network_requests") == 0,
        "R10_SELECTION_SCORES_INVALID",
        "R10 sealed parent scores drift",
    )
    return record


def validate_selection(value: Mapping[str, Any], parent_scores: Mapping[str, Any]) -> dict[str, Any]:
    scores_record = validate_parent_scores(parent_scores)
    record = _validate_seal(value, SELECTION_SCHEMA, "R10_SELECTION_HASH_DRIFT")
    ranked = rank_parent_scores(scores_record["parent_scores"])
    expected_selected = ranked[:SELECTED_PARENT_COUNT]
    require(
        record.get("selector") == FROZEN_LOCK_SELECTOR
        and record.get("parent_scores_sha256") == scores_record["content_sha256"]
        and record.get("selected_parent_count") == SELECTED_PARENT_COUNT
        and record.get("selected_parent_identities")
        == [[row["parent_id"], row["video_id"]] for row in expected_selected]
        and record.get("selected_parent_scores") == expected_selected
        and record.get("selection_sealed_before_faro") is True
        and record.get("source_reselection_after_faro") is False
        and record.get("threshold_reselection_after_faro") is False
        and record.get("unknown_is_negative") is False
        and record.get("faro_reads") == record.get("truth_reads") == record.get("label_reads") == record.get("outcome_reads") == 0
        and record.get("clear_output_emitted") is False
        and record.get("training_steps") == record.get("network_requests") == 0,
        "R10_SELECTION_RESULT_INVALID",
        "R10 sealed top-eight selection drift",
    )
    return record


def build_selection(
    completion: Mapping[str, Any],
    source_frame_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(len(source_frame_records) == FRAME_COUNT, "R10_SELECTION_SOURCE_COUNT", "R10 source record count drift")
    require(
        adapter.canonical_sha256([row["content_sha256"] for row in source_frame_records])
        == completion.get("source_frame_hash_sequence_sha256"),
        "R10_SELECTION_SOURCE_SEQUENCE_DRIFT",
        "R10 source hash sequence differs from Phase A completion",
    )
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for source in source_frame_records:
        grouped[(str(source["parent_id"]), str(source["video_id"]))].append(source)
    require(
        set(grouped) == set(EXPECTED_PARENT_IDENTITIES),
        "R10_SELECTION_ROSTER_DRIFT",
        "R10 source parent roster differs from the frozen pool",
    )
    scores = [score_parent(grouped[identity], FROZEN_RULE) for identity in EXPECTED_PARENT_IDENTITIES]
    require(
        [row["frame_count"] for row in scores] == phase_a.FROZEN_FRAME_COUNTS
        and sum(row["frame_count"] for row in scores) == FRAME_COUNT
        and sum(row["query_count"] for row in scores) == QUERY_COUNT,
        "R10_SELECTION_FRAME_PLAN_DRIFT",
        "R10 source parent frame plan drift",
    )
    ranked = rank_parent_scores(scores)
    scores_record = validate_parent_scores(
        _seal(
            {
                "schema": PARENT_SCORES_SCHEMA,
                "selector": FROZEN_LOCK_SELECTOR,
                "phase_a_completion_sha256": completion["content_sha256"],
                "source_frame_hash_sequence_sha256": completion["source_frame_hash_sequence_sha256"],
                "parent_count": PARENT_COUNT,
                "frame_count": FRAME_COUNT,
                "query_count": QUERY_COUNT,
                "parent_scores": scores,
                "ranked_parent_identities": [[row["parent_id"], row["video_id"]] for row in ranked],
                "all_32_scores_sealed_before_faro": True,
                "faro_reads": 0,
                "truth_reads": 0,
                "label_reads": 0,
                "outcome_reads": 0,
                "clear_output_emitted": False,
                "training_steps": 0,
                "network_requests": 0,
            }
        )
    )
    selected = ranked[:SELECTED_PARENT_COUNT]
    selection = validate_selection(
        _seal(
            {
                "schema": SELECTION_SCHEMA,
                "selector": FROZEN_LOCK_SELECTOR,
                "parent_scores_sha256": scores_record["content_sha256"],
                "selected_parent_count": SELECTED_PARENT_COUNT,
                "selected_parent_identities": [[row["parent_id"], row["video_id"]] for row in selected],
                "selected_parent_scores": selected,
                "selection_sealed_before_faro": True,
                "source_reselection_after_faro": False,
                "threshold_reselection_after_faro": False,
                "unknown_is_negative": False,
                "faro_reads": 0,
                "truth_reads": 0,
                "label_reads": 0,
                "outcome_reads": 0,
                "clear_output_emitted": False,
                "training_steps": 0,
                "network_requests": 0,
            }
        ),
        scores_record,
    )
    return scores_record, selection


def _expected_phase_a_paths(frames: Sequence[Any]) -> set[str]:
    paths = {"execution-receipt.json", "candidate-completion.json", "phase-a-completion.json", "result.json"}
    for frame in frames:
        paths.update(
            {
                phase_a._candidate_input_relative(frame),
                phase_a._candidate_blob_relative(frame),
                phase_a._candidate_record_relative(frame),
                phase_a._source_receipt_relative(frame),
                phase_a._lineage_relative(frame),
            }
        )
    require(len(paths) == PHASE_A_FILE_COUNT, "R10_SELECTION_EXPECTED_PATH_COUNT", "R10 Phase-A expected path count drift")
    return paths


def verify_phase_a_manifest(root: Path, frames: Sequence[Any]) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    files = manifest.get("files")
    expected_paths = _expected_phase_a_paths(frames)
    require(
        manifest.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_phase_a_manifest.v1"
        and manifest.get("terminal") == phase_a_r1.PASS_TERMINAL
        and isinstance(files, dict)
        and len(files) == manifest.get("file_count_before_manifest") == PHASE_A_FILE_COUNT
        and set(files) == expected_paths,
        "R10_SELECTION_PHASE_A_MANIFEST",
        "R10 Phase-A R1 manifest schema, terminal, cardinality, or file set drift",
    )
    total = 0
    for relative, receipt in files.items():
        target = materializer.safe_join(root, relative)
        require(
            isinstance(receipt, dict)
            and receipt.get("path") == relative
            and target.is_file()
            and target.stat().st_size == receipt.get("bytes")
            and materializer.sha256_file(target) == receipt.get("sha256"),
            "R10_SELECTION_PHASE_A_FILE_DRIFT",
            f"R10 Phase-A artifact drift: {relative}",
        )
        total += target.stat().st_size
    require(total == manifest.get("bytes_before_manifest"), "R10_SELECTION_PHASE_A_BYTE_DRIFT", "R10 Phase-A byte total drift")
    return manifest


def load_phase_a_sources(root: Path, frames: Sequence[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completion = phase_a._validate_seal(
        _read_json(root / "phase-a-completion.json"),
        "blindassist.taro.o1r.r10_fresh_pool_phase_a_completion.v1",
    )
    require(
        completion.get("parent_count") == PARENT_COUNT
        and completion.get("frame_count") == FRAME_COUNT
        and completion.get("query_count") == QUERY_COUNT
        and completion.get("candidate_payload_reads") == {"color": FRAME_COUNT, "intrinsics": FRAME_COUNT}
        and completion.get("source_payload_reads") == {"confidence": FRAME_COUNT, "lowres_depth": FRAME_COUNT}
        and completion.get("faro_reads") == completion.get("truth_reads") == 0
        and completion.get("clear_output_allowed") is False
        and completion.get("all_source_records_sealed_before_faro") is True
        and completion.get("training_steps") == completion.get("network_requests") == 0,
        "R10_SELECTION_PHASE_A_COMPLETION",
        "R10 Phase-A R1 completion is not source-only or exact 32/710/6390",
    )
    sources: list[dict[str, Any]] = []
    for frame in frames:
        source_receipt = phase_a._validate_seal(
            _read_json(root / phase_a._source_receipt_relative(frame)),
            "blindassist.taro.o1r.r10_fresh_pool_source_frame_receipt.v1",
        )
        lineage = _read_gzip_json(root / phase_a._lineage_relative(frame))
        require(
            isinstance(lineage, dict)
            and set(lineage) == {"prospective_bundle", "r6_reducer_bundle", "r7_source_frame_record"},
            "R10_SELECTION_LINEAGE_SCHEMA_DRIFT",
            "R10 source lineage file set drift",
        )
        source = r7_canary.validate_source_frame_record(lineage["r7_source_frame_record"])
        require(
            source["physical_frame_id"] == frame.physical_frame_id
            and source["parent_id"] == frame.parent_id
            and source["video_id"] == frame.video_id
            and source["timestamp_token"] == frame.timestamp_token
            and source["source_frame_receipt_sha256"] == source_receipt["content_sha256"]
            and source_receipt["faro_payload_read"] is False
            and source_receipt["truth_payload_read"] is False,
            "R10_SELECTION_SOURCE_LINEAGE_DRIFT",
            "R10 source receipt/record lineage drift",
        )
        sources.append(source)
    require(
        adapter.canonical_sha256([row["content_sha256"] for row in sources])
        == completion["source_frame_hash_sequence_sha256"],
        "R10_SELECTION_SOURCE_SEQUENCE_DRIFT",
        "R10 source sequence seal drift",
    )
    return completion, sources


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = _read_json(lock_path)
    require(
        lock.get("schema") == LOCK_SCHEMA
        and lock.get("lock_id") == LOCK_ID
        and lock.get("status") == "AUTHORIZED_UNCONSUMED"
        and lock.get("consumed") is False,
        "R10_SELECTION_LOCK_IDENTITY",
        "R10 selection lock identity drift",
    )
    require(lock.get("user_authority") == EXPECTED_USER_AUTHORITY, "R10_SELECTION_USER_AUTHORITY", "R10 selection user authority drift")
    actual_argv = [
        Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(),
        "--execution-lock",
        lock_path.relative_to(REPO_ROOT).as_posix(),
    ]
    require(
        lock.get("argv") == actual_argv
        and lock.get("phase_a_root") == PHASE_A_ROOT
        and lock.get("inventory_path") == INVENTORY_PATH
        and lock.get("output_root") == OUTPUT_ROOT
        and lock.get("overwrite") is False
        and lock.get("rerun") is False
        and lock.get("frozen_selector") == FROZEN_LOCK_SELECTOR,
        "R10_SELECTION_LOCK_POLICY",
        "R10 selection argv/root/selector policy drift",
    )
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R10_SELECTION_BINDINGS", "R10 selection binding count drift")
    seen: set[str] = set()
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(
            set(row) == {"role", "path", "bytes", "sha256"}
            and role not in seen
            and EXPECTED_BINDINGS.get(role) == relative,
            "R10_SELECTION_BINDING_ROW",
            "R10 selection binding row drift",
        )
        seen.add(role)
        target = _repo_path(relative)
        require(
            target.is_file()
            and target.stat().st_size == row["bytes"]
            and materializer.sha256_file(target) == row["sha256"],
            "R10_SELECTION_BINDING_HASH",
            f"R10 selection binding drift: {relative}",
        )
    require(seen == set(EXPECTED_BINDINGS), "R10_SELECTION_BINDINGS", "R10 selection binding roles drift")
    validate_protocol(_read_json(_repo_path(EXPECTED_BINDINGS["R10_PROTOCOL"])))
    validate_frozen_selector(_read_json(_repo_path(EXPECTED_BINDINGS["R9_FROZEN_SELECTOR"])))
    phase_result = _read_json(_repo_path(EXPECTED_BINDINGS["R10_PHASE_A_R1_RESULT"]))
    require(
        phase_result.get("schema") == "blindassist.taro.o1r.r10_fresh_pool_phase_a_result.v1"
        and phase_result.get("terminal") == phase_a_r1.PASS_TERMINAL
        and phase_result.get("passed") is True
        and phase_result.get("execution_valid") is True
        and phase_result.get("parent_count") == PARENT_COUNT
        and phase_result.get("frame_count") == FRAME_COUNT
        and phase_result.get("query_count") == QUERY_COUNT
        and phase_result.get("candidate_inference_count") == FRAME_COUNT
        and phase_result.get("faro_reads") == 0
        and phase_result.get("truth_scoring") is False
        and phase_result.get("clear_output_allowed") is False
        and phase_result.get("training_steps") == phase_result.get("network_requests") == 0,
        "R10_SELECTION_PHASE_A_NOT_ADMITTED",
        "R10 Phase-A R1 result is not admitted",
    )
    require(
        lock.get("execution_authority") == EXPECTED_AUTHORITY
        and lock.get("resource_budget") == {"maximum_evidence_bytes": 16_777_216}
        and lock.get("one_shot_policy")
        == {
            "consumed_on_output_root_creation": True,
            "failure_does_not_restore_authority": True,
            "expected_file_count_before_manifest": SELECTION_FILE_COUNT,
        },
        "R10_SELECTION_AUTHORITY",
        "R10 selection authority, budget, or one-shot policy drift",
    )
    require(not _repo_path(OUTPUT_ROOT).exists(), "R10_SELECTION_ROOT_COLLISION", "R10 selection output root exists")
    lock["_lock_path"] = lock_path
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    writer = FactorEvidenceWriter(_repo_path(OUTPUT_ROOT), int(lock["resource_budget"]["maximum_evidence_bytes"]))
    writer.activate(
        {
            "schema": "blindassist.taro.o1r.r10_fresh_pool_top8_selection_execution_receipt.v1",
            "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]),
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "phase_a_root": PHASE_A_ROOT,
            "expected_phase_a_file_count_before_manifest": PHASE_A_FILE_COUNT,
            "frozen_selector": FROZEN_LOCK_SELECTOR,
            "faro_reads": 0,
            "truth_reads": 0,
            "label_reads": 0,
            "outcome_reads": 0,
            "training_steps": 0,
            "network_requests": 0,
            "one_shot_consumed_on_root_creation": True,
        }
    )
    try:
        frames = phase_a._load_frames(_repo_path(INVENTORY_PATH))
        phase_root = _repo_path(PHASE_A_ROOT)
        phase_manifest = verify_phase_a_manifest(phase_root, frames)
        completion, sources = load_phase_a_sources(phase_root, frames)
        parent_scores, selection = build_selection(completion, sources)
        writer.write_json("parent-scores.json", parent_scores)
        writer.write_json("selection.json", selection)
        result = {
            "schema": "blindassist.taro.o1r.r10_fresh_pool_top8_selection_result.v1",
            "terminal": PASS_TERMINAL,
            "passed": True,
            "execution_valid": True,
            "parent_count": PARENT_COUNT,
            "frame_count": FRAME_COUNT,
            "query_count": QUERY_COUNT,
            "phase_a_manifest_file_count_before_manifest": phase_manifest["file_count_before_manifest"],
            "phase_a_completion_sha256": completion["content_sha256"],
            "parent_scores_sha256": parent_scores["content_sha256"],
            "selection_sha256": selection["content_sha256"],
            "selected_parent_count": SELECTED_PARENT_COUNT,
            "selected_parent_identities": selection["selected_parent_identities"],
            "all_32_scores_sealed_before_faro": True,
            "selection_sealed_before_faro": True,
            "faro_reads": 0,
            "truth_reads": 0,
            "label_reads": 0,
            "outcome_reads": 0,
            "clear_output_emitted": False,
            "training_steps": 0,
            "network_requests": 0,
            "one_shot_consumed": True,
            "claim_ceiling": "Source-only R10 parent scores and sealed top-eight identity; no FARO label, effectiveness, deployment, product, or safety evidence.",
        }
        writer.write_json("result.json", result)
        require(
            len(writer.file_receipts) == SELECTION_FILE_COUNT,
            "R10_SELECTION_MANIFEST_COUNT_DRIFT",
            "R10 selection file count before manifest drift",
        )
        writer.write_json(
            "manifest.json",
            {
                "schema": "blindassist.taro.o1r.r10_fresh_pool_top8_selection_manifest.v1",
                "terminal": PASS_TERMINAL,
                "files": dict(sorted(writer.file_receipts.items())),
                "file_count_before_manifest": len(writer.file_receipts),
                "bytes_before_manifest": writer.bytes_written,
            },
        )
        return result
    except Exception as error:
        try:
            writer.write_json(
                "failure.json",
                {
                    "schema": "blindassist.taro.o1r.r10_fresh_pool_top8_selection_failure.v1",
                    "terminal": FAIL_TERMINAL,
                    "execution_valid": False,
                    "failure_code": str(getattr(error, "code", type(error).__name__)),
                    "message": str(error),
                    "faro_reads": 0,
                    "truth_reads": 0,
                    "label_reads": 0,
                    "outcome_reads": 0,
                    "one_shot_consumed": True,
                },
            )
            writer.write_json(
                "manifest.json",
                {
                    "schema": "blindassist.taro.o1r.r10_fresh_pool_top8_selection_manifest.v1",
                    "terminal": FAIL_TERMINAL,
                    "files": dict(sorted(writer.file_receipts.items())),
                    "file_count_before_manifest": len(writer.file_receipts),
                    "bytes_before_manifest": writer.bytes_written,
                },
            )
        except Exception:
            pass
        raise


def assert_public_api_source_only() -> None:
    for function in (score_parent, rank_parent_scores, build_selection):
        names = inspect.signature(function).parameters
        require(
            not any(token in name.lower() for name in names for token in ("faro", "truth", "label", "outcome")),
            "R10_SELECTION_RESULT_SIDE_API",
            "R10 selector public API has a result-side parameter",
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
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
