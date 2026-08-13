#!/usr/bin/env python3
"""Independent, no-FARO-replay validator for sealed R11 selected-top24 Phase B."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r7_canary_runtime import positive_occupancy_factor as r7_positive
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r11_abstention_runtime import abstention_candidate
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_inventory
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_phase_a as phase_a


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = "artifacts.local/evidence/taro/o1r-r11-selected-top24-faro-phase-b-r0"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_ONE_SHOT_EXECUTION_LOCK_2026-08-13.json"
INVENTORY_RELATIVE = phase_a.INVENTORY_PATH
PHASE_A_ROOT = phase_a.OUTPUT_ROOT
TOP24_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-top24-selection-r0"
TOP24_SELECTION_RELATIVE = f"{TOP24_ROOT}/selection.json"
LOCK_SCHEMA = "blindassist.taro.o1r.r11_selected_top24_faro_phase_b_execution_lock.v1"
LOCK_ID = "TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_ONE_SHOT_EXECUTION_LOCK"
TERMINAL_SCHEMA = "blindassist.taro.o1r.r11_selected_top24_faro_phase_b_terminal.v1"
RESULT_SCHEMA = "blindassist.taro.o1r.r11_selected_top24_faro_phase_b_result.v1"
COMPLETION_SCHEMA = "blindassist.taro.o1r.r11_selected_top24_faro_label_completion.v1"
PASS_TERMINAL = "WILD_LAB_RESEARCH_FACTOR_CONFIRMATION_PASS"
FAIL_TERMINAL = "FAIL_FIXED_CONFIRMATION_GATE"
NOT_EVALUABLE_TERMINAL = "NOT_EVALUABLE_DUAL_CLASS_COVERAGE"
SELECTED_PARENT_COUNT = 24
SELECTED_FRAME_COUNT = 674
SELECTED_QUERY_COUNT = 6066
FINAL_FILE_COUNT = 678
TOP24_SELECTION_CONTENT_SHA256 = "629ECF7069EE5942EAEF7946059CAD03D20D0F66CBD4DAF95E06A5315211A7B7"

EVALUABILITY = {
    "selected_parent_count": 24,
    "minimum_evaluable_parents": 16,
    "minimum_parents_with_definite_occupied": 12,
    "minimum_definite_occupied_queries": 200,
    "minimum_parents_with_definite_clear": 4,
    "minimum_physical_frames_with_definite_clear": 12,
    "minimum_definite_clear_queries": 20,
}
GATES = {
    "minimum_candidate_occupied_precision": 0.9,
    "minimum_one_sided_95_wilson_candidate_occupied_precision_lower_bound": 0.8,
    "minimum_candidate_occupied_recall": 0.9,
    "minimum_parent_macro_definite_occupied_recall": 0.9,
    "maximum_micro_occupied_recall_loss_vs_r7": 0.01,
    "maximum_parent_macro_occupied_recall_loss_vs_r7": 0.01,
    "candidate_false_positives_must_not_exceed_r7": True,
    "minimum_query_clear_specificity": 0.9,
    "minimum_clear_frame_specificity": 0.9,
    "minimum_one_sided_95_wilson_clear_frame_specificity_lower_bound": 0.8,
    "minimum_parent_macro_clear_frame_specificity": 0.9,
    "maximum_clear_outputs": 0,
    "unknown_is_negative": False,
}
EXPECTED_BINDINGS = {
    "R11_PROTOCOL": "docs/research/taro/TARO_O1R_R11_POSITIVE_OCCUPANCY_ABSTENTION_AND_FRESH_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json",
    "R11_DATA_USE_AUTHORIZATION": "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-12.json",
    "R11_INVENTORY": INVENTORY_RELATIVE,
    "R11_PHASE_A_COMPLETION": f"{PHASE_A_ROOT}/phase-a-completion.json",
    "R11_PHASE_A_TERMINAL": f"{PHASE_A_ROOT}/terminal.json",
    "R11_PHASE_A_REPAIRED_AUDIT": "artifacts.local/evidence/taro/o1r-r11-fresh-pool-phase-a-validator-round12-repair-r0/post-result-audit.json",
    "R11_TOP24_TERMINAL": f"{TOP24_ROOT}/terminal.json",
    "R11_TOP24_PARENT_SCORES": f"{TOP24_ROOT}/parent-scores.json",
    "R11_TOP24_SELECTION": TOP24_SELECTION_RELATIVE,
    "R11_TOP24_FORMAL_RESULT": "docs/research/taro/TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_RESULT_2026-08-13.json",
    "SOURCE_ADAPTER_RUNTIME": "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py",
    "TRUTH_MATERIALIZER_RUNTIME": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "CANDIDATE_SCALE_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py",
    "REDUCER_INTEGRATION_RUNTIME": "scripts/research/taro_o1r_reducer_integration_runtime/reducer_integration.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "R7_CANARY_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "R7_POSITIVE_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/positive_occupancy_factor.py",
    "R11_ABSTENTION_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/abstention_candidate.py",
    "R11_INVENTORY_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_inventory.py",
    "R11_PHASE_A_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_phase_a.py",
    "R11_PROTOCOL_VALIDATOR": "scripts/research/taro_o1r_r11_abstention_runtime/validate_protocol_lock.py",
    "R11_PHASE_B_METRICS": "scripts/research/taro_o1r_r11_abstention_runtime/phase_b_metrics.py",
    "R11_PHASE_B_METRICS_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_phase_b_metrics.py",
    "R11_PHASE_B_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_selected_phase_b.py",
    "R11_PHASE_B_RUNNER_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_run_selected_phase_b.py",
    "R11_PHASE_B_INDEPENDENT_VALIDATOR": "scripts/research/taro_o1r_r11_abstention_runtime/validate_selected_phase_b.py",
    "R11_PHASE_B_VALIDATOR_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_validate_selected_phase_b.py",
    "R11_PHASE_B_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK_2026-08-13.md",
}
ARTIFACT_BINDING_ROLES = {
    "R11_INVENTORY", "R11_PHASE_A_COMPLETION", "R11_PHASE_A_TERMINAL",
    "R11_PHASE_A_REPAIRED_AUDIT", "R11_TOP24_TERMINAL", "R11_TOP24_PARENT_SCORES",
    "R11_TOP24_SELECTION",
}
STATES = {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}


class R11PhaseBValidationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise R11PhaseBValidationError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "R11_PHASE_B_VALIDATION_JSON", f"JSON object required: {path}")
    return value


def _load_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), "R11_PHASE_B_VALIDATION_JSON", f"gzip JSON object required: {path}")
    return value


def _seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), "R11_PHASE_B_VALIDATION_SEAL", "sealed object required")
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema and isinstance(observed, str) and adapter.canonical_sha256(record) == observed,
        "R11_PHASE_B_VALIDATION_SEAL",
        f"record seal/schema drift: {schema}",
    )
    record["content_sha256"] = observed
    return record


def _git_bytes(commit: str, relative: str) -> bytes:
    completed = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=REPO_ROOT, capture_output=True, check=False)
    require(completed.returncode == 0, "R11_PHASE_B_VALIDATION_IMPLEMENTATION", f"implementation lacks {relative}")
    return completed.stdout


def _commit_on_master(commit: Any) -> bool:
    return isinstance(commit, str) and len(commit) == 40 and all(
        subprocess.run(["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, ref], capture_output=True).returncode == 0
        for ref in ("HEAD", "refs/remotes/origin/master")
    )


def validate_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R11_PHASE_B_VALIDATION_LOCK_PATH", "lock path drift")
    lock = _seal(_load_json(lock_path), LOCK_SCHEMA)
    require(
        lock.get("lock_id") == LOCK_ID
        and lock.get("status") == "AUTHORIZED_UNCONSUMED"
        and lock.get("consumed") is False
        and lock.get("selected_frame_count") == SELECTED_FRAME_COUNT
        and lock.get("selected_query_count") == SELECTED_QUERY_COUNT
        and lock.get("evaluability_gates") == EVALUABILITY
        and lock.get("confirmation_gates") == GATES
        and lock.get("top24_selection_content_sha256") == TOP24_SELECTION_CONTENT_SHA256
        and lock.get("implementation_on_origin_master") is True
        and _commit_on_master(lock.get("implementation_commit")),
        "R11_PHASE_B_VALIDATION_LOCK",
        "lock identity/cohort/gates drift",
    )
    rows = lock.get("bindings")
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_BINDINGS), "R11_PHASE_B_VALIDATION_BINDINGS", "binding count drift")
    seen = set()
    for row in rows:
        role, relative = row.get("role"), row.get("path")
        require(
            set(row) == {"role", "path", "bytes", "sha256"}
            and isinstance(role, str) and role not in seen and EXPECTED_BINDINGS.get(role) == relative,
            "R11_PHASE_B_VALIDATION_BINDING",
            "binding role/path drift",
        )
        payload = _repo_path(str(relative)).read_bytes()
        require(
            len(payload) == row.get("bytes") and materializer.sha256_bytes(payload) == row.get("sha256"),
            "R11_PHASE_B_VALIDATION_BINDING",
            f"binding bytes drift: {relative}",
        )
        if role not in ARTIFACT_BINDING_ROLES:
            require(payload == _git_bytes(str(lock["implementation_commit"]), str(relative)), "R11_PHASE_B_VALIDATION_IMPLEMENTATION", f"implementation drift: {relative}")
        seen.add(role)
    require(seen == set(EXPECTED_BINDINGS), "R11_PHASE_B_VALIDATION_BINDINGS", "binding role set drift")
    lock["_path"] = lock_path
    return lock


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(float(numerator / denominator), 12)


def _wilson(successes: int, total: int) -> float:
    return round(float(r7_canary._wilson_lower(successes, total)), 12)


def _state(row: Mapping[str, Any]) -> str:
    value = str(row.get("state"))
    require(value in STATES, "R11_PHASE_B_VALIDATION_STATE", "query state invalid")
    return value


def _reduce(
    identities: Sequence[tuple[str, str]],
    baselines: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require(len(identities) == 24 and len(set(identities)) == 24, "R11_PHASE_B_VALIDATION_IDENTITIES", "24 identities required")
    require(len(baselines) == len(candidates) == len(labels) > 0, "R11_PHASE_B_VALIDATION_COUNT", "frame count drift")
    truth_counts, base_counts, cand_counts = Counter(), Counter(), Counter()
    parent_truth: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    parent_occ: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    parent_clear: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    base_tp = base_fp = cand_tp = cand_fp = cand_fn = base_unknown = cand_unknown = clear_outputs = 0
    clear_frames = base_clear_success = cand_clear_success = abstained_frames = 0
    abstained_parents = set()
    for base, cand, label in zip(baselines, candidates, labels, strict=True):
        identity = (str(label.get("parent_id")), str(label.get("video_id")))
        require(identity in set(identities) and base.get("physical_frame_id") == cand.get("physical_frame_id") == label.get("physical_frame_id"), "R11_PHASE_B_VALIDATION_ALIGNMENT", "frame alignment drift")
        base_rows, cand_rows, truth_rows = base["query_results"], cand["query_results"], label["query_labels"]
        require(len(base_rows) == len(cand_rows) == len(truth_rows) == 9, "R11_PHASE_B_VALIDATION_QUERY_COUNT", "query count drift")
        frame_clear = frame_base_fp = frame_cand_fp = frame_abstained_clear = False
        for index, (base_row, cand_row, truth_row) in enumerate(zip(base_rows, cand_rows, truth_rows, strict=True)):
            require(base_row["grid_index"] == cand_row["grid_index"] == truth_row["grid_index"] == index and base_row["query_id"] == cand_row["query_id"] == truth_row["query_id"], "R11_PHASE_B_VALIDATION_QUERY_ALIGNMENT", "query alignment drift")
            bs, cs, ts = _state(base_row), _state(cand_row), _state(truth_row)
            require(cs != "OCCUPIED_OBSERVED" or bs == "OCCUPIED_OBSERVED", "R11_PHASE_B_VALIDATION_SUBSET", "R11 occupied not R7 subset")
            truth_counts[ts] += 1; base_counts[bs] += 1; cand_counts[cs] += 1; parent_truth[identity][ts] += 1
            clear_outputs += cs == "CLEAR_OBSERVED"
            if ts == "OCCUPIED_OBSERVED":
                base_tp += bs == "OCCUPIED_OBSERVED"; cand_tp += cs == "OCCUPIED_OBSERVED"; cand_fn += cs != "OCCUPIED_OBSERVED"
                parent_occ[identity]["truth"] += 1; parent_occ[identity]["base"] += bs == "OCCUPIED_OBSERVED"; parent_occ[identity]["cand"] += cs == "OCCUPIED_OBSERVED"
            elif ts == "CLEAR_OBSERVED":
                frame_clear = True; frame_base_fp |= bs == "OCCUPIED_OBSERVED"; frame_cand_fp |= cs == "OCCUPIED_OBSERVED"
                frame_abstained_clear |= bs == "OCCUPIED_OBSERVED" and cs == "UNKNOWN"
                base_fp += bs == "OCCUPIED_OBSERVED"; cand_fp += cs == "OCCUPIED_OBSERVED"
            else:
                base_unknown += bs == "OCCUPIED_OBSERVED"; cand_unknown += cs == "OCCUPIED_OBSERVED"
        if frame_clear:
            clear_frames += 1
            base_success, cand_success = not frame_base_fp, not frame_cand_fp
            base_clear_success += base_success; cand_clear_success += cand_success
            parent_clear[identity]["total"] += 1; parent_clear[identity]["base"] += base_success; parent_clear[identity]["cand"] += cand_success
            if frame_abstained_clear:
                abstained_frames += 1; abstained_parents.add(identity)
    definite_occ, definite_clear = truth_counts["OCCUPIED_OBSERVED"], truth_counts["CLEAR_OBSERVED"]
    base_recall, cand_recall = _ratio(base_tp, definite_occ), _ratio(cand_tp, definite_occ)
    occ_identities = [identity for identity in identities if parent_occ[identity]["truth"]]
    clear_identities = [identity for identity in identities if parent_clear[identity]["total"]]
    per_parent = []
    base_occ_values, cand_occ_values, cand_clear_values = [], [], []
    for identity in identities:
        occ, clear = parent_occ[identity], parent_clear[identity]
        base_occ = None if not occ["truth"] else _ratio(occ["base"], occ["truth"])
        cand_occ = None if not occ["truth"] else _ratio(occ["cand"], occ["truth"])
        base_clear = None if not clear["total"] else _ratio(clear["base"], clear["total"])
        cand_clear = None if not clear["total"] else _ratio(clear["cand"], clear["total"])
        if base_occ is not None: base_occ_values.append(base_occ); cand_occ_values.append(cand_occ)
        if cand_clear is not None: cand_clear_values.append(cand_clear)
        per_parent.append({
            "parent_id": identity[0], "video_id": identity[1],
            "label_state_counts": {state: int(parent_truth[identity][state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
            "definite_occupied_query_count": int(occ["truth"]), "baseline_definite_occupied_recall": base_occ,
            "candidate_definite_occupied_recall": cand_occ, "definite_clear_frame_count": int(clear["total"]),
            "baseline_clear_frame_specificity": base_clear, "candidate_clear_frame_specificity": cand_clear,
        })
    base_macro = round(sum(base_occ_values) / len(base_occ_values), 12) if base_occ_values else 0.0
    cand_macro = round(sum(cand_occ_values) / len(cand_occ_values), 12) if cand_occ_values else 0.0
    clear_macro = round(sum(cand_clear_values) / len(cand_clear_values), 12) if cand_clear_values else 0.0
    evaluability = {
        "selected_parent_count": 24,
        "evaluable_parent_count": sum(parent_truth[i]["CLEAR_OBSERVED"] + parent_truth[i]["OCCUPIED_OBSERVED"] > 0 for i in identities),
        "parents_with_definite_occupied": len(occ_identities), "definite_occupied_query_count": int(definite_occ),
        "parents_with_definite_clear": len(clear_identities), "physical_frames_with_definite_clear": int(clear_frames),
        "definite_clear_query_count": int(definite_clear),
    }
    evaluable = (
        evaluability["evaluable_parent_count"] >= 16 and len(occ_identities) >= 12 and definite_occ >= 200
        and len(clear_identities) >= 4 and clear_frames >= 12 and definite_clear >= 20
    )
    precision_den = cand_tp + cand_fp
    gates = {
        "candidate_occupied_precision": {"value": _ratio(cand_tp, precision_den), "minimum": 0.9, "denominator": int(precision_den)},
        "one_sided_95_wilson_candidate_occupied_precision_lower_bound": {"value": _wilson(cand_tp, precision_den), "minimum": 0.8},
        "candidate_occupied_recall": {"value": cand_recall, "minimum": 0.9},
        "parent_macro_definite_occupied_recall": {"value": cand_macro, "minimum": 0.9, "parent_denominator": len(occ_identities)},
        "micro_occupied_recall_loss_vs_r7": {"value": round(base_recall - cand_recall, 12), "maximum": 0.01},
        "parent_macro_occupied_recall_loss_vs_r7": {"value": round(base_macro - cand_macro, 12), "maximum": 0.01, "parent_denominator": len(occ_identities)},
        "candidate_false_positives_not_exceed_r7": {"candidate_value": int(cand_fp), "r7_value": int(base_fp)},
        "query_clear_specificity": {"value": _ratio(definite_clear - cand_fp, definite_clear), "minimum": 0.9, "denominator": int(definite_clear)},
        "clear_frame_specificity": {"value": _ratio(cand_clear_success, clear_frames), "minimum": 0.9, "denominator": int(clear_frames)},
        "one_sided_95_wilson_clear_frame_specificity_lower_bound": {"value": _wilson(cand_clear_success, clear_frames), "minimum": 0.8},
        "parent_macro_clear_frame_specificity": {"value": clear_macro, "minimum": 0.9, "parent_denominator": len(clear_identities)},
        "maximum_clear_outputs": {"value": int(clear_outputs), "maximum": 0},
    }
    for row in gates.values():
        row["passed"] = row["value"] >= row["minimum"] if "minimum" in row else row["value"] <= row["maximum"] if "maximum" in row else row["candidate_value"] <= row["r7_value"]
    all_pass = all(row["passed"] for row in gates.values())
    terminal = NOT_EVALUABLE_TERMINAL if not evaluable else PASS_TERMINAL if all_pass else FAIL_TERMINAL
    return {
        "terminal": terminal, "passed": bool(evaluable and all_pass), "scientifically_evaluable": bool(evaluable),
        "evaluability": evaluability,
        "label_state_counts": {state: int(truth_counts[state]) for state in sorted(STATES)},
        "r7_prediction_state_counts": {state: int(base_counts[state]) for state in sorted(STATES)},
        "r11_prediction_state_counts": {state: int(cand_counts[state]) for state in sorted(STATES)},
        "r7_occupied_true_positive": int(base_tp), "r7_occupied_false_positive_against_definite_clear": int(base_fp),
        "r11_occupied_true_positive": int(cand_tp), "r11_occupied_false_positive_against_definite_clear": int(cand_fp),
        "r11_occupied_false_negative": int(cand_fn), "r7_occupied_predictions_on_truth_unknown": int(base_unknown),
        "r11_occupied_predictions_on_truth_unknown": int(cand_unknown), "baseline_occupied_recall": base_recall,
        "candidate_occupied_recall": cand_recall, "baseline_parent_macro_definite_occupied_recall": base_macro,
        "candidate_parent_macro_definite_occupied_recall": cand_macro,
        "baseline_clear_frame_specificity": _ratio(base_clear_success, clear_frames),
        "candidate_clear_frame_specificity": _ratio(cand_clear_success, clear_frames),
        "abstention_effect": {
            "abstained_definite_clear_frames": int(abstained_frames), "parents_with_abstained_definite_clear_frame": len(abstained_parents),
            "effect_evaluable": abstained_frames >= 2 and len(abstained_parents) >= 2,
            "status": "ABSTENTION_EFFECT_EVALUABLE" if abstained_frames >= 2 and len(abstained_parents) >= 2 else "ABSTENTION_EFFECT_NOT_EVALUABLE",
            "effect_claim_required_for_absolute_confirmation": False,
        },
        "unknown_is_negative": False, "gates": gates, "all_confirmation_gates_passed": bool(all_pass), "per_parent": per_parent,
    }


def validate_evidence(root: Path | None = None, lock_path: Path | None = None) -> dict[str, Any]:
    evidence = (root or _repo_path(ROOT)).resolve()
    require(evidence.is_dir(), "R11_PHASE_B_VALIDATION_ROOT", "Phase B root missing")
    lock = validate_lock(lock_path or _repo_path(LOCK_RELATIVE))
    terminal = _seal(_load_json(evidence / "terminal.json"), TERMINAL_SCHEMA)
    require(terminal.get("execution_valid") is True and terminal.get("terminal") in {PASS_TERMINAL, FAIL_TERMINAL, NOT_EVALUABLE_TERMINAL}, "R11_PHASE_B_VALIDATION_TERMINAL", "terminal identity drift")
    files = terminal.get("files")
    actual = {path.relative_to(evidence).as_posix() for path in evidence.rglob("*") if path.is_file()}
    require(isinstance(files, Mapping) and len(files) == 677 and actual == set(files) | {"terminal.json"} and len(actual) == FINAL_FILE_COUNT, "R11_PHASE_B_VALIDATION_ROOT_SET", "exact 678-file root drift")
    total = 0
    for relative, receipt in files.items():
        path = materializer.safe_join(evidence, relative)
        require(receipt == {"path": relative, "bytes": path.stat().st_size, "sha256": materializer.sha256_file(path)}, "R11_PHASE_B_VALIDATION_RECEIPT", f"receipt drift: {relative}")
        total += path.stat().st_size
    require(total == terminal.get("bytes_before_terminal"), "R11_PHASE_B_VALIDATION_BYTES", "byte ledger drift")
    result = _seal(terminal.get("result"), RESULT_SCHEMA)
    completion = _seal(_load_json(evidence / "label-completion.json"), COMPLETION_SCHEMA)
    selection = _seal(_load_json(_repo_path(TOP24_SELECTION_RELATIVE)), "blindassist.taro.o1r.r11_fresh_pool_top24_source_only_selection.v1")
    identities = [tuple(row) for row in selection["selected_parent_identities"]]
    inventory = run_pool_inventory.validate_inventory(_load_json(_repo_path(INVENTORY_RELATIVE)))
    parent_map = {(str(row["visit_id"]), str(row["video_id"])): row for row in inventory["parents"]}
    frames = [(identity, str(token)) for identity in identities for token in parent_map[identity]["frame_plan"]["exact_timestamp_tokens"]]
    require(len(frames) == SELECTED_FRAME_COUNT, "R11_PHASE_B_VALIDATION_COHORT", "selected frame count drift")
    baselines, candidates, labels, label_hashes = [], [], [], []
    phase_root = _repo_path(PHASE_A_ROOT)
    for identity, token in frames:
        lineage = phase_a._validate_seal(
            _load_gzip(phase_root / f"phase-a-lineage/{identity[0]}/{identity[1]}/{token}.json.gz"),
            "blindassist.taro.o1r.r11_fresh_pool_phase_a_lineage.v1",
        )
        source = r7_canary.validate_source_frame_record(lineage["r7_source_frame_record"])
        baseline = r7_positive.validate_positive_occupancy_factor(lineage["r7_positive_factor_bundle"])
        candidate = abstention_candidate.validate_abstention_bundle(lineage["r11_abstention_bundle"])
        label = r7_canary.validate_label_frame_record(
            _seal(_load_gzip(evidence / f"labels/{identity[0]}/{identity[1]}/{token}.json.gz"), r7_canary.LABEL_FRAME_SCHEMA),
            source,
        )
        baselines.append(baseline); candidates.append(candidate); labels.append(label); label_hashes.append(label["content_sha256"])
    expected = _reduce(identities, baselines, candidates, labels)
    for key, value in expected.items():
        require(result.get(key) == value, "R11_PHASE_B_VALIDATION_METRIC", f"independent metric drift: {key}")
    expected_reads = [{"parent_id": row["parent_id"], "video_id": row["video_id"], "reads": row["frame_count"]} for row in selection["selected_parent_scores"]]
    require(
        completion.get("selected_parent_count") == 24 and completion.get("frame_count") == 674 and completion.get("query_count") == 6066
        and completion.get("label_hash_sequence_sha256") == adapter.canonical_sha256(label_hashes)
        and completion.get("faro_read_attempts") == completion.get("faro_read_completed") == 674
        and completion.get("per_parent_faro_reads") == expected_reads and completion.get("unselected_faro_reads") == 0
        and completion.get("only_payload_role_read") == "highres_depth" and completion.get("unknown_is_negative") is False
        and result.get("label_completion_sha256") == completion["content_sha256"]
        and result.get("faro_frame_count") == 674 and result.get("unselected_faro_frame_count") == 0
        and result.get("model_executions") == result.get("training_steps") == result.get("network_requests") == 0,
        "R11_PHASE_B_VALIDATION_LEDGER",
        "label/read/firewall ledger drift",
    )
    require(terminal.get("terminal") == result["terminal"] and terminal.get("passed") == result["passed"], "R11_PHASE_B_VALIDATION_TERMINAL", "terminal/result drift")
    return {
        "schema": "blindassist.taro.o1r.r11_selected_top24_faro_phase_b_independent_validation.v1",
        "passed": True, "scientific_terminal": result["terminal"], "scientific_passed": result["passed"],
        "scientifically_evaluable": result["scientifically_evaluable"], "parent_count": 24,
        "frame_count": 674, "query_count": 6066, "root_file_count": 678,
        "independently_recomputed_metrics": True, "faro_payload_replay_reads": 0,
        "unselected_faro_reads": 0, "unknown_is_negative": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_repo_path(ROOT))
    parser.add_argument("--execution-lock", type=Path, default=_repo_path(LOCK_RELATIVE))
    args = parser.parse_args(argv)
    try:
        result = validate_evidence(args.root, args.execution_lock)
    except Exception as error:
        print(json.dumps({"passed": False, "failure_code": getattr(error, "code", type(error).__name__), "message": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
