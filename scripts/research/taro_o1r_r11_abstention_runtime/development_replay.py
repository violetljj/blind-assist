#!/usr/bin/env python3
"""Read-only R10 replay for forming the R11 development-only candidate."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r7_canary_runtime import positive_occupancy_factor as r7_positive
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r10_clear_runtime import run_selected_phase_b as r10_phase_b
from scripts.research.taro_o1r_r11_abstention_runtime import abstention_candidate


SCHEMA = "blindassist.taro.o1r.r11_abstention_development_replay_result.v1"
STATUS = "TUNED_ON_CONSUMED_R10_DEVELOPMENT_ONLY_REQUIRES_FRESH_CONFIRMATION"
R10_FORMAL_RESULT = "docs/research/taro/TARO_O1R_R10_FRESH_CLEAR_ENRICHED_CONFIRMATION_RESULT_2026-08-12.json"
EXPECTED_R10_TERMINAL = "TARO_O1R_R10_FRESH_CLEAR_ENRICHED_NOT_EVALUABLE_DUAL_CLASS_COVERAGE"


class DevelopmentReplayError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise DevelopmentReplayError(code, message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R11_DEVELOPMENT_SEAL_COLLISION", "caller supplied content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _state_metrics(tp: int, fp: int, occupied_count: int, clear_count: int) -> dict[str, Any]:
    precision_denominator = tp + fp
    return {
        "occupied_true_positive": tp,
        "occupied_false_positive_against_definite_clear": fp,
        "occupied_false_negative": occupied_count - tp,
        "occupied_precision_on_definite_labels": tp / precision_denominator if precision_denominator else 0.0,
        "one_sided_95_wilson_occupied_precision_lower_bound": r7_canary._wilson_lower(tp, precision_denominator) if precision_denominator else 0.0,
        "occupied_recall": tp / occupied_count if occupied_count else 0.0,
        "clear_specific_successes": clear_count - fp,
        "clear_specificity_on_definite_clear": (clear_count - fp) / clear_count if clear_count else 0.0,
        "one_sided_95_wilson_clear_specificity_lower_bound": r7_canary._wilson_lower(clear_count - fp, clear_count) if clear_count else 0.0,
    }


def summarize(
    sources: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    selected_identities: Sequence[tuple[str, str]],
    *,
    validate_records: bool = True,
) -> dict[str, Any]:
    require(len(sources) == len(labels) and len(sources) > 0, "R11_DEVELOPMENT_RECORD_COUNT", "source/label frame count drift")
    expected_identities = set(selected_identities)
    require(bool(expected_identities), "R11_DEVELOPMENT_IDENTITIES", "selected identity set is empty")
    truth_counts: Counter[str] = Counter()
    base_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    suppressed_truth: Counter[str] = Counter()
    parent_occupied: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    clear_frame_rows: dict[tuple[str, str], list[tuple[bool, bool]]] = defaultdict(list)
    query_count = 0
    for raw_source, raw_label in zip(sources, labels, strict=True):
        source = r7_canary.validate_source_frame_record(dict(raw_source)) if validate_records else dict(raw_source)
        label_record = r7_canary.validate_label_frame_record(dict(raw_label), source) if validate_records else dict(raw_label)
        identity = (str(source["parent_id"]), str(source["video_id"]))
        require(identity in expected_identities and source["physical_frame_id"] == label_record["physical_frame_id"], "R11_DEVELOPMENT_FRAME_ALIGNMENT", "source/label frame or identity drift")
        features = source["query_features"]
        query_labels = label_record["query_labels"]
        require(len(features) == len(query_labels), "R11_DEVELOPMENT_QUERY_COUNT", "source/label query count drift")
        for feature, label in zip(features, query_labels, strict=True):
            require(feature["query_id"] == label["query_id"] and label["state"] in {"CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN"}, "R11_DEVELOPMENT_QUERY_ALIGNMENT", "query lineage or truth state drift")
            base_state, _base_reasons = r7_positive.state_from_feature(feature)
            candidate_state, _candidate_reasons = abstention_candidate.state_from_feature(feature)
            base_positive = base_state == "OCCUPIED_OBSERVED"
            candidate_positive = candidate_state == "OCCUPIED_OBSERVED"
            require(not candidate_positive or base_positive, "R11_DEVELOPMENT_SUBSET_DRIFT", "candidate positive is not a subset of R7 positive")
            truth = str(label["state"])
            truth_counts[truth] += 1
            base_counts[base_state] += 1
            candidate_counts[candidate_state] += 1
            if base_positive and not candidate_positive:
                suppressed_truth[truth] += 1
            if truth == "OCCUPIED_OBSERVED":
                parent_occupied[identity[0]].append((base_positive, candidate_positive))
            elif truth == "CLEAR_OBSERVED":
                clear_frame_rows[(identity[0], str(source["physical_frame_id"]))].append((base_positive, candidate_positive))
            query_count += 1

    occupied_count = int(truth_counts["OCCUPIED_OBSERVED"])
    clear_count = int(truth_counts["CLEAR_OBSERVED"])
    base_tp = sum(base for rows in parent_occupied.values() for base, _candidate in rows)
    candidate_tp = sum(candidate for rows in parent_occupied.values() for _base, candidate in rows)
    base_fp = sum(base for rows in clear_frame_rows.values() for base, _candidate in rows)
    candidate_fp = sum(candidate for rows in clear_frame_rows.values() for _base, candidate in rows)
    base_macro = sum(sum(base for base, _candidate in rows) / len(rows) for rows in parent_occupied.values()) / len(parent_occupied)
    candidate_macro = sum(sum(candidate for _base, candidate in rows) / len(rows) for rows in parent_occupied.values()) / len(parent_occupied)

    clear_frame_count = len(clear_frame_rows)
    base_clear_frame_successes = sum(not any(base for base, _candidate in rows) for rows in clear_frame_rows.values())
    candidate_clear_frame_successes = sum(not any(candidate for _base, candidate in rows) for rows in clear_frame_rows.values())
    clear_parent_frames: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for (parent, _frame), rows in clear_frame_rows.items():
        clear_parent_frames[parent].append((not any(base for base, _candidate in rows), not any(candidate for _base, candidate in rows)))
    base_parent_macro_clear = sum(sum(base for base, _candidate in rows) / len(rows) for rows in clear_parent_frames.values()) / len(clear_parent_frames)
    candidate_parent_macro_clear = sum(sum(candidate for _base, candidate in rows) / len(rows) for rows in clear_parent_frames.values()) / len(clear_parent_frames)

    base_metrics = _state_metrics(base_tp, base_fp, occupied_count, clear_count)
    candidate_metrics = _state_metrics(candidate_tp, candidate_fp, occupied_count, clear_count)
    base_metrics.update(
        {
            "parent_macro_definite_occupied_recall": base_macro,
            "definite_clear_physical_frame_count": clear_frame_count,
            "clear_frame_successes": base_clear_frame_successes,
            "clear_frame_specificity": base_clear_frame_successes / clear_frame_count,
            "one_sided_95_wilson_clear_frame_specificity_lower_bound": r7_canary._wilson_lower(base_clear_frame_successes, clear_frame_count),
            "parent_macro_clear_frame_specificity": base_parent_macro_clear,
        }
    )
    candidate_metrics.update(
        {
            "parent_macro_definite_occupied_recall": candidate_macro,
            "definite_clear_physical_frame_count": clear_frame_count,
            "clear_frame_successes": candidate_clear_frame_successes,
            "clear_frame_specificity": candidate_clear_frame_successes / clear_frame_count,
            "one_sided_95_wilson_clear_frame_specificity_lower_bound": r7_canary._wilson_lower(candidate_clear_frame_successes, clear_frame_count),
            "parent_macro_clear_frame_specificity": candidate_parent_macro_clear,
        }
    )
    return json.loads(
        adapter.canonical_json_bytes(
            {
                "frame_count": len(sources),
                "query_count": query_count,
                "parent_count": len(expected_identities),
                "parents_with_definite_occupied": len(parent_occupied),
                "parents_with_definite_clear": len(clear_parent_frames),
                "truth_state_counts": {state: int(truth_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
                "base_prediction_state_counts": {state: int(base_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
                "candidate_prediction_state_counts": {state: int(candidate_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
                "suppressed_base_positive_by_truth": {state: int(suppressed_truth[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
                "base_metrics": base_metrics,
                "candidate_metrics": candidate_metrics,
                "candidate_minus_base": {
                    "occupied_recall": candidate_metrics["occupied_recall"] - base_metrics["occupied_recall"],
                    "parent_macro_definite_occupied_recall": candidate_macro - base_macro,
                    "clear_specificity_on_definite_clear": candidate_metrics["clear_specificity_on_definite_clear"] - base_metrics["clear_specificity_on_definite_clear"],
                    "clear_frame_specificity": candidate_metrics["clear_frame_specificity"] - base_metrics["clear_frame_specificity"],
                },
                "candidate_positive_subset_of_base_positive": True,
                "clear_output_count": 0,
                "unknown_is_negative": False,
            }
        ).decode("utf-8")
    )


def _verify_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    require(set(actual) == set(manifest.get("files", {})), "R11_DEVELOPMENT_R10_MANIFEST_SET", "R10 manifest file set drift")
    for relative, path in actual.items():
        row = manifest["files"][relative]
        require(path.stat().st_size == row["bytes"] and sha256_file(path) == row["sha256"], "R11_DEVELOPMENT_R10_MANIFEST_HASH", f"R10 evidence hash drift: {relative}")
    require(sum(path.stat().st_size for path in actual.values()) == manifest["bytes_before_manifest"], "R11_DEVELOPMENT_R10_MANIFEST_BYTES", "R10 manifest byte total drift")
    return manifest


def build_actual_replay(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    formal_path = root / R10_FORMAL_RESULT
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    require(
        formal.get("terminal") == EXPECTED_R10_TERMINAL
        and formal.get("passed") is False
        and formal.get("scientifically_evaluable") is False
        and formal.get("interpretation", {}).get("r10_consumed_and_not_retargetable") is True,
        "R11_DEVELOPMENT_R10_TERMINAL",
        "R10 predecessor terminal or consumption drift",
    )
    evidence_root = root / r10_phase_b.OUTPUT_ROOT
    manifest = _verify_manifest(evidence_root)
    evidence = formal["evidence"]
    require(
        sha256_file(evidence_root / "result.json") == evidence["phase_b_result_sha256"]
        and sha256_file(evidence_root / "label-completion.json") == evidence["phase_b_label_completion_file_sha256"]
        and sha256_file(evidence_root / "manifest.json") == evidence["phase_b_manifest_sha256"],
        "R11_DEVELOPMENT_R10_CORE_HASH",
        "R10 core evidence hash drift",
    )
    _parent_scores, selection, frames, sources, _receipts = r10_phase_b.load_selected_rows()
    labels = []
    for frame, source in zip(frames, sources, strict=True):
        with gzip.open(evidence_root / r10_phase_b._label_relative(frame), "rt", encoding="utf-8") as stream:
            label = json.load(stream)
        labels.append(r7_canary.validate_label_frame_record(label, source))
    identities = [tuple(row) for row in selection["selected_parent_identities"]]
    summary = summarize(sources, labels, identities)
    return validate_development_result(
        _seal(
            {
                "schema": SCHEMA,
                "status": STATUS,
                "date": "2026-08-12",
                "formation_role": "TUNED_ON_CONSUMED_R10_DEVELOPMENT_ONLY",
                "candidate": abstention_candidate.FROZEN_ALGORITHM,
                "r10_lineage": {
                    "formal_result_record_path": R10_FORMAL_RESULT,
                    "formal_result_record_sha256": sha256_file(formal_path),
                    "terminal": formal["terminal"],
                    "consumed": True,
                    "terminal_immutable": True,
                    "phase_b_result_sha256": evidence["phase_b_result_sha256"],
                    "phase_b_label_completion_sha256": evidence["phase_b_label_completion_file_sha256"],
                    "phase_b_manifest_sha256": evidence["phase_b_manifest_sha256"],
                    "manifest_entries_verified": len(manifest["files"]),
                    "manifest_bytes_verified": manifest["bytes_before_manifest"],
                },
                "summary": summary,
                "identity_specific_exception_allowed": False,
                "additional_threshold_search_authorized": False,
                "fresh_confirmation_authority": False,
                "new_source_reads": 0,
                "raw_faro_reads": 0,
                "model_inference_count": 0,
                "training_steps": 0,
                "network_requests": 0,
                "route_promotion": False,
                "deployment_promotion": False,
                "product_promotion": False,
                "safety_promotion": False,
                "claim_ceiling": "Consumed-R10 development replay of one truth-blind abstention candidate only; R10 remains NOT_EVALUABLE and the candidate requires a fresh parent-disjoint confirmation.",
            }
        )
    )


def validate_development_result(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    claimed = record.pop("content_sha256", None)
    require(record.get("schema") == SCHEMA and claimed == adapter.canonical_sha256(record), "R11_DEVELOPMENT_RESULT_SEAL", "development result seal or schema drift")
    record["content_sha256"] = claimed
    require(
        record.get("status") == STATUS
        and record.get("formation_role") == "TUNED_ON_CONSUMED_R10_DEVELOPMENT_ONLY"
        and record.get("candidate") == abstention_candidate.FROZEN_ALGORITHM,
        "R11_DEVELOPMENT_RESULT_IDENTITY",
        "development role or candidate drift",
    )
    lineage = record.get("r10_lineage", {})
    require(
        lineage.get("terminal") == EXPECTED_R10_TERMINAL
        and lineage.get("consumed") is True
        and lineage.get("terminal_immutable") is True
        and lineage.get("manifest_entries_verified") == 263,
        "R11_DEVELOPMENT_RESULT_LINEAGE",
        "R10 lineage drift",
    )
    summary = record.get("summary", {})
    base = summary.get("base_metrics", {})
    candidate_metrics = summary.get("candidate_metrics", {})
    require(
        summary.get("frame_count") == 260
        and summary.get("query_count") == 2340
        and summary.get("parent_count") == 8
        and base.get("occupied_true_positive") == 1769
        and base.get("occupied_false_positive_against_definite_clear") == 1
        and candidate_metrics.get("occupied_true_positive") == 1768
        and candidate_metrics.get("occupied_false_positive_against_definite_clear") == 0
        and summary.get("suppressed_base_positive_by_truth") == {"CLEAR_OBSERVED": 1, "OCCUPIED_OBSERVED": 1, "UNKNOWN": 0}
        and summary.get("candidate_positive_subset_of_base_positive") is True
        and summary.get("clear_output_count") == 0
        and summary.get("unknown_is_negative") is False,
        "R11_DEVELOPMENT_RESULT_METRICS",
        "retrospective development metrics drift",
    )
    require(
        record.get("identity_specific_exception_allowed") is False
        and record.get("additional_threshold_search_authorized") is False
        and record.get("fresh_confirmation_authority") is False
        and record.get("new_source_reads") == record.get("raw_faro_reads") == record.get("model_inference_count") == record.get("training_steps") == record.get("network_requests") == 0
        and record.get("route_promotion") is False
        and record.get("deployment_promotion") is False
        and record.get("product_promotion") is False
        and record.get("safety_promotion") is False,
        "R11_DEVELOPMENT_RESULT_AUTHORITY",
        "development result exceeded its authority",
    )
    return record


if __name__ == "__main__":
    print(adapter.canonical_json_bytes(build_actual_replay(Path(__file__).resolve().parents[3])).decode("utf-8"))
