#!/usr/bin/env python3
"""Compare guarded, frame-paired REveL full-frame and fixed-tiling runs.

This is a failure-enriched public-RGB screening gate.  It does not estimate the
dataset-wide effect and never grants device, assistive-event, or production
authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BASELINE_MODE = "full_frame"
CANDIDATE_MODE = "full_plus_4_corner_crops"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_valid(guard: dict[str, Any], benchmark: dict[str, Any], expected_mode: str) -> bool:
    limits = guard.get("limits", {})
    observed = guard.get("observed", {})
    frames = benchmark.get("dataset", {}).get("evaluated_frames")
    expected_views = frames * (5 if expected_mode == CANDIDATE_MODE else 1) if isinstance(frames, int) else None
    return (
        guard.get("format") == "blindassist_guarded_gpu_run_v1"
        and guard.get("exit_code") == 0
        and guard.get("stop_reason") is None
        and guard.get("monitor_samples", 0) > 0
        and limits.get("max_frames") == frames
        and isinstance(limits.get("batch"), int)
        and limits.get("batch") > 0
        and limits.get("batch") == benchmark.get("model", {}).get("batch")
        and limits.get("inference_mode") == benchmark.get("model", {}).get("inference_mode") == expected_mode
        and limits.get("max_inference_views") == expected_views
        and benchmark.get("compute_backend", {}).get("inference_views") == expected_views
        and observed.get("relevant_system_events") == 0
        and observed.get("max_temperature_c") is not None
        and observed.get("max_temperature_c") < limits.get("max_temperature_c", 0)
    )


def _paired_rows(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(baseline) != len(candidate):
        raise ValueError("details row count mismatch")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for left, right in zip(baseline, candidate):
        identity_fields = ("selected_index", "image_name", "source_timestamp_ns")
        if any(left.get(field) != right.get(field) for field in identity_fields):
            raise ValueError("details frame identity mismatch")
        left_truth = [{key: item.get(key) for key in ("xyxy_normalized", "normalized_area", "stratum")} for item in left.get("ground_truth", [])]
        right_truth = [{key: item.get(key) for key in ("xyxy_normalized", "normalized_area", "stratum")} for item in right.get("ground_truth", [])]
        if left_truth != right_truth:
            raise ValueError("paired ground truth mismatch")
        pairs.append((left, right))
    return pairs


def _matched_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary = {name: {"ground_truth": 0, "matched": 0} for name in ("small", "medium", "large")}
    for row in rows:
        for item in row.get("ground_truth", []):
            bucket = item["stratum"]
            summary[bucket]["ground_truth"] += 1
            summary[bucket]["matched"] += int(item["matched_at_fixed_score"])
    return summary


def _one_sided_mcnemar(recovered: int, regressed: int) -> float:
    discordant = recovered + regressed
    if discordant == 0 or recovered <= regressed:
        return 1.0
    return sum(math.comb(discordant, successes) for successes in range(recovered, discordant + 1)) / (2 ** discordant)


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_guard: dict[str, Any],
    candidate_guard: dict[str, Any],
    baseline_details: list[dict[str, Any]],
    candidate_details: list[dict[str, Any]],
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    if contract.get("format") != "blindassist_revel_crop_tiling_selection_v1":
        raise ValueError("unsupported selection contract")
    stage = contract.get("stage_frames")
    if stage not in (8, 32):
        raise ValueError("selection contract stage must be 8 or 32")
    baseline_dataset = baseline.get("dataset", {})
    candidate_dataset = candidate.get("dataset", {})
    baseline_model = baseline.get("model", {})
    candidate_model = candidate.get("model", {})
    receipt_sha = baseline_dataset.get("selection_receipt", {}).get("sha256")
    if (
        baseline.get("format") != candidate.get("format") or baseline.get("format") != "blindassist_revel_yolo11n_person_benchmark_v2"
        or baseline_dataset != candidate_dataset
        or baseline_dataset.get("selected_indices") != contract.get("selected_indices")
        or receipt_sha != contract_sha256
        or candidate_dataset.get("selection_receipt", {}).get("sha256") != contract_sha256
        or baseline_model.get("weights_sha256") != candidate_model.get("weights_sha256")
        or baseline_model.get("imgsz") != candidate_model.get("imgsz") or baseline_model.get("imgsz") != 256
        or baseline_model.get("batch") != candidate_model.get("batch")
        or not isinstance(baseline_model.get("batch"), int) or baseline_model.get("batch") < 1
        or baseline_model.get("half") is not False or candidate_model.get("half") is not False
        or baseline_model.get("score_floor") != candidate_model.get("score_floor")
        or baseline_model.get("iou_threshold") != candidate_model.get("iou_threshold")
        or baseline_model.get("inference_mode") != BASELINE_MODE
        or candidate_model.get("inference_mode") != CANDIDATE_MODE
    ):
        raise ValueError("runs are not a valid paired crop/tiling comparison")
    if not _guard_valid(baseline_guard, baseline, BASELINE_MODE) or not _guard_valid(candidate_guard, candidate, CANDIDATE_MODE):
        raise ValueError("one or both GPU guard receipts are invalid")

    pairs = _paired_rows(baseline_details, candidate_details)
    if len(pairs) != stage:
        raise ValueError("paired details count does not match the frozen stage")
    baseline_expected = contract.get("expected_baseline", {})
    baseline_fixed = baseline.get("fixed_score_metrics", {})
    baseline_strata = _matched_summary(baseline_details)
    for name in ("tp", "fp", "fn"):
        if baseline_fixed.get(name) != baseline_expected.get(name):
            raise ValueError(f"baseline did not reproduce frozen {name}")
    for name in ("small", "medium", "large"):
        expected = baseline_expected.get("strata", {}).get(name)
        if expected != baseline_strata[name]:
            raise ValueError(f"baseline did not reproduce frozen {name} stratum")

    gt_pairs: list[dict[str, Any]] = []
    recovered_small_indices: set[int] = set()
    empty_control_candidate_fp = 0
    for baseline_row, candidate_row in pairs:
        if not baseline_row["ground_truth"]:
            empty_control_candidate_fp += int(candidate_row["fixed_score_counts"]["fp"])
        for ordinal, (left_gt, right_gt) in enumerate(zip(baseline_row["ground_truth"], candidate_row["ground_truth"])):
            baseline_matched = bool(left_gt["matched_at_fixed_score"])
            candidate_matched = bool(right_gt["matched_at_fixed_score"])
            recovered = not baseline_matched and candidate_matched
            regressed = baseline_matched and not candidate_matched
            if left_gt["stratum"] == "small" and recovered:
                recovered_small_indices.add(int(baseline_row["selected_index"]))
            gt_pairs.append({
                "selected_index": baseline_row["selected_index"],
                "source_timestamp_ns": baseline_row["source_timestamp_ns"],
                "gt_ordinal": ordinal,
                "stratum": left_gt["stratum"],
                "baseline_matched": baseline_matched,
                "candidate_matched": candidate_matched,
                "recovered": recovered,
                "regressed": regressed,
            })

    candidate_strata = _matched_summary(candidate_details)
    recovered_small = sum(item["stratum"] == "small" and item["recovered"] for item in gt_pairs)
    regressed_small = sum(item["stratum"] == "small" and item["regressed"] for item in gt_pairs)
    regressed_all = sum(item["regressed"] for item in gt_pairs)
    failure_segments = contract.get("failure_segment_by_selected_index", {})
    recovered_segment_count = len({failure_segments[str(index)] for index in recovered_small_indices if str(index) in failure_segments})
    mcnemar_p = _one_sided_mcnemar(recovered_small, regressed_small)
    candidate_fixed = candidate["fixed_score_metrics"]

    common_gate = (
        regressed_all == 0
        and candidate_fixed["f1"] >= baseline_fixed["f1"]
        and candidate_fixed["fp"] <= (6 if stage == 8 else 23)
    )
    if stage == 8:
        passed = common_gate and recovered_small >= 2
        decision = "advance_to_32_frame_gate" if passed else "stop_after_8_frame_canary"
    else:
        passed = (
            common_gate
            and recovered_small >= 5
            and regressed_small == 0
            and recovered_segment_count >= 5
            and candidate_strata["small"]["matched"] >= 13
            and candidate_strata["medium"]["matched"] >= 12
            and candidate_strata["large"]["matched"] >= 7
            and empty_control_candidate_fp <= 1
            and mcnemar_p <= 0.05
        )
        decision = "candidate_for_pre_registered_128_source_evaluation" if passed else "do_not_expand_tiling_candidate"

    return {
        "format": "blindassist_revel_detector_crop_tiling_pair_v1",
        "stage_frames": stage,
        "paired_receipt_valid": True,
        "sample_role": contract["sample_role"],
        "contract_sha256": contract_sha256,
        "baseline": {"fixed_score_metrics": baseline_fixed, "strata": baseline_strata, "guard": baseline_guard["observed"]},
        "candidate": {"fixed_score_metrics": candidate_fixed, "strata": candidate_strata, "guard": candidate_guard["observed"], "empty_control_fixed_score_fp": empty_control_candidate_fp},
        "paired_gt": {"records": gt_pairs, "small_recovered": recovered_small, "small_regressed": regressed_small, "all_regressed": regressed_all, "recovered_failure_segments": recovered_segment_count, "one_sided_exact_mcnemar_p": mcnemar_p},
        "gate_passed": passed,
        "decision": decision,
        "authority": "bounded-public-rgb-tiling-screening-only",
        "not_authorized_for": ["dataset-wide effect", "distance", "physical TTC", "body-local risk", "assistive event truth", "device safety", "production"],
        "production_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-guard", type=Path, required=True)
    parser.add_argument("--candidate-guard", type=Path, required=True)
    parser.add_argument("--baseline-details", type=Path, required=True)
    parser.add_argument("--candidate-details", type=Path, required=True)
    parser.add_argument("--selection-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        _read_json(args.baseline),
        _read_json(args.candidate),
        _read_json(args.baseline_guard),
        _read_json(args.candidate_guard),
        _read_jsonl(args.baseline_details),
        _read_jsonl(args.candidate_details),
        _read_json(args.selection_contract),
        _sha256(args.selection_contract),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage_frames": report["stage_frames"], "gate_passed": report["gate_passed"], "decision": report["decision"], "small_recovered": report["paired_gt"]["small_recovered"], "small_regressed": report["paired_gt"]["small_regressed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
