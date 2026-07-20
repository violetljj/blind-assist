#!/usr/bin/env python3
"""Evaluate frozen semantic exit rules on an isolated public-video challenge.

Candidate generation consumes only the timestamped prompt-free semantic scan.
The model/VLM review is opened afterwards solely to score candidate timing. The
result is discovery evidence, never human event truth or production evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_external_exit_router_challenge_v1"
VARIANTS: dict[str, tuple[str, ...]] = {
    "surface_only": ("surface_material",),
    "barrier_only": ("barrier_structure",),
    "risk_profile_union": ("surface_material", "barrier_structure"),
}


def verify_json_sidecar(path: Path) -> dict[str, Any]:
    path = path.resolve()
    mil.reject_independent_direction(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"JSON sidecar is missing: {path}")
    expected = sidecar.read_text(encoding="ascii").strip().lower()
    actual = common.sha256_file(path)
    if expected != actual:
        raise ValueError(f"JSON sidecar mismatch: {path}")
    return common.load_json(path)


def active_groups(sample: dict[str, Any], selected_groups: set[str]) -> set[str]:
    counts = sample.get("semantic_group_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("sample semantic_group_counts must be an object")
    return {
        group
        for group in selected_groups
        if isinstance(counts.get(group, 0), (int, float)) and counts.get(group, 0) > 0
    }


def stable_exit_candidates(
    samples: Sequence[dict[str, Any]],
    selected_groups: Sequence[str],
    *,
    min_absent_run_samples: int,
) -> list[dict[str, Any]]:
    if min_absent_run_samples <= 0:
        raise ValueError("minimum absent run must be positive")
    if len(samples) < 2:
        return []
    group_set = set(selected_groups)
    ordered = sorted(samples, key=lambda sample: int(sample["timestamp_ms"]))
    candidates: list[dict[str, Any]] = []
    for index in range(len(ordered) - 1):
        present = active_groups(ordered[index], group_set)
        if not present or active_groups(ordered[index + 1], group_set):
            continue
        absent_run = 0
        for following in ordered[index + 1 :]:
            if active_groups(following, group_set):
                break
            absent_run += 1
        if absent_run < min_absent_run_samples:
            continue
        candidates.append({
            "present_timestamp_ms": int(ordered[index]["timestamp_ms"]),
            "absent_timestamp_ms": int(ordered[index + 1]["timestamp_ms"]),
            "present_groups": sorted(present),
            "absent_run_sample_count": absent_run,
            "minimum_absent_run_samples": min_absent_run_samples,
        })
    return candidates


def score_candidates(candidates: Sequence[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    expected_present = int(reference["present_timestamp_ms"])
    expected_absent = int(reference["absent_timestamp_ms"])
    exact = [
        candidate
        for candidate in candidates
        if candidate["present_timestamp_ms"] == expected_present
        and candidate["absent_timestamp_ms"] == expected_absent
    ]
    premature = [
        candidate
        for candidate in candidates
        if candidate["absent_timestamp_ms"] < expected_absent
    ]
    late = [
        candidate
        for candidate in candidates
        if candidate["absent_timestamp_ms"] > expected_absent
    ]
    passed = len(exact) == 1 and not premature and not late and len(candidates) == 1
    return {
        "candidate_count": len(candidates),
        "exact_reference_match_count": len(exact),
        "premature_candidate_count": len(premature),
        "late_candidate_count": len(late),
        "passed": passed,
    }


def activity_window_diagnostics(
    samples: Sequence[dict[str, Any]],
    selected_groups: Sequence[str],
    window_ms: Sequence[int],
) -> dict[str, Any]:
    if len(window_ms) != 2:
        raise ValueError("diagnostic window must contain start and end timestamps")
    start_ms, end_ms = (int(window_ms[0]), int(window_ms[1]))
    if start_ms > end_ms:
        raise ValueError("diagnostic window start must not exceed end")
    selected = set(selected_groups)
    window_samples = sorted(
        (
            sample for sample in samples
            if start_ms <= int(sample["timestamp_ms"]) <= end_ms
        ),
        key=lambda sample: int(sample["timestamp_ms"]),
    )
    active = [bool(active_groups(sample, selected)) for sample in window_samples]
    longest_active = 0
    longest_absent = 0
    current_active = 0
    current_absent = 0
    for value in active:
        if value:
            current_active += 1
            current_absent = 0
            longest_active = max(longest_active, current_active)
        else:
            current_absent += 1
            current_active = 0
            longest_absent = max(longest_absent, current_absent)
    active_count = sum(active)
    sample_count = len(active)
    return {
        "window_ms": [start_ms, end_ms],
        "sample_count": sample_count,
        "active_sample_count": active_count,
        "active_fraction": active_count / sample_count if sample_count else None,
        "longest_active_run_samples": longest_active,
        "longest_absent_run_samples": longest_absent,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    scan = verify_json_sidecar(args.scan)
    review = verify_json_sidecar(args.review)
    if scan.get("schema") != "blindassist_public_video_prompt_free_exit_discovery_v1":
        raise ValueError("unexpected scan schema")
    if review.get("schema") != "blindassist_public_video_exit_candidate_gpt_review_v1":
        raise ValueError("unexpected review schema")
    review_body = review.get("review")
    if not isinstance(review_body, dict):
        raise ValueError("review body is missing")
    source_id = review_body.get("source_id")
    reference = review_body.get("candidate_boundary")
    if not isinstance(source_id, str) or not isinstance(reference, dict):
        raise ValueError("review source or candidate boundary is missing")
    sources = scan.get("sources")
    if not isinstance(sources, list):
        raise ValueError("scan sources are missing")
    matches = [source for source in sources if source.get("source_id") == source_id]
    if len(matches) != 1 or not isinstance(matches[0].get("samples"), list):
        raise ValueError("review source does not bind exactly one scan sequence")
    samples = matches[0]["samples"]
    risk_present_window = review_body.get("risk_present_window_ms")
    stable_clear_window = review_body.get("stable_clear_window_ms")
    if not isinstance(risk_present_window, list) or not isinstance(stable_clear_window, list):
        raise ValueError("review risk-present or stable-clear diagnostic window is missing")

    # Freeze all candidates before consuming the review timing reference.
    frozen_candidates = {
        name: stable_exit_candidates(
            samples,
            groups,
            min_absent_run_samples=args.min_absent_run_samples,
        )
        for name, groups in VARIANTS.items()
    }
    variants = {
        name: {
            "semantic_groups": list(VARIANTS[name]),
            "candidates": candidates,
            "score": score_candidates(candidates, reference),
            "diagnostics": {
                "risk_present_window": activity_window_diagnostics(
                    samples, VARIANTS[name], risk_present_window
                ),
                "stable_clear_window": activity_window_diagnostics(
                    samples, VARIANTS[name], stable_clear_window
                ),
            },
        }
        for name, candidates in frozen_candidates.items()
    }
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "inputs": {
            "scan": {"path": str(args.scan.resolve()), "sha256": common.sha256_file(args.scan)},
            "review": {"path": str(args.review.resolve()), "sha256": common.sha256_file(args.review)},
        },
        "contract": {
            "minimum_consecutive_absent_samples": args.min_absent_run_samples,
            "candidate_generation_reads_review": False,
            "learned_parameters": 0,
            "variant_groups_preregistered_in_code": True,
        },
        "reference": {
            "kind": "GPT timestamped multiframe review; not human truth",
            "present_timestamp_ms": int(reference["present_timestamp_ms"]),
            "absent_timestamp_ms": int(reference["absent_timestamp_ms"]),
        },
        "variants": variants,
        "summary": {
            "surface_only_passed": variants["surface_only"]["score"]["passed"],
            "barrier_only_passed": variants["barrier_only"]["score"]["passed"],
            "risk_profile_union_passed": variants["risk_profile_union"]["score"]["passed"],
            "risk_profile_avoids_surface_premature_exit": (
                variants["surface_only"]["score"]["premature_candidate_count"] > 0
                and variants["risk_profile_union"]["score"]["premature_candidate_count"] == 0
                and variants["risk_profile_union"]["score"]["passed"]
            ),
        },
        "evidence_limit": "External licensed vehicle-view challenge with GPT review; not pedestrian event truth, calibration, blind evaluation, or production evidence.",
        "human_event_truth_present": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-absent-run-samples", type=int, default=3)
    args = parser.parse_args()
    if args.min_absent_run_samples <= 0:
        parser.error("minimum absent run must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
