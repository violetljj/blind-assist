#!/usr/bin/env python3
"""Evaluate a fixed present/uncertain/clear lifecycle on a public-video scan.

The state machine has no learned parameters. Entry requires two active samples
inside a three-sample window; an active event becomes uncertain on absence and
clear only after three consecutive absent samples. A single post-exit semantic
spike therefore cannot reopen an event. GPT timing is opened only after all
state transitions are frozen and is used as discovery-only interval reference.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import public_video_tristate_contract as prospective


SCHEMA = "blindassist_public_video_tristate_lifecycle_external_challenge_v1"
SELECTED_GROUPS = prospective.SELECTED_GROUPS


def verify_json_sidecar(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    mil.reject_independent_direction(resolved)
    sidecar = Path(str(resolved) + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"JSON or sidecar is missing: {resolved}")
    if sidecar.read_text(encoding="ascii").strip().lower() != common.sha256_file(resolved):
        raise ValueError(f"JSON sidecar mismatch: {resolved}")
    return common.load_json(resolved)


def sample_is_active(sample: dict[str, Any], selected_groups: set[str]) -> bool:
    counts = sample.get("semantic_group_counts", {})
    if not isinstance(counts, dict):
        raise ValueError("sample semantic_group_counts must be an object")
    return any(float(counts.get(group, 0)) > 0 for group in selected_groups)


def tristate_exit_intervals(
    samples: Sequence[dict[str, Any]],
    selected_groups: Sequence[str],
    *,
    entry_window_samples: int = 3,
    entry_min_active_samples: int = 2,
    clear_absent_samples: int = 3,
) -> dict[str, Any]:
    if (
        entry_window_samples <= 0
        or entry_min_active_samples <= 0
        or entry_min_active_samples > entry_window_samples
        or clear_absent_samples <= 0
    ):
        raise ValueError("invalid tri-state lifecycle parameters")
    ordered = sorted(samples, key=lambda sample: int(sample["timestamp_ms"]))
    groups = set(selected_groups)
    entry_window: deque[tuple[int, bool]] = deque(maxlen=entry_window_samples)
    state = "clear"
    event_entry_ms: int | None = None
    last_active_ms: int | None = None
    first_absent_ms: int | None = None
    absent_run = 0
    intervals: list[dict[str, Any]] = []
    state_counts = {"clear": 0, "present": 0, "uncertain": 0}

    for sample in ordered:
        timestamp_ms = int(sample["timestamp_ms"])
        active = sample_is_active(sample, groups)
        entry_window.append((timestamp_ms, active))

        if state == "clear":
            if (
                len(entry_window) == entry_window_samples
                and sum(value for _, value in entry_window) >= entry_min_active_samples
            ):
                state = "present"
                event_entry_ms = next(ts for ts, value in entry_window if value)
                last_active_ms = timestamp_ms if active else max(
                    ts for ts, value in entry_window if value
                )
                first_absent_ms = None
                absent_run = 0
        elif active:
            state = "present"
            last_active_ms = timestamp_ms
            first_absent_ms = None
            absent_run = 0
        else:
            if state == "present":
                state = "uncertain"
                first_absent_ms = timestamp_ms
                absent_run = 1
            else:
                absent_run += 1
            if absent_run >= clear_absent_samples:
                assert event_entry_ms is not None
                assert last_active_ms is not None
                assert first_absent_ms is not None
                intervals.append({
                    "event_entry_timestamp_ms": event_entry_ms,
                    "last_active_timestamp_ms": last_active_ms,
                    "first_absent_timestamp_ms": first_absent_ms,
                    "confirmed_clear_timestamp_ms": timestamp_ms,
                    "clear_absent_sample_count": absent_run,
                })
                state = "clear"
                event_entry_ms = None
                last_active_ms = None
                first_absent_ms = None
                absent_run = 0
                entry_window.clear()
        state_counts[state] += 1

    open_event = None
    if state != "clear":
        assert event_entry_ms is not None
        assert last_active_ms is not None
        open_event = {
            "event_entry_timestamp_ms": event_entry_ms,
            "last_active_timestamp_ms": last_active_ms,
            "first_absent_timestamp_ms": first_absent_ms,
            "terminal_state": state,
        }
    return {
        "intervals": intervals,
        "open_event": open_event,
        "terminal_state": state,
        "state_sample_counts": state_counts,
    }


def score_intervals(
    intervals: Sequence[dict[str, Any]], reference: dict[str, Any]
) -> dict[str, Any]:
    reference_present = int(reference["present_timestamp_ms"])
    reference_absent = int(reference["absent_timestamp_ms"])
    containing = [
        interval for interval in intervals
        if int(interval["last_active_timestamp_ms"]) <= reference_present
        and int(interval["confirmed_clear_timestamp_ms"]) >= reference_absent
        and int(interval["first_absent_timestamp_ms"]) <= reference_absent
    ]
    return {
        "interval_count": len(intervals),
        "reference_containing_interval_count": len(containing),
        "passed": len(intervals) == 1 and len(containing) == 1,
    }


def activity_window_diagnostics(
    samples: Sequence[dict[str, Any]],
    selected_groups: Sequence[str],
    window_ms: Sequence[int],
) -> dict[str, Any]:
    if len(window_ms) != 2:
        raise ValueError("diagnostic window must contain start and end timestamps")
    start_ms, end_ms = int(window_ms[0]), int(window_ms[1])
    if start_ms > end_ms:
        raise ValueError("diagnostic window start exceeds end")
    groups = set(selected_groups)
    selected = [
        sample for sample in samples
        if start_ms <= int(sample["timestamp_ms"]) <= end_ms
    ]
    active_count = sum(sample_is_active(sample, groups) for sample in selected)
    return {
        "window_ms": [start_ms, end_ms],
        "sample_count": len(selected),
        "active_sample_count": active_count,
        "active_fraction": active_count / len(selected) if selected else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    scan = verify_json_sidecar(args.scan)
    review = verify_json_sidecar(args.review)
    contract: dict[str, Any] | None = None
    contract_attestation: dict[str, str] | None = None
    if args.contract is not None:
        contract, contract_attestation = prospective.load_contract(args.contract)
        scan_contract = scan.get("prospective_contract")
        if scan_contract != contract_attestation:
            raise ValueError("scan is not bound to the supplied prospective contract")
        lifecycle_contract = contract["lifecycle"]
        if (
            args.entry_window_samples != lifecycle_contract["entry_window_samples"]
            or args.entry_min_active_samples != lifecycle_contract["entry_min_active_samples"]
            or args.clear_absent_samples != lifecycle_contract["clear_absent_samples"]
        ):
            raise ValueError("lifecycle arguments differ from the frozen prospective contract")
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
        raise ValueError("review source or reference boundary is missing")
    if not isinstance(review_body.get("risk_present_window_ms"), list) or not isinstance(
        review_body.get("stable_clear_window_ms"), list
    ):
        raise ValueError("review diagnostic windows are missing")
    sources = scan.get("sources")
    if not isinstance(sources, list):
        raise ValueError("scan sources are missing")
    matches = [source for source in sources if source.get("source_id") == source_id]
    if len(matches) != 1 or not isinstance(matches[0].get("samples"), list):
        raise ValueError("review source does not bind exactly one scan sequence")
    samples = matches[0]["samples"]

    # Freeze all state transitions before consuming the review reference.
    lifecycle = tristate_exit_intervals(
        samples,
        SELECTED_GROUPS,
        entry_window_samples=args.entry_window_samples,
        entry_min_active_samples=args.entry_min_active_samples,
        clear_absent_samples=args.clear_absent_samples,
    )
    score = score_intervals(lifecycle["intervals"], reference)
    diagnostics = {
        "risk_present_window": activity_window_diagnostics(
            samples, SELECTED_GROUPS, review_body["risk_present_window_ms"]
        ),
        "stable_clear_window": activity_window_diagnostics(
            samples, SELECTED_GROUPS, review_body["stable_clear_window_ms"]
        ),
    }
    prospective_acceptance: dict[str, Any] | None = None
    if contract is not None:
        acceptance = contract["acceptance"]
        risk_fraction = diagnostics["risk_present_window"]["active_fraction"]
        clear_fraction = diagnostics["stable_clear_window"]["active_fraction"]
        if risk_fraction is None or clear_fraction is None:
            raise ValueError("prospective diagnostic window has no scan samples")
        prospective_acceptance = {
            "interval_gate_passed": score["passed"],
            "risk_present_active_fraction": risk_fraction,
            "minimum_risk_present_active_fraction": acceptance[
                "minimum_risk_present_active_fraction"
            ],
            "risk_present_coverage_passed": risk_fraction >= acceptance[
                "minimum_risk_present_active_fraction"
            ],
            "stable_clear_active_fraction": clear_fraction,
            "maximum_stable_clear_active_fraction": acceptance[
                "maximum_stable_clear_active_fraction"
            ],
            "stable_clear_false_activation_passed": clear_fraction <= acceptance[
                "maximum_stable_clear_active_fraction"
            ],
        }
        prospective_acceptance["passed"] = all([
            prospective_acceptance["interval_gate_passed"],
            prospective_acceptance["risk_present_coverage_passed"],
            prospective_acceptance["stable_clear_false_activation_passed"],
        ])
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "inputs": {
            "scan": {"path": str(args.scan.resolve()), "sha256": common.sha256_file(args.scan)},
            "review": {"path": str(args.review.resolve()), "sha256": common.sha256_file(args.review)},
        },
        "prospective_contract": contract_attestation,
        "contract": {
            "selected_groups": list(SELECTED_GROUPS),
            "entry_window_samples": args.entry_window_samples,
            "entry_min_active_samples": args.entry_min_active_samples,
            "clear_absent_samples": args.clear_absent_samples,
            "learned_parameters": 0,
            "candidate_generation_reads_review": False,
        },
        "reference": {
            "kind": "GPT timestamped multiframe review; not human truth",
            "present_timestamp_ms": int(reference["present_timestamp_ms"]),
            "absent_timestamp_ms": int(reference["absent_timestamp_ms"]),
        },
        "lifecycle": lifecycle,
        "diagnostics": diagnostics,
        "score": score,
        "prospective_acceptance": prospective_acceptance,
        "evidence_limit": "Exploratory licensed vehicle-view mechanism challenge only; not training truth, pedestrian event truth, calibration, blind evaluation, Android runtime authorization, or production evidence.",
        "human_event_truth_present": False,
        "training_execution_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(
        common.sha256_file(args.output) + "\n", encoding="ascii"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--entry-window-samples", type=int, default=3)
    parser.add_argument("--entry-min-active-samples", type=int, default=2)
    parser.add_argument("--clear-absent-samples", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    summary = report["prospective_acceptance"] or report["score"]
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
