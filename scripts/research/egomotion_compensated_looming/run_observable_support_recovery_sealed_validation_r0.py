from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.egomotion_compensated_looming.rcle_minimal.protocol import (
    PROTOCOL_SHA256,
    TrialSpec,
    enumerate_trials,
    load_protocol,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal.receipt import (
    runtime_summary,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_r0.evaluation import (
    IMPLEMENTATION_REVISION,
    run_trial,
    summarize_and_decide,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_r0.receipt import (
    DESIGN_LOCK_SHA256,
)
from scripts.research.egomotion_compensated_looming.rcle_observable_support_validation_r0.receipt import (
    build_receipt,
    validate_existing,
    validate_validation_lock,
    write_json,
    write_jsonl,
)


SEALED_VALIDATION_SEEDS = tuple(range(3000, 3020))
OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts.local"
    / "evidence"
    / "rcle_observable_support_recovery_r0"
    / "sealed_validation_gate_r0"
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sealed_validation_protocol() -> dict[str, Any]:
    protocol = deepcopy(load_protocol())
    protocol["trials"]["seeds"] = list(SEALED_VALIDATION_SEEDS)
    trials = enumerate_trials(protocol)
    if len(trials) != 2520:
        raise ValueError("SEALED_VALIDATION_MATRIX_COUNT_DRIFT")
    if sum(trial.split == "clean" for trial in trials) != 1680:
        raise ValueError("SEALED_VALIDATION_CLEAN_DENOMINATOR_DRIFT")
    if sum(trial.split == "stress" for trial in trials) != 840:
        raise ValueError("SEALED_VALIDATION_STRESS_DENOMINATOR_DRIFT")
    if {trial.seed for trial in trials} != set(SEALED_VALIDATION_SEEDS):
        raise ValueError("SEALED_VALIDATION_SEED_ROLE_DRIFT")
    return protocol


def _worker(
    spec_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_trial(TrialSpec(**spec_values), sealed_validation_protocol())


def formal_run(workers: int, command: list[str]) -> dict[str, Any]:
    validate_validation_lock(REPO_ROOT)
    protocol = sealed_validation_protocol()
    trials = enumerate_trials(protocol)
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            "sealed-validation output already exists; no overwrite, resume, "
            f"patch, or rerun is permitted: {OUTPUT_ROOT}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    started_at = _iso_now()
    state = {
        "schema_version": "rcle.observable_support_recovery.validation_run_state.v1",
        "status": "RUNNING",
        "candidate_id": IMPLEMENTATION_REVISION,
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "seed_role": "SEALED_VALIDATION_ONLY",
        "seeds": list(SEALED_VALIDATION_SEEDS),
        "planned_trials": 2520,
        "completed_trials": 0,
        "started_at": started_at,
    }
    write_json(OUTPUT_ROOT / "formal_run_state.json", state)
    rows_by_id: dict[str, dict[str, Any]] = {}
    runtime_by_id: dict[str, dict[str, Any]] = {}
    partial_metrics = OUTPUT_ROOT / "trial_metrics.partial.jsonl"
    partial_runtime = OUTPUT_ROOT / "runtime.partial.jsonl"
    with partial_metrics.open(
        "w", encoding="utf-8", newline="\n"
    ) as metrics_handle, partial_runtime.open(
        "w", encoding="utf-8", newline="\n"
    ) as runtime_handle:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_worker, spec.to_dict()): spec
                for spec in trials
            }
            for completed, future in enumerate(
                as_completed(futures), start=1
            ):
                spec = futures[future]
                metric, runtime = future.result()
                rows_by_id[spec.trial_id] = metric
                runtime_by_id[spec.trial_id] = runtime
                metrics_handle.write(
                    json.dumps(
                        metric,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                runtime_handle.write(
                    json.dumps(
                        runtime,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                if completed % 20 == 0 or completed == len(trials):
                    metrics_handle.flush()
                    runtime_handle.flush()
                    write_json(
                        OUTPUT_ROOT / "formal_run_state.json",
                        {
                            **state,
                            "completed_trials": completed,
                            "updated_at": _iso_now(),
                        },
                    )

    ordered_rows = [rows_by_id[trial.trial_id] for trial in trials]
    ordered_runtime = [runtime_by_id[trial.trial_id] for trial in trials]
    write_jsonl(OUTPUT_ROOT / "trial_metrics.jsonl", ordered_rows)
    write_jsonl(
        OUTPUT_ROOT / "trial_manifest.jsonl",
        [
            {
                **trial.to_dict(),
                "base_sha256": rows_by_id[trial.trial_id]["base_sha256"],
                "sequence_sha256": rows_by_id[trial.trial_id][
                    "sequence_sha256"
                ],
                "planned_pair_count": rows_by_id[trial.trial_id][
                    "planned_pair_count"
                ],
            }
            for trial in trials
        ],
    )
    write_jsonl(OUTPUT_ROOT / "runtime_trials.jsonl", ordered_runtime)
    partial_metrics.unlink()
    partial_runtime.unlink()
    summary = summarize_and_decide(ordered_rows, protocol)
    write_json(OUTPUT_ROOT / "scientific_summary.json", summary)
    write_json(OUTPUT_ROOT / "runtime_summary.json", runtime_summary(ordered_runtime))
    finished_at = _iso_now()
    terminal = (
        "INDEPENDENT_SYNTHETIC_PASS_ONLY_PHASE_B_REMAINS_CLOSED_PENDING_SEPARATE_DECISION"
        if summary["scientific_gate_pass"]
        else "CLOSE_OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0_WITH_NO_PATCH_OR_RERUN"
    )
    write_json(
        OUTPUT_ROOT / "formal_run_state.json",
        {
            **state,
            "status": "COMPLETE",
            "completed_trials": len(trials),
            "finished_at": finished_at,
            "scientific_gate_pass": bool(summary["scientific_gate_pass"]),
            "terminal_state": terminal,
        },
    )
    receipt = build_receipt(
        REPO_ROOT,
        OUTPUT_ROOT,
        command,
        summary,
        started_at,
        finished_at,
        workers,
    )
    write_json(OUTPUT_ROOT / "receipt.json", receipt)
    validation = validate_existing(REPO_ROOT, OUTPUT_ROOT)
    write_json(OUTPUT_ROOT / "receipt_validation.json", validation)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single locked sealed-validation matrix for "
            "OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0."
        )
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, max(1, os.cpu_count() or 1)),
    )
    parser.add_argument("--validate-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_existing:
        validation = validate_existing(REPO_ROOT, OUTPUT_ROOT)
    else:
        if args.workers <= 0:
            raise ValueError("workers must be positive")
        validation = formal_run(args.workers, [sys.executable, *sys.argv])
    print(json.dumps(validation, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
