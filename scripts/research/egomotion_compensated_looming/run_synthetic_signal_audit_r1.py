from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    sha256_file,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.evaluation import (
    IMPLEMENTATION_REVISION,
    run_trial,
    summarize_and_decide,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.receipt import (
    build_receipt,
    runtime_summary,
    validate_existing,
    write_json,
    write_jsonl,
)
from scripts.research.egomotion_compensated_looming.rcle_minimal_r1.visualization import (
    plot_closing_error,
    plot_expansion_curve,
    plot_paired_rotation_leakage,
    render_flow_comparison,
    save_representative_sequence,
    write_main_tables,
    write_markdown_report,
)


PREREGISTRATION_PATH = (
    REPO_ROOT
    / "scripts"
    / "research"
    / "egomotion_compensated_looming"
    / "rcle_minimal_r1"
    / "PHASE_A_COVERAGE_REVISION_R1.md"
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _worker(
    spec_values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_trial(TrialSpec(**spec_values), load_protocol())


def _select_representatives(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    evaluable = [row for row in rows if row["evaluable"]]
    rotation = [
        row
        for row in evaluable
        if row["split"] == "clean"
        and row["motion_family"] == "pure_rotation"
        and row["axis"] in {"yaw", "pitch"}
    ]
    mixed = [
        row
        for row in evaluable
        if row["split"] == "clean"
        and row["motion_family"] == "rotation_plus_scale_up"
    ]
    representatives = {
        "rotation_success": max(
            rotation, key=lambda row: row["paired_leakage_reduction_per_s"]
        ),
        "rotation_worst": min(
            rotation, key=lambda row: row["paired_leakage_reduction_per_s"]
        ),
        "closing_success": min(
            mixed, key=lambda row: row["compensated_closing_error_per_s"]
        ),
        "closing_worst": max(
            mixed, key=lambda row: row["compensated_closing_error_per_s"]
        ),
    }
    failures = [row for row in rows if not row["evaluable"]]
    if failures:
        representatives["coverage_failure"] = failures[0]
    return representatives


def formal_run(
    output_root: Path,
    dataset_root: Path,
    workers: int,
    command: list[str],
) -> dict[str, Any]:
    protocol = load_protocol()
    trials = enumerate_trials(protocol)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"formal output already exists and will not be overwritten: {output_root}"
        )
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(
            f"dataset output already exists and will not be overwritten: {dataset_root}"
        )
    output_root.mkdir(parents=True, exist_ok=False)
    dataset_root.mkdir(parents=True, exist_ok=False)
    preregistration_hash = sha256_file(PREREGISTRATION_PATH)
    started_at = _iso_now()
    state = {
        "status": "RUNNING",
        "implementation_revision": IMPLEMENTATION_REVISION,
        "implementation_preregistration_sha256": preregistration_hash,
        "protocol_sha256": PROTOCOL_SHA256,
        "planned_trials": len(trials),
        "completed_trials": 0,
        "started_at": started_at,
    }
    write_json(output_root / "formal_run_state.json", state)
    write_json(
        output_root / "preregistration.json",
        {
            "implementation_revision": IMPLEMENTATION_REVISION,
            "implementation_preregistration_sha256": preregistration_hash,
            "protocol_sha256": PROTOCOL_SHA256,
            "protocol": protocol,
        },
    )
    rows_by_id: dict[str, dict[str, Any]] = {}
    runtime_by_id: dict[str, dict[str, Any]] = {}
    partial_metrics = output_root / "trial_metrics.partial.jsonl"
    partial_runtime = output_root / "runtime.partial.jsonl"
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
                        output_root / "formal_run_state.json",
                        {
                            **state,
                            "completed_trials": completed,
                            "updated_at": _iso_now(),
                        },
                    )

    ordered_rows = [rows_by_id[trial.trial_id] for trial in trials]
    ordered_runtime = [runtime_by_id[trial.trial_id] for trial in trials]
    write_jsonl(output_root / "trial_metrics.jsonl", ordered_rows)
    write_jsonl(
        output_root / "trial_manifest.jsonl",
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
    write_jsonl(output_root / "runtime_trials.jsonl", ordered_runtime)
    partial_metrics.unlink()
    partial_runtime.unlink()

    summary = summarize_and_decide(ordered_rows, protocol)
    write_json(output_root / "scientific_summary.json", summary)
    runtime = runtime_summary(ordered_runtime)
    write_json(output_root / "runtime_summary.json", runtime)
    write_main_tables(ordered_rows, summary, output_root)
    figures = output_root / "figures"
    plot_paired_rotation_leakage(
        ordered_rows, figures / "paired_rotation_leakage.png"
    )
    plot_closing_error(ordered_rows, figures / "closing_error.png")
    representatives = _select_representatives(ordered_rows)
    write_json(output_root / "representative_cases.json", representatives)
    for name, row in representatives.items():
        if not row["evaluable"]:
            continue
        plot_expansion_curve(row, figures / f"{name}_expansion_curve.png")
        spec = TrialSpec(
            **{
                field: row[field]
                for field in TrialSpec.__dataclass_fields__
            }
        )
        render_flow_comparison(spec, protocol, figures / f"{name}_flow.png")
        save_representative_sequence(
            spec, protocol, dataset_root / name
        )
    write_markdown_report(
        summary,
        runtime,
        output_root / "RCLE_PHASE_A_COVERAGE_REVISION_R1_RESULT.md",
    )
    finished_at = _iso_now()
    write_json(
        output_root / "formal_run_state.json",
        {
            **state,
            "status": "COMPLETE",
            "completed_trials": len(trials),
            "finished_at": finished_at,
            "verdict": summary["verdict"],
        },
    )
    receipt = build_receipt(
        REPO_ROOT,
        output_root,
        dataset_root,
        command,
        summary,
        started_at,
        finished_at,
        workers,
        preregistration_hash,
    )
    write_json(output_root / "receipt.json", receipt)
    validation = validate_existing(REPO_ROOT, output_root, dataset_root)
    write_json(output_root / "receipt_validation.json", validation)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or validate RCLE-Minimal Phase A coverage revision R1."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "artifacts.local/evidence/rcle_minimal_r1/formal_run_r1"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "artifacts.local/datasets/rcle_minimal_r1/formal_run_r1"
        ),
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
    output_root = args.output_root.resolve()
    dataset_root = args.dataset_root.resolve()
    if args.validate_existing:
        validation = validate_existing(REPO_ROOT, output_root, dataset_root)
    else:
        if args.workers <= 0:
            raise ValueError("workers must be positive")
        validation = formal_run(
            output_root,
            dataset_root,
            args.workers,
            [sys.executable, *sys.argv],
        )
    print(
        json.dumps(validation, sort_keys=True, ensure_ascii=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
