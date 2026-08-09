#!/usr/bin/env python3
"""Exercise the A0 evaluator with self-contained synthetic fixtures."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.research.assistive_geometry.evaluate_b1_a0_synthetic import (
    BANDS,
    EXPECTED_CHECKPOINT_STEPS,
    EXPECTED_SEEDS,
    HORIZONS_M,
    SYNTHETIC_ROLE,
    TERMINALS,
    atomic_write_json,
    atomic_write_text,
    run_package,
    sha256_file,
    state_dict_digest,
    utc_now,
)


SCENARIOS = {
    "pass": TERMINALS["PASS"],
    "checkpoint_incomplete": TERMINALS["CHECKPOINT"],
    "protocol_drift": TERMINALS["PROTOCOL"],
    "missing_horizon": TERMINALS["SCHEMA"],
    "zero_denominator": TERMINALS["DENOMINATOR"],
    "coverage_collapse": TERMINALS["COVERAGE"],
    "best_seed_forbidden": TERMINALS["BEST_SEED"],
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def synthetic_training_protocol() -> dict[str, Any]:
    return {
        "schema": "blindassist_assistive_geometry_b1_a0_formal_train_execution_protocol_v1",
        "protocol_id": "SYNTHETIC_A0_TRAIN_PROTOCOL_FOR_EVALUATOR_DRY_RUN",
        "inputs": {"initialization_checkpoint": {"sha256": "A" * 64}},
        "execution": {
            "seeds": list(EXPECTED_SEEDS),
            "epochs_per_seed": 20,
            "optimizer_steps_per_seed": 6000,
            "retained_checkpoint_epochs": list(EXPECTED_CHECKPOINT_STEPS),
        },
        "firewalls": {
            "development_or_confirmation_content_access": False,
            "best_seed_early_stop_or_selection": False,
        },
        "claim_ceiling": "Synthetic evaluator fixture only.",
    }


def _epoch_history(epoch: int) -> list[dict[str, Any]]:
    checkpoints = sorted(EXPECTED_CHECKPOINT_STEPS)
    history: list[dict[str, Any]] = []
    carry_steps = {
        current: EXPECTED_CHECKPOINT_STEPS[current]
        for current in checkpoints
    }
    for current in range(1, epoch + 1):
        if current in carry_steps:
            steps = carry_steps[current]
        else:
            # The exact retained steps are claim-bearing; intermediate synthetic
            # history only needs to be monotonic and terminate at the retained value.
            end_epoch = min(item for item in checkpoints if item >= current)
            start_epoch = max((item for item in checkpoints if item < current), default=0)
            start_steps = EXPECTED_CHECKPOINT_STEPS.get(start_epoch, 0)
            end_steps = EXPECTED_CHECKPOINT_STEPS[end_epoch]
            fraction = (current - start_epoch) / (end_epoch - start_epoch)
            steps = int(start_steps + fraction * (end_steps - start_steps))
        history.append({"epoch": current, "optimizer_steps_completed": steps, "mean_train_loss": 1.0 / (current + 1), "mean_gradient_norm_before_clip": 0.5, "carry_counts": {"portrait": 0, "landscape": 0}})
    history[-1]["optimizer_steps_completed"] = EXPECTED_CHECKPOINT_STEPS[epoch]
    return history


def _checkpoint_payload(seed: int, epoch: int, steps: int, protocol_sha256: str) -> dict[str, Any]:
    model = {
        "metric_depthart.synthetic_weight": torch.tensor([seed, epoch, steps], dtype=torch.float32),
        "metric_depthart.synthetic_bias": torch.tensor([seed / 100.0], dtype=torch.float32),
    }
    return {
        "schema": "blindassist_assistive_geometry_b1_a0_checkpoint_v1",
        "protocol_sha256": protocol_sha256,
        "initialization_checkpoint_sha256": "A" * 64,
        "seed": seed,
        "next_epoch": epoch,
        "next_optimizer_step": steps + 1,
        "sampler": {"carry": {"portrait": [], "landscape": []}, "formal_epoch_complete": True, "smoke_only": False},
        "model": model,
        "optimizer": {"state": {0: {"step": torch.tensor(steps)}}, "param_groups": [{"params": [0]}]},
        "scheduler": {"total_steps": 6000, "completed_steps": steps, "base_lrs": [2e-5, 1e-4]},
        "scaler": {},
        "rng": {
            "python": random.Random(seed + epoch).getstate(),
            "numpy": np.random.RandomState(seed + epoch).get_state(),
            "torch_cpu": torch.Generator().manual_seed(seed + epoch).get_state(),
            "torch_cuda": [],
        },
        "epoch_history": _epoch_history(epoch),
        "model_state_sha256": state_dict_digest(model),
    }


def synthetic_observations(seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_error = {17: 0.02, 29: 0.04, 43: 0.06}[seed]
    for parent_index, parent in enumerate(("synthetic-parent-a", "synthetic-parent-b")):
        for sequence_index in range(4):
            bands: list[dict[str, Any]] = []
            for band_index, band_name in enumerate(BANDS):
                truth_clearance = 0.7 + 0.15 * band_index + 0.05 * sequence_index + 0.03 * parent_index
                signed_error = seed_error if sequence_index % 2 == 0 else -seed_error
                cells: list[dict[str, Any]] = []
                for horizon_index, horizon in enumerate(HORIZONS_M):
                    occupied = (sequence_index + parent_index + band_index + horizon_index) % 2 == 1
                    truth = "OCCUPIED_OBSERVED" if occupied else "CLEAR_OBSERVED"
                    if parent_index == 1 and sequence_index == 3 and band_name == "right" and horizon == 2.0:
                        truth = "UNKNOWN"
                    cells.append({"horizon_m": horizon, "truth_state": truth, "predicted_state": truth if truth != "UNKNOWN" else "CLEAR_OBSERVED"})
                bands.append(
                    {
                        "band": band_name,
                        "clearance_valid": True,
                        "truth_clearance_m": truth_clearance,
                        "predicted_clearance_m": truth_clearance + signed_error,
                        "cells": cells,
                    }
                )
            rows.append(
                {
                    "schema": "blindassist_assistive_geometry_b1_a0_evaluation_frame_v1",
                    "seed": seed,
                    "data_role": SYNTHETIC_ROLE,
                    "parent_id": parent,
                    "session_id": f"{parent}-session",
                    "frame_id": f"{parent}-{sequence_index:03d}",
                    "sequence_index": sequence_index,
                    "orientation": "portrait" if sequence_index < 2 else "landscape",
                    "environment": "indoor" if parent_index == 0 else "outdoor",
                    "near_field": sequence_index < 2,
                    "low_light_blur": sequence_index == 3,
                    "bands": bands,
                }
            )
    return rows


def build_fixture(root: Path, evaluation_protocol_sha256: str, scenario: str) -> Path:
    root.mkdir(parents=True)
    training_protocol_path = root / "synthetic_training_protocol.json"
    write_json(training_protocol_path, synthetic_training_protocol())
    training_protocol_sha = sha256_file(training_protocol_path)
    seed_runs: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        seed_root = root / f"seed-{seed}"
        seed_root.mkdir()
        receipts: list[dict[str, Any]] = []
        final_model_state_sha = ""
        for epoch, steps in EXPECTED_CHECKPOINT_STEPS.items():
            checkpoint_path = seed_root / f"checkpoint-epoch-{epoch:02d}.pt"
            torch.save(_checkpoint_payload(seed, epoch, steps, training_protocol_sha), checkpoint_path)
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            final_model_state_sha = checkpoint_payload["model_state_sha256"]
            receipts.append({"path": str(checkpoint_path.resolve()), "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path), "epoch": epoch, "optimizer_steps_completed": steps})
        observations = synthetic_observations(seed)
        observations_path = seed_root / "observations.jsonl"
        write_jsonl(observations_path, observations)
        result = {
            "schema": "blindassist_assistive_geometry_b1_a0_formal_train_result_v1",
            "protocol_sha256": training_protocol_sha,
            "runner_sha256": "B" * 64,
            "mode": "formal",
            "seed": seed,
            "workers": 1,
            "completed_optimizer_steps": 6000,
            "final_model_state_sha256": final_model_state_sha,
            "checkpoints": receipts,
            "development_or_confirmation_content_opened": False,
            "teacher_import_or_execution": False,
            "terminal": "B1_A0_DEPTH_ONLY_FORMAL_TRAIN_SEED_COMPLETE",
        }
        result_path = seed_root / "result.json"
        write_json(result_path, result)
        seed_runs.append({"seed": seed, "train_result_path": str(result_path.resolve()), "observations_path": str(observations_path.resolve())})

    package = {
        "schema": "blindassist_assistive_geometry_b1_a0_evaluation_package_v1",
        "package_id": f"SYNTHETIC_A0_EVALUATION_{scenario.upper()}",
        "data_role": SYNTHETIC_ROLE,
        "evaluation_protocol_sha256": evaluation_protocol_sha256,
        "training_protocol_path": str(training_protocol_path.resolve()),
        "training_protocol_sha256": training_protocol_sha,
        "seed_runs": seed_runs,
        "development_or_confirmation_content_access": False,
    }

    if scenario == "checkpoint_incomplete":
        result_path = Path(seed_runs[1]["train_result_path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["checkpoints"] = [item for item in result["checkpoints"] if item["epoch"] != 15]
        write_json(result_path, result)
    elif scenario == "protocol_drift":
        result_path = Path(seed_runs[0]["train_result_path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        receipt = result["checkpoints"][0]
        checkpoint_path = Path(receipt["path"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["protocol_sha256"] = "F" * 64
        torch.save(checkpoint, checkpoint_path)
        receipt["bytes"] = checkpoint_path.stat().st_size
        receipt["sha256"] = sha256_file(checkpoint_path)
        write_json(result_path, result)
    elif scenario in {"missing_horizon", "zero_denominator", "coverage_collapse"}:
        observations_path = Path(seed_runs[0]["observations_path"])
        rows = synthetic_observations(17)
        if scenario == "missing_horizon":
            rows[0]["bands"][1]["cells"].pop(1)
        elif scenario == "zero_denominator":
            for row in rows:
                for band in row["bands"]:
                    for cell in band["cells"]:
                        cell["truth_state"] = "CLEAR_OBSERVED"
                        cell["predicted_state"] = "CLEAR_OBSERVED"
        else:
            for row in rows[:2]:
                for band in row["bands"]:
                    for cell in band["cells"]:
                        if cell["truth_state"] != "UNKNOWN":
                            cell["predicted_state"] = "UNKNOWN"
        write_jsonl(observations_path, rows)
    elif scenario == "best_seed_forbidden":
        package["selected_seed"] = 17

    package_path = root / "evaluation_package.json"
    write_json(package_path, package)
    return package_path


def render_dry_run_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Assistive Geometry A0 evaluator synthetic dry-run",
        "",
        f"Terminal: `{summary['terminal']}`",
        "",
        "| Scenario | Expected terminal | Observed terminal | Match |",
        "|---|---|---|---|",
    ]
    for scenario in summary["scenarios"]:
        lines.append(f"| {scenario['name']} | `{scenario['expected_terminal']}` | `{scenario['observed_terminal']}` | {str(scenario['match']).lower()} |")
    lines.extend(
        [
            "",
            "The pass case validates 12 tiny checkpoints, exact 1499/2999/4499/6000 retained-step identities, all nine band×horizon cells, UNKNOWN exclusion, and three-seed statistics with no selected seed.",
            "",
            "Negative cases retain adjacent `failure.json` and `failure.log` receipts. No Development or Confirmation file is referenced or opened.",
            "",
        ]
    )
    return "\n".join(lines)


def run_dry_run(evaluation_protocol_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"dry-run output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    fixtures_root = output_root / "fixtures"
    scenarios_root = output_root / "scenarios"
    fixtures_root.mkdir()
    scenarios_root.mkdir()
    evaluation_protocol_sha = sha256_file(evaluation_protocol_path)
    observed: list[dict[str, Any]] = []
    for name, expected_terminal in SCENARIOS.items():
        package_path = build_fixture(fixtures_root / name, evaluation_protocol_sha, name)
        result = run_package(package_path, evaluation_protocol_path, scenarios_root / name, expected_data_role=SYNTHETIC_ROLE)
        receipt_path = (scenarios_root / name / ("evaluation_result.json" if result["status"] == "PASS" else "failure.json"))
        observed.append(
            {
                "name": name,
                "expected_terminal": expected_terminal,
                "observed_terminal": result["terminal"],
                "match": result["terminal"] == expected_terminal,
                "receipt": {"path": str(receipt_path.resolve()), "bytes": receipt_path.stat().st_size, "sha256": sha256_file(receipt_path)},
                "failure_log_path": str((scenarios_root / name / "failure.log").resolve()) if result["status"] == "FAIL" else None,
            }
        )
    passed = all(item["match"] for item in observed)
    summary = {
        "schema": "blindassist_assistive_geometry_b1_a0_evaluation_dry_run_result_v1",
        "status": "PASS" if passed else "FAIL",
        "terminal": TERMINALS["PASS"] if passed else TERMINALS["INTERNAL"],
        "evaluation_protocol_sha256": evaluation_protocol_sha,
        "scenario_count": len(observed),
        "scenarios": observed,
        "development_content_opened": False,
        "confirmation_content_opened": False,
        "completed_at": utc_now(),
        "claim_ceiling": "Synthetic evaluator mechanics only; no model quality, Development, Confirmation, deployment, product or safety authority.",
    }
    atomic_write_json(output_root / "dry_run_result.json", summary)
    atomic_write_text(output_root / "report.md", render_dry_run_report(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = run_dry_run(args.evaluation_protocol.resolve(), args.output_root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
