#!/usr/bin/env python3
"""Validate FP-aware DDRNet result bindings and recompute every reported metric."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import CANDIDATE_ID, PROTOCOL_ID
from .contract import (
    PENDING_VALIDATION_TERMINAL,
    validate_config_contract,
    validate_config_sha256,
)
from .evaluate import (
    EVALUATION_ROLES,
    aggregate_model,
    build_frame_row,
    compare_seed,
    load_evaluation_inputs,
    load_images_and_truth,
    predict_checkpoint,
    read_json,
    read_jsonl,
    resolve,
    unpack_ids,
    validate_checkpoint_payload,
    write_json,
)
from .models import sha256_file


def _validate_core(
    *,
    repo_root: Path,
    config_path: Path,
    result_path: Path,
    output_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = resolve(repo_root, config_path)
    result_path = resolve(repo_root, result_path)
    output_path = resolve(repo_root, output_path)
    config = read_json(config_path)
    result = read_json(result_path)
    errors: list[str] = []
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    try:
        validate_config_sha256(sha256_file(config_path))
        validate_config_contract(config)
        checks += 1
    except Exception as exc:
        checks += 1
        errors.append(f"config contract invalid: {exc}")
        receipt = {
            "schema_version": "blindassist.dual_loop_segmentation_fp_aware_ddrnet_r0.validation.v1",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": CANDIDATE_ID,
            "status": "INVALID",
            "terminal": "FP_WEIGHTED_SAMPLING_NOT_EVALUABLE",
            "checks": checks,
            "errors": errors,
            "result_sha256": sha256_file(result_path),
        }
        write_json(output_path, receipt)
        return receipt
    check(result.get("protocol_id") == PROTOCOL_ID, "result protocol mismatch")
    check(result.get("candidate_id") == CANDIDATE_ID, "result candidate mismatch")
    check(result.get("status") == "EVALUATION_COMPLETE_UNVALIDATED", "result status mismatch")
    check(result.get("terminal") == PENDING_VALIDATION_TERMINAL, "unvalidated terminal mismatch")
    check(result.get("config_sha256") == sha256_file(config_path), "result config SHA mismatch")
    training_report_path = Path(str(result.get("training_report", "")))
    check(training_report_path.is_file(), "training report missing")
    training_report = read_json(training_report_path) if training_report_path.is_file() else {}
    if training_report_path.is_file():
        check(
            training_report.get("protocol_id") == PROTOCOL_ID,
            "training report protocol mismatch",
        )
        check(
            training_report.get("candidate_id") == CANDIDATE_ID,
            "training report candidate mismatch",
        )
        check(
            training_report.get("status") == "TRAINING_COMPLETE",
            "training report status mismatch",
        )
        check(
            training_report.get("config_sha256") == sha256_file(config_path),
            "training report config binding mismatch",
        )
        check(
            training_report.get("cross_seed_selection") == "FORBIDDEN_NOT_PERFORMED",
            "training report cross-seed selection contract mismatch",
        )
    if training_report_path.is_file():
        check(
            result.get("training_report_sha256") == sha256_file(training_report_path),
            "training report SHA mismatch",
        )
    frames_path = Path(str(result.get("frame_predictions", "")))
    check(frames_path.is_file(), "frame predictions missing")
    if not frames_path.is_file():
        receipt = {
            "schema_version": "blindassist.dual_loop_segmentation_fp_aware_ddrnet_r0.validation.v1",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": CANDIDATE_ID,
            "status": "INVALID",
            "terminal": "FP_WEIGHTED_SAMPLING_NOT_EVALUABLE",
            "checks": checks,
            "errors": errors,
            "result_sha256": sha256_file(result_path),
        }
        write_json(output_path, receipt)
        return receipt
    check(result.get("frame_predictions_sha256") == sha256_file(frames_path), "frame SHA mismatch")
    frame_rows = read_jsonl(frames_path)
    check(len(frame_rows) == 1920, "expected 320 frames x 2 arms x 3 seeds")
    check(result.get("frame_row_count") == len(frame_rows), "reported frame row count mismatch")

    manifest_rows, traces = load_evaluation_inputs(repo_root, config)
    manifest_by_id = {str(row["id"]): row for row in manifest_rows}
    check(len(manifest_by_id) == 320, "evaluation manifest identity count mismatch")
    view_manifest = resolve(repo_root, config["evaluation"]["canonical_view_manifest"]["path"])
    view_root = view_manifest.parent
    images, truths, source_sizes = load_images_and_truth(
        repo_root,
        view_root,
        manifest_rows,
    )
    truth_by_id = {
        str(manifest["id"]): truth
        for manifest, truth in zip(manifest_rows, truths, strict=True)
    }
    source_size_by_id = {
        str(manifest["id"]): source_size
        for manifest, source_size in zip(manifest_rows, source_sizes, strict=True)
    }
    training_seed_reports = {
        int(item["seed"]): item for item in training_report.get("seed_reports", [])
    }
    check(
        set(training_seed_reports) == set(config["training"]["seeds"]),
        "candidate training seed set mismatch",
    )
    expected_checkpoint_sha: dict[tuple[int, str], str] = {}
    checkpoint_paths: dict[tuple[int, str], Path] = {}
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        baseline_binding = config["inputs"][f"r1_baseline_seed_{seed}"]
        expected_checkpoint_sha[(seed, "R1_BASELINE")] = baseline_binding["sha256"]
        checkpoint_paths[(seed, "R1_BASELINE")] = resolve(
            repo_root,
            baseline_binding["path"],
        )
        candidate = training_seed_reports.get(seed)
        check(candidate is not None, f"candidate training report missing seed {seed}")
        if candidate is not None:
            expected_checkpoint_sha[(seed, "FP_AWARE_CANDIDATE")] = candidate[
                "checkpoint_sha256"
            ]
            checkpoint_paths[(seed, "FP_AWARE_CANDIDATE")] = Path(
                str(candidate["checkpoint"])
            ).resolve()

    architecture_binding = config["inputs"]["ddrnet_architecture"]
    source_binding = config["inputs"]["ddrnet_source_checkpoint"]
    architecture = resolve(repo_root, architecture_binding["path"])
    source_checkpoint = resolve(repo_root, source_binding["path"])
    check(architecture.is_file(), "DDRNet architecture source missing")
    check(source_checkpoint.is_file(), "DDRNet source checkpoint missing")
    if architecture.is_file():
        check(
            sha256_file(architecture) == architecture_binding["sha256"],
            "DDRNet architecture SHA mismatch",
        )
    if source_checkpoint.is_file():
        check(
            sha256_file(source_checkpoint) == source_binding["sha256"],
            "DDRNet source checkpoint SHA mismatch",
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for checkpoint-output validation but is unavailable")

    inferred_predictions: dict[tuple[int, str, str], np.ndarray] = {}
    if architecture.is_file() and source_checkpoint.is_file():
        for identity, checkpoint_path in sorted(checkpoint_paths.items()):
            seed, arm = identity
            check(checkpoint_path.is_file(), f"{arm} seed {seed}: checkpoint missing")
            if not checkpoint_path.is_file():
                continue
            actual_checkpoint_sha = sha256_file(checkpoint_path)
            check(
                actual_checkpoint_sha == expected_checkpoint_sha[identity],
                f"{arm} seed {seed}: checkpoint SHA mismatch",
            )
            try:
                validate_checkpoint_payload(
                    checkpoint_path,
                    expected_seed=seed,
                    candidate=arm == "FP_AWARE_CANDIDATE",
                )
                checks += 1
            except Exception as exc:
                checks += 1
                errors.append(f"{arm} seed {seed}: checkpoint payload invalid: {exc}")
                continue
            if actual_checkpoint_sha != expected_checkpoint_sha[identity]:
                continue
            predictions = predict_checkpoint(
                architecture=architecture,
                source_checkpoint=source_checkpoint,
                checkpoint=checkpoint_path,
                images=images,
                device=device,
                batch_size=int(config["evaluation"]["batch_size"]),
            )
            check(
                len(predictions) == len(manifest_rows),
                f"{arm} seed {seed}: inference row count mismatch",
            )
            for manifest, prediction in zip(manifest_rows, predictions, strict=True):
                inferred_predictions[(seed, arm, str(manifest["id"]))] = prediction

    seen: set[tuple[int, str, str]] = set()
    recomputed_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(frame_rows):
        check(row.get("protocol_id") == PROTOCOL_ID, f"row {row_index}: protocol mismatch")
        check(row.get("candidate_id") == CANDIDATE_ID, f"row {row_index}: candidate mismatch")
        check(row.get("formal_authority") is False, f"row {row_index}: formal authority drifted")
        seed = int(row.get("seed", -1))
        arm = str(row.get("arm", ""))
        view_row_id = str(row.get("view_row_id", ""))
        identity = (seed, arm, view_row_id)
        check(identity not in seen, f"row {row_index}: duplicate identity")
        seen.add(identity)
        manifest = manifest_by_id.get(view_row_id)
        check(manifest is not None, f"row {row_index}: missing manifest identity")
        check(
            row.get("checkpoint_sha256") == expected_checkpoint_sha.get((seed, arm)),
            f"row {row_index}: checkpoint binding mismatch",
        )
        if manifest is None:
            continue
        check(row.get("role") in EVALUATION_ROLES, f"row {row_index}: forbidden role")
        check(row.get("role") == manifest["role"], f"row {row_index}: role mismatch")
        check(row.get("session_id") == manifest["session_id"], f"row {row_index}: session mismatch")
        check(row.get("image_sha256") == manifest["image_sha256"], f"row {row_index}: image mismatch")
        check(
            row.get("canonical_mask_sha256") == manifest["canonical_mask_sha256"],
            f"row {row_index}: truth hash mismatch",
        )
        try:
            predicted_ids = unpack_ids(str(row["predicted_ids_zlib_base64"]))
            check(bool((predicted_ids <= 3).all()), f"row {row_index}: predicted ID outside 0..3")
        except Exception as exc:
            errors.append(f"row {row_index}: prediction decode failed: {exc}")
            continue
        inferred = inferred_predictions.get(identity)
        check(inferred is not None, f"row {row_index}: checkpoint inference missing")
        if inferred is not None:
            check(
                np.array_equal(predicted_ids, inferred),
                f"row {row_index}: prediction does not match checkpoint inference",
            )
        truth_ids = truth_by_id[view_row_id]
        source_size = source_size_by_id[view_row_id]
        key = (
            str(manifest["source_id"]),
            int(manifest["frame_id"]),
            str(manifest["image_sha256"]),
        )
        expected = build_frame_row(
            manifest=manifest,
            trace=traces[key],
            source_size=source_size,
            truth_ids=truth_ids,
            predicted_ids=predicted_ids,
            seed=seed,
            arm=arm,
            checkpoint_sha256=str(row["checkpoint_sha256"]),
        )
        check(row == expected, f"row {row_index}: recomputed frame row mismatch")
        recomputed_rows.append(expected)

    check(len(seen) == 1920, "frame identity set is not exhaustive")
    by_seed_arm: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in recomputed_rows:
        by_seed_arm[(int(row["seed"]), str(row["arm"]))].append(row)
    recomputed_seed_summaries: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        baseline = aggregate_model(by_seed_arm[(seed, "R1_BASELINE")])
        candidate = aggregate_model(by_seed_arm[(seed, "FP_AWARE_CANDIDATE")])
        comparison = compare_seed(
            seed=seed,
            baseline=baseline,
            candidate=candidate,
            gates=config["evaluation"]["gates"],
        )
        summary = {
            "baseline": baseline,
            "candidate": candidate,
            "comparison": comparison,
        }
        recomputed_seed_summaries[str(seed)] = summary
        comparisons.append(comparison)
        check(
            result.get("seed_summaries", {}).get(str(seed)) == summary,
            f"seed {seed}: aggregate summary mismatch",
        )
    all_pass = bool(comparisons) and all(item["all_nine_gates_passed"] for item in comparisons)
    worst = min(comparisons, key=lambda item: (item["minimum_gate_margin"], item["seed"]))
    terminal = (
        "FP_WEIGHTED_SAMPLING_SUPPORTED_DEVELOPMENT_ONLY"
        if all_pass
        else "FP_WEIGHTED_SAMPLING_NOT_SUPPORTED"
    )
    check(result.get("all_three_seeds_passed") is all_pass, "all-seed decision mismatch")
    check(result.get("worst_seed") == worst["seed"], "worst seed mismatch")
    check(
        result.get("provisional_metric_terminal") == terminal,
        "provisional metric terminal mismatch",
    )
    authority = result.get("authority", {})
    check(authority.get("stage") == "DEVELOPMENT", "authority stage mismatch")
    check(
        authority.get("fresh_or_confirmation_outcome_accessed") is False,
        "fresh/confirmation access claimed",
    )
    check(authority.get("int8_or_runtime_authority") is False, "INT8/runtime authority drifted")
    check(authority.get("android_or_alert_authority") is False, "Android/alert authority drifted")
    final_terminal = "FP_WEIGHTED_SAMPLING_NOT_EVALUABLE" if errors else terminal
    receipt = {
        "schema_version": "blindassist.dual_loop_segmentation_fp_aware_ddrnet_r0.validation.v1",
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "status": "VALID" if not errors else "INVALID",
        "terminal": final_terminal,
        "checks": checks,
        "errors": errors,
        "result_sha256": sha256_file(result_path),
        "frame_predictions_sha256": sha256_file(frames_path),
        "frame_row_count": len(frame_rows),
        "recomputed_metric_terminal": terminal,
        "recomputed_all_three_seeds_passed": all_pass,
        "recomputed_worst_seed": worst["seed"],
    }
    write_json(output_path, receipt)
    return receipt


def validate(
    *,
    repo_root: Path,
    config_path: Path,
    result_path: Path,
    output_path: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    resolved_repo_root = repo_root.resolve()
    resolved_output = resolve(resolved_repo_root, output_path)
    if resolved_output.exists():
        raise FileExistsError(f"refusing to overwrite validation output: {resolved_output}")
    resolved_result = resolve(resolved_repo_root, result_path)
    try:
        return _validate_core(
            repo_root=resolved_repo_root,
            config_path=config_path,
            result_path=resolved_result,
            output_path=resolved_output,
            device=torch.device(device),
        )
    except Exception as exc:
        receipt = {
            "schema_version": "blindassist.dual_loop_segmentation_fp_aware_ddrnet_r0.validation.v1",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": CANDIDATE_ID,
            "status": "INVALID",
            "terminal": "FP_WEIGHTED_SAMPLING_NOT_EVALUABLE",
            "checks": 1,
            "errors": [f"validation execution failed closed: {type(exc).__name__}: {exc}"],
        }
        if resolved_result.is_file():
            receipt["result_sha256"] = sha256_file(resolved_result)
        write_json(resolved_output, receipt)
        return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path.cwd())
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    receipt = validate(
        repo_root=Path(args.repo_root),
        config_path=Path(args.config),
        result_path=Path(args.result),
        output_path=Path(args.output),
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "checks": receipt["checks"],
                "error_count": len(receipt["errors"]),
                "terminal": receipt.get("terminal"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
