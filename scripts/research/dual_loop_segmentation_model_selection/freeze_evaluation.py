"""Freeze R1 model-selection identities immediately before fresh evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROTOCOL_ID = "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def file_record(repo_root: Path, path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "relative_path": path.relative_to(repo_root).as_posix(), "sha256": sha256_file(path)}


def candidate_record(repo_root: Path, *, name: str, config: str, training_report: str, checkpoint: str, tflite: str, tflite_receipt: str, runtime: str) -> dict[str, Any]:
    config_path = repo_root / config
    training_path = repo_root / training_report
    checkpoint_path = repo_root / checkpoint
    tflite_path = repo_root / tflite
    receipt_path = repo_root / tflite_receipt
    runtime_path = repo_root / runtime
    training = read_json(training_path)
    receipt = read_json(receipt_path)
    runtime_receipt = read_json(runtime_path)
    if training.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"{name}: training report is not R1")
    if receipt.get("protocol_id") != PROTOCOL_ID or receipt.get("status") != "INT8_TFLITE_EXPORTED":
        raise ValueError(f"{name}: INT8 receipt is not complete R1")
    if receipt.get("tflite_sha256") != sha256_file(tflite_path):
        raise ValueError(f"{name}: TFLite SHA256 differs from receipt")
    if receipt.get("fresh_holdout_consumed") is not False:
        raise ValueError(f"{name}: conversion receipt consumed fresh holdout")
    if runtime_receipt.get("protocol_id") != PROTOCOL_ID or runtime_receipt.get("status") != "RUNTIME_BENCHMARK_COMPLETE":
        raise ValueError(f"{name}: runtime receipt is not complete R1")
    if runtime_receipt.get("model_sha256") != sha256_file(tflite_path):
        raise ValueError(f"{name}: runtime receipt model SHA256 differs")
    selected_checkpoint_sha = training.get("selected_checkpoint_sha256")
    if selected_checkpoint_sha != sha256_file(checkpoint_path):
        raise ValueError(f"{name}: selected checkpoint SHA256 differs from training report")
    return {
        "model_id": training.get("model_id"),
        "implementation_identity": training.get("implementation_identity"),
        "config": file_record(repo_root, config_path),
        "training_report": file_record(repo_root, training_path),
        "selected_checkpoint": file_record(repo_root, checkpoint_path),
        "selected_seed": training.get("selected_seed"),
        "tflite": file_record(repo_root, tflite_path),
        "tflite_receipt": file_record(repo_root, receipt_path),
        "runtime_receipt": file_record(repo_root, runtime_path),
    }


def run(repo_root: Path, output: Path) -> dict[str, Any]:
    protocol_path = repo_root / "docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1_PROTOCOL_2026-07-31.json"
    ledger_path = repo_root / "docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1_DATASET_ROLE_LEDGER_2026-07-31.json"
    training_manifest = repo_root / "test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713/training_manifest.jsonl"
    fresh_manifest = repo_root / "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/fresh_holdout/manifest.jsonl"
    fresh_freeze = repo_root / "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/fresh_holdout/freeze_receipt.json"
    dev_trace = repo_root / "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/dev/yolo_trace.jsonl"
    fresh_trace = repo_root / "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/fresh_holdout/yolo_trace.jsonl"
    protocol = read_json(protocol_path)
    ledger = read_json(ledger_path)
    fresh_freeze_value = read_json(fresh_freeze)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("status") != "DESIGN_FROZEN":
        raise ValueError("protocol is not frozen R1")
    if ledger.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("dataset role ledger is not bound to R1")
    if fresh_freeze_value.get("status") != "FRESH_FORMAL_HOLDOUT_FROZEN":
        raise ValueError("fresh holdout is not frozen")
    candidates = {
        "ddrnet23_slim": candidate_record(
            repo_root,
            name="ddrnet23_slim",
            config="configs/dual_loop_segmentation_model_selection_r1/ddrnet23_slim.json",
            training_report="artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/ddrnet23_slim/fp32/training_report.json",
            checkpoint="artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/ddrnet23_slim/fp32/fp32_checkpoint.pt",
            tflite="artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/ddrnet23_slim/int8/model_int8.tflite",
            tflite_receipt="artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/ddrnet23_slim/int8/model_int8.receipt.json",
            runtime="artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/dev/ddrnet23_slim/runtime.json",
        ),
        "segformer_b0": candidate_record(
            repo_root,
            name="segformer_b0",
            config="configs/dual_loop_segmentation_model_selection_r1/segformer_b0.json",
            training_report="artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/segformer_b0/fp32/training_report.json",
            checkpoint="artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/segformer_b0/fp32/fp32_checkpoint.pt",
            tflite="artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/segformer_b0/int8/model_int8_native_v4.tflite",
            tflite_receipt="artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/segformer_b0/int8/model_int8_native_v4.receipt.json",
            runtime="artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/dev/segformer_b0/runtime.json",
        ),
    }
    result = {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.formal_freeze_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "FORMAL_MODEL_SELECTION_INPUTS_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "DEVELOPMENT_HOST_SOURCE_NATIVE_MODEL_SELECTION_ONLY",
        "fresh_holdout_truth_accessed_before_freeze": False,
        "protocol": file_record(repo_root, protocol_path),
        "dataset_role_ledger": file_record(repo_root, ledger_path),
        "training_manifest": file_record(repo_root, training_manifest),
        "fresh_holdout": {
            "manifest": file_record(repo_root, fresh_manifest),
            "freeze_receipt": file_record(repo_root, fresh_freeze),
            "yolo_trace": file_record(repo_root, fresh_trace),
            "formal_frame_count": 200,
        },
        "shared_dev_yolo_trace": file_record(repo_root, dev_trace),
        "candidates": candidates,
        "evaluation_code": {
            name: file_record(repo_root, repo_root / "scripts/research/dual_loop_segmentation_model_selection" / name)
            for name in ("evaluate_model_selection.py", "benchmark_runtime.py", "convert_int8.py")
        },
        "operator": {
            "hazard_classes": ["boundary_step_curb", "obstacle"],
            "candidate_mask": "segmentation_hazard_minus_frozen_YOLO_box_union",
            "unknown_nonwalkable": "separate_ablation_only",
            "threshold_tuning_after_freeze": False,
            "active_reminder_or_event_logic": False,
        },
        "next_authorized_action": "one_shot_fresh_formal_evaluation_then_independent_validation",
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite freeze receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    output.with_suffix(".sha256.json").write_text(
        json.dumps({"sha256": sha256_file(output)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    result = run(args.repo_root.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
