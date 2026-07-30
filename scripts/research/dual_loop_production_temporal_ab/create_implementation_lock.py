#!/usr/bin/env python3
"""Create the hash-bound implementation lock after build and device prestart."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL_ID = "DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0"
IMPLEMENTATION_ID = "PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_IMPL_R0"
EXPECTED_DEVICE = ("SM-S9280", "SM8650")
EXPECTED_FRAME_COUNT = 4422
EXPECTED_RGB_TOTAL_BYTES = 2612679375
EXPECTED_INVENTORY_SHA256 = (
    "45621b226b4f6286962ec39c548234f92c3a34331cc4a1b2c413ef0bd3f7dd3b"
)
EXPECTED_INPUT_RECEIPT_SHA256 = (
    "32c80d61bdedf0fa678d09a25e43d84232c4976fd5af0a644bb579c350d4d910"
)
EXPECTED_TRUTH_MEMBERSHIP_SHA256 = (
    "42f36add7863a16210b4c0add41060ede94a50787591f1744bdb9a8aabce5290"
)
LOCKED_FILES = (
    "docs/research/dual-loop/DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_PROTOCOL_2026-07-30.json",
    "core/assist/src/main/java/com/linnan/blindassist/risk/TemporalRiskTracker.kt",
    "core/assist/src/main/java/com/linnan/blindassist/session/AssistEngine.kt",
    "core/assist/src/main/java/com/linnan/blindassist/session/AssistDecisionKernel.kt",
    "core/assist/src/main/java/com/linnan/blindassist/session/AssistSessionCoordinator.kt",
    "core/assist/src/main/java/com/linnan/blindassist/session/SessionTrace.kt",
    "core/assist/src/main/java/com/linnan/blindassist/alert/AlertProfile.kt",
    "core/assist/src/main/java/com/linnan/blindassist/alert/AssistScenario.kt",
    "core/assist/src/main/java/com/linnan/blindassist/model/BoundingBox.kt",
    "core/assist/src/main/java/com/linnan/blindassist/model/Detection.kt",
    "core/assist/src/main/java/com/linnan/blindassist/model/FrameSize.kt",
    "core/assist/src/main/java/com/linnan/blindassist/risk/RiskModels.kt",
    "core/assist/src/main/java/com/linnan/blindassist/risk/DistanceEvidence.kt",
    "core/assist/src/test/java/com/linnan/blindassist/risk/TemporalRiskTrackerTest.kt",
    "core/assist/src/test/java/com/linnan/blindassist/session/AssistEngineTest.kt",
    "device-benchmark/build.gradle.kts",
    "device-benchmark/src/main/java/com/linnan/blindassist/benchmark/ProductionTemporalGeometryFactorialAbDeviceTest.kt",
    "scripts/run_dual_loop_production_temporal_ab.py",
    "scripts/run_dual_loop_production_temporal_ab_device.ps1",
    "scripts/research/dual_loop_production_temporal_ab/create_implementation_lock.py",
    "scripts/research/dual_loop_production_temporal_ab/activate.py",
    "scripts/research/dual_loop_production_temporal_ab/validate_producer.py",
    "scripts/research/dual_loop_production_temporal_ab/evaluate_trace.py",
    "scripts/research/dual_loop_production_temporal_ab/test_tools.py",
    "core/assist/src/main/java/com/linnan/blindassist/risk/ConservativeRiskFusionPolicy.kt",
    "core/assist/src/main/java/com/linnan/blindassist/risk/RiskAnalyzer.kt",
    "core/assist/src/main/java/com/linnan/blindassist/risk/RiskStabilizer.kt",
    "core/assist/src/main/java/com/linnan/blindassist/risk/RiskEventTracker.kt",
    "core/assist/src/main/java/com/linnan/blindassist/feedback/FeedbackPlanner.kt",
    "core/assist/src/main/java/com/linnan/blindassist/feedback/FeedbackModels.kt",
    "core/assist/src/main/java/com/linnan/blindassist/feedback/SpeechStyle.kt",
    "core/assist/src/main/java/com/linnan/blindassist/feedback/VibrationStrength.kt",
    "core/device/src/main/java/com/linnan/blindassist/feedback/FeedbackController.kt",
    "core/device/src/main/java/com/linnan/blindassist/feedback/FeedbackFatigueController.kt",
    "core/vision/src/main/java/com/linnan/blindassist/vision/TfliteYoloDetector.kt",
    "core/vision/src/main/java/com/linnan/blindassist/vision/ProductionDetectorRoutePolicy.kt",
    "core/vision/src/main/java/com/linnan/blindassist/vision/ObjectDetector.kt",
    "core/vision/src/main/java/com/linnan/blindassist/vision/DetectorBackendPolicy.kt",
    "core/vision/src/main/java/com/linnan/blindassist/vision/ImagePreprocessor.kt",
    "core/vision/src/main/java/com/linnan/blindassist/vision/YoloOutputDecoder.kt",
    "app/src/main/java/com/linnan/blindassist/vision/ProductionQnnRoutingObjectDetectorProvider.kt",
    "app/src/main/assets/yolo11n_fp16_320.tflite",
    "app/src/main/assets/coco_labels.txt",
    "gradle/libs.versions.toml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def git(repo_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=repo_root,
        text=True,
        encoding="utf-8",
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--prestart-receipt", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--truth-membership-receipt", type=Path, required=True)
    parser.add_argument("--app-apk", type=Path, required=True)
    parser.add_argument("--test-apk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if git(repo_root, "status", "--short"):
        raise ValueError("implementation lock requires a clean committed worktree")
    head = git(repo_root, "rev-parse", "HEAD")
    origin_master = git(repo_root, "rev-parse", "origin/master")
    if head != origin_master:
        raise ValueError("implementation lock requires HEAD == origin/master")

    prestart_path = args.prestart_receipt.resolve()
    prestart = json.loads(prestart_path.read_text(encoding="utf-8"))
    if prestart.get("status") != "VALID":
        raise ValueError("device prestart receipt is not VALID")
    if prestart.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("device prestart protocol mismatch")
    if prestart.get("decision_rgb_decoded") is not False:
        raise ValueError("device prestart decoded decision RGB")
    if prestart.get("synthetic_qnn_probe_completed") is not True:
        raise ValueError("device prestart lacks live synthetic QNN probe")
    if prestart.get("synthetic_branch_order_invariance_completed") is not True:
        raise ValueError("device prestart lacks full-chain branch-order mutation proof")
    if prestart.get("backend") != "qualcomm_qnn_htp":
        raise ValueError("device prestart backend is not strict QNN HTP")
    if prestart.get("qnn_maven_version") != "2.47.0":
        raise ValueError("device prestart QNN Maven version mismatch")
    if not isinstance(prestart.get("qnn_runtime_version"), list):
        raise ValueError("device prestart QNN runtime version is missing")
    device = prestart.get("device", {})
    if (device.get("model"), device.get("soc_model")) != EXPECTED_DEVICE:
        raise ValueError("device prestart target identity mismatch")
    prestart_input = prestart.get("input", {})
    if prestart_input.get("frame_count") != EXPECTED_FRAME_COUNT:
        raise ValueError("device prestart frame denominator mismatch")
    if prestart_input.get("rgb_total_bytes") != EXPECTED_RGB_TOTAL_BYTES:
        raise ValueError("device prestart RGB byte denominator mismatch")
    if (
        prestart_input.get("canonical_rgb_inventory_sha256")
        != EXPECTED_INVENTORY_SHA256
    ):
        raise ValueError("device prestart input inventory mismatch")
    input_receipt_path = args.input_receipt.resolve()
    if sha256_file(input_receipt_path) != EXPECTED_INPUT_RECEIPT_SHA256:
        raise ValueError("frozen input receipt identity mismatch")
    input_receipt = json.loads(input_receipt_path.read_text(encoding="utf-8"))
    if (
        input_receipt.get("schema_version")
        != "blindassist.dual_loop_input_preflight.v1"
        or input_receipt.get("protocol_id") != PROTOCOL_ID
        or input_receipt.get("status") != "VALID"
        or input_receipt.get("outcome_blind") is not True
        or input_receipt.get("truth_opened") is not False
        or input_receipt.get("canonical_rgb_inventory_sha256")
        != EXPECTED_INVENTORY_SHA256
    ):
        raise ValueError("frozen input receipt contract mismatch")
    truth_membership_path = args.truth_membership_receipt.resolve()
    if sha256_file(truth_membership_path) != EXPECTED_TRUTH_MEMBERSHIP_SHA256:
        raise ValueError("frozen truth-membership receipt identity mismatch")
    truth_membership = json.loads(truth_membership_path.read_text(encoding="utf-8"))
    if (
        truth_membership.get("schema_version")
        != "blindassist.dual_loop_truth_membership_preflight.v1"
        or truth_membership.get("protocol_id") != PROTOCOL_ID
        or truth_membership.get("status") != "VALID"
        or truth_membership.get("fixed_scored_item_denominator") != 15
        or truth_membership.get("candidate_output_opened") is not False
    ):
        raise ValueError("frozen truth-membership receipt contract mismatch")

    sources = {
        relative: sha256_file(repo_root / relative)
        for relative in LOCKED_FILES
    }
    app_apk = args.app_apk.resolve()
    test_apk = args.test_apk.resolve()
    app_apk_sha256 = sha256_file(app_apk)
    test_apk_sha256 = sha256_file(test_apk)
    if prestart.get("installed_app_apk_sha256") != app_apk_sha256:
        raise ValueError("prestart production app APK differs from lock candidate")
    if prestart.get("installed_test_apk_sha256") != test_apk_sha256:
        raise ValueError("prestart instrumentation APK differs from lock candidate")
    lock = {
        "schema_version": "blindassist.production_temporal_ab_implementation_lock.v1",
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": "LOCKED",
        "git_commit": head,
        "origin_master": origin_master,
        "repo_root": str(repo_root),
        "source_sha256": sources,
        "app_apk": {
            "path": str(app_apk),
            "sha256": app_apk_sha256,
        },
        "test_apk": {
            "path": str(test_apk),
            "sha256": test_apk_sha256,
        },
        "device_prestart": {
            "path": str(prestart_path),
            "sha256": sha256_file(prestart_path),
            "device": prestart["device"],
            "qnn_runtime_version": prestart["qnn_runtime_version"],
            "input": prestart["input"],
        },
        "input_receipt": {
            "path": str(input_receipt_path),
            "sha256": EXPECTED_INPUT_RECEIPT_SHA256,
        },
        "truth_membership_receipt": {
            "path": str(truth_membership_path),
            "sha256": EXPECTED_TRUTH_MEMBERSHIP_SHA256,
        },
        "formal_execution_authorized": False,
        "truth_join_authorized": False,
    }
    atomic_json(args.output.resolve(), lock)
    print(json.dumps({"status": "LOCKED", "git_commit": head}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
