#!/usr/bin/env python3
"""Materialize exactly one CrowdBot detector ledger through target-private storage."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import exploratory_profiles_r2_l1 as r1
import run_l1e_materialization_recovery_r3_canary as recovery


STAGE = recovery.STAGE
ATTEMPT_NAMESPACE = recovery.ATTEMPT_NAMESPACE
TARGET_PACKAGE = recovery.TARGET_PACKAGE
TEST_METHOD = (
    "com.linnan.blindassist.benchmark."
    "UstrfR2L1MaterializationRecoveryR3ExporterDeviceTest"
    "#exportOneFrozenMaskedLedgerRecoveryR3"
)


class MaterializationError(RuntimeError):
    pass


def verify_overlay(repo: Path, path: Path) -> dict[str, Any]:
    overlay = recovery.load_json(path)
    if set(overlay) != {
        "schema",
        "stage",
        "status",
        "frozen_on",
        "canary_config",
        "canary_receipt",
        "execution",
        "implementation_bindings",
        "authority",
    }:
        raise MaterializationError("materialization_overlay_key_roster_drift")
    if (
        overlay["schema"]
        != "blindassist_ustrf_l1e_materialization_recovery_r3_one_shard"
        or overlay["stage"] != STAGE
        or overlay["status"] != "FROZEN_BEFORE_FIRST_R3_CANDIDATE_OUTPUT"
    ):
        raise MaterializationError("materialization_overlay_identity_drift")
    canary_config_path = recovery.verify_binding(
        repo, overlay["canary_config"], "canary_config"
    )
    recovery.verify_config(repo, canary_config_path)
    canary_receipt_path = recovery.verify_binding(
        repo, overlay["canary_receipt"], "canary_receipt"
    )
    canary_receipt = recovery.load_json(canary_receipt_path)
    if (
        canary_receipt.get("status") != "TARGET_PRIVATE_TRANSPORT_CANARY_PASS"
        or canary_receipt.get("candidate_output_count") != 0
    ):
        raise MaterializationError("canary_receipt_not_admissible")
    if overlay["execution"] != {
        "maximum_crowdbot_shards": 1,
        "initial_attempts_per_ledger": 1,
        "bounded_retries_per_ledger": 2,
        "one_host_process_per_shard": True,
        "raw_pull_transport": "adb_exec_out_run_as_target_stream_to_file",
        "delete_raw_after_validated_successor": True,
        "minimum_system_available_physical_memory_bytes": 6 * 1024**3,
        "minimum_device_free_bytes": 5 * 1024**3,
    }:
        raise MaterializationError("materialization_execution_contract_drift")
    for label, binding in overlay["implementation_bindings"].items():
        recovery.verify_binding(repo, binding, f"materialization_{label}")
    if overlay["authority"] != recovery.verify_config(repo, canary_config_path)[
        "authority"
    ]:
        raise MaterializationError("materialization_authority_drift")
    return overlay


def stream_app_file(
    adb: Path,
    serial: str,
    remote_relative: str,
    destination: Path,
    *,
    timeout: int,
) -> None:
    command = recovery.adb_command(
        adb,
        serial,
        "exec-out",
        "run-as",
        TARGET_PACKAGE,
        "cat",
        f"files/{remote_relative}",
    )
    with destination.open("xb") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.PIPE,
        )
        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise MaterializationError("stream_app_file_timeout") from error
    if process.returncode != 0:
        destination.unlink(missing_ok=True)
        raise MaterializationError(
            "stream_app_file_failed:"
            + stderr.decode("utf-8", "replace").strip()
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    overlay_path = args.config.resolve()
    overlay = verify_overlay(repo, overlay_path)
    canary_config_path = repo / overlay["canary_config"]["path"]
    canary_config = recovery.verify_config(repo, canary_config_path)
    output_root = (repo / canary_config["output_root"]).resolve()
    output_root.relative_to((repo / "artifacts.local").resolve())
    memory_samples = recovery.readiness_samples(canary_config)
    base_prereg_path = repo / canary_config["parent_bindings"]["base_prereg"]["path"]
    descriptor, rows, context, bundle_path, image_rows = (
        recovery.first_missing_crowdbot(repo, base_prereg_path)
    )
    post_input_memory = recovery.require_memory(
        canary_config, "materialization_post_input_validation"
    )
    ledger_path, successor_path = r1.compact_paths(
        output_root, descriptor["source_id"], descriptor["sequence_id"]
    )
    if r1.validate_compact_ledger(ledger_path, successor_path, descriptor, rows):
        raise MaterializationError("first_crowdbot_ledger_already_materialized")

    slug = r1.stable_slug(descriptor["source_id"], descriptor["sequence_id"])
    attempt_root = output_root / "materialization" / slug
    existing_attempts = (
        sorted(attempt_root.glob("attempt-*")) if attempt_root.exists() else []
    )
    maximum_attempts = int(
        overlay["execution"]["initial_attempts_per_ledger"]
    ) + int(overlay["execution"]["bounded_retries_per_ledger"])
    if len(existing_attempts) >= maximum_attempts:
        raise MaterializationError("r3_materialization_retry_limit_exhausted")
    attempt_number = len(existing_attempts) + 1
    attempt_dir = attempt_root / f"attempt-{attempt_number:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = attempt_dir / "device-manifest.json"
    raw_path = attempt_dir / "canonical-raw.gz"
    receipt_path = attempt_dir / "device-raw-receipt.json"
    stdout_path = attempt_dir / "instrumentation.stdout.txt"
    manifest = {
        "schema": r1.DEVICE_MANIFEST_SCHEMA,
        "stage": STAGE,
        "parent_compatible_stage": "R2-L1E",
        "attempt_namespace": ATTEMPT_NAMESPACE,
        "source_id": descriptor["source_id"],
        "sequence_id": descriptor["sequence_id"],
        "frame_mask_sha256": descriptor["frame_mask_sha256"],
        "frame_count": len(rows),
        "input_shape": context["base_config"]["input_contract"]["detector"][
            "input_shape"
        ],
        "output_shape": context["base_config"]["input_contract"]["detector"][
            "output_shape"
        ],
        "model_sha256": context["base_config"]["input_contract"]["detector"][
            "model_sha256"
        ],
        "labels_sha256": context["base_config"]["input_contract"]["detector"][
            "labels_sha256"
        ],
        "person_class_index": 0,
        "confidence_threshold": 0.35,
        "nms_iou_threshold": 0.45,
        "bundle_sha256": r1.sha256_file(bundle_path),
        "projected_raw_bytes": len(rows) * r1.RAW_BYTES_PER_FRAME,
        "memory_guard": {
            "readiness_samples": memory_samples,
            "post_input_validation_available_bytes": post_input_memory,
            "required_available_bytes": overlay["execution"][
                "minimum_system_available_physical_memory_bytes"
            ],
        },
        "frames": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in image_rows
        ],
        "authority": {
            "candidate_input_only": True,
            **overlay["authority"],
        },
    }
    r1.atomic_write_json(manifest_path, manifest)

    image_directory = Path(image_rows[0]["_host_image_path"]).parent.resolve()
    if any(
        Path(row["_host_image_path"]).parent.resolve() != image_directory
        for row in image_rows
    ):
        raise MaterializationError("materialization_images_span_multiple_directories")
    adb = r1.locate_adb(repo)
    serial = recovery.select_serial(adb)
    device_free = recovery.device_free_bytes(adb, serial)
    if device_free < int(overlay["execution"]["minimum_device_free_bytes"]):
        raise MaterializationError("materialization_device_free_space_guard")
    remote_relative = (
        f"{ATTEMPT_NAMESPACE}/{slug}/attempt-{attempt_number:03d}"
    )
    shell_root = f"/data/local/tmp/blindassist-r3-{slug}-attempt-{attempt_number:03d}"
    app_root = f"files/{remote_relative}"
    started = time.monotonic()
    try:
        recovery.run(
            recovery.adb_command(
                adb, serial, "shell", "mkdir", "-p", f"{shell_root}/images"
            ),
            timeout=60,
        )
        recovery.run(
            recovery.adb_command(
                adb, serial, "push", str(manifest_path), f"{shell_root}/manifest.json"
            ),
            timeout=300,
        )
        recovery.run(
            recovery.adb_command(
                adb,
                serial,
                "push",
                f"{image_directory}{os.sep}.",
                f"{shell_root}/images/",
            ),
            timeout=1800,
        )
        recovery.run(
            recovery.adb_command(
                adb, serial, "shell", "run-as", TARGET_PACKAGE,
                "mkdir", "-p", f"{app_root}/images"
            ),
            timeout=60,
        )
        recovery.run(
            recovery.adb_command(
                adb, serial, "shell", "run-as", TARGET_PACKAGE,
                "cp", f"{shell_root}/manifest.json", f"{app_root}/manifest.json"
            ),
            timeout=60,
        )
        recovery.run(
            recovery.adb_command(
                adb, serial, "shell", "run-as", TARGET_PACKAGE,
                "cp", "-R", f"{shell_root}/images/.", f"{app_root}/images/"
            ),
            timeout=1800,
        )
        pre_instrument_memory = recovery.require_memory(
            canary_config, "materialization_pre_instrument"
        )
        instrument = recovery.adb_command(
            adb,
            serial,
            "shell",
            "am",
            "instrument",
            "-w",
            "-r",
            "-e",
            "class",
            TEST_METHOD,
            "-e",
            "ustrfR2L1eRecoveryR3ExportRequired",
            "true",
            "-e",
            "ustrfR2L1eRecoveryR3ExportInput",
            f"{remote_relative}/manifest.json",
            "-e",
            "ustrfR2L1eRecoveryR3RawOutput",
            f"{remote_relative}/canonical-raw.gz",
            "-e",
            "ustrfR2L1eRecoveryR3ReceiptOutput",
            f"{remote_relative}/device-raw-receipt.json",
            recovery.INSTRUMENTATION_COMPONENT,
        )
        completed = recovery.run(instrument, timeout=3600, check=False)
        transcript = completed.stdout + "\n" + completed.stderr
        stdout_path.write_text(transcript, encoding="utf-8")
        if (
            completed.returncode != 0
            or "FAILURES!!!" in transcript
            or "OK (1 test)" not in transcript
        ):
            raise MaterializationError("r3_materialization_instrumentation_failed")
        pulled_receipt = recovery.run(
            recovery.adb_command(
                adb, serial, "exec-out", "run-as", TARGET_PACKAGE,
                "cat", f"{app_root}/device-raw-receipt.json"
            ),
            timeout=120,
            binary=True,
        )
        receipt_path.write_bytes(bytes(pulled_receipt.stdout))
        stream_app_file(
            adb,
            serial,
            f"{remote_relative}/canonical-raw.gz",
            raw_path,
            timeout=1800,
        )
        receipt = r1.load_json(receipt_path)
        if receipt.get("stage") != STAGE:
            raise MaterializationError("unexpected_r3_device_receipt_stage")
        r1.validate_device_receipt(
            receipt, manifest_path, raw_path, descriptor, rows
        )
        labels = r1.detector_labels(context["base_config"], repo)
        frames = []
        raw_digest = hashlib.sha256()
        with gzip.open(raw_path, "rb") as stream:
            for expected, device_row in zip(
                rows, receipt["frames"], strict=True
            ):
                raw_bytes = r1.read_exact(
                    stream,
                    r1.RAW_BYTES_PER_FRAME,
                    f"{descriptor['sequence_id']}/{expected['frame_id']}",
                )
                raw_digest.update(raw_bytes)
                raw_sha = hashlib.sha256(raw_bytes).hexdigest()
                if raw_sha != device_row["android_raw_output_sha256"]:
                    raise MaterializationError("r3_raw_record_sha256_mismatch")
                decode_started = time.perf_counter_ns()
                detections = r1.decode_raw_record(
                    raw_bytes, [640, 480], labels, context["base_config"]
                )
                frames.append(
                    {
                        **expected,
                        "source_size": [640, 480],
                        "android_raw_output_sha256": raw_sha,
                        "detector_processing_latency_ns": int(
                            device_row["detector_processing_latency_ns"]
                        ),
                        "host_decode_latency_ns": (
                            time.perf_counter_ns() - decode_started
                        ),
                        "person_detections": detections,
                    }
                )
                r1.enforce_host_rss_guard(context["base_config"])
            if stream.read(1):
                raise MaterializationError("r3_raw_stream_has_trailing_records")
        if (
            raw_digest.hexdigest()
            != receipt["canonical_raw_stream"]["uncompressed_sha256"]
        ):
            raise MaterializationError("r3_raw_uncompressed_sha256_mismatch")
        ledger = {
            "schema": r1.COMPACT_SCHEMA,
            "stage": "R2-L1E",
            "recovery_stage": STAGE,
            "attempt_namespace": ATTEMPT_NAMESPACE,
            "authority": (
                "candidate_input_only_no_selection_android_h2_human_or_production_authority"
            ),
            "source_id": descriptor["source_id"],
            "sequence_id": descriptor["sequence_id"],
            "frame_mask_sha256": descriptor["frame_mask_sha256"],
            "frame_count": len(frames),
            "canonical_raw_source": "R3_target_private_android_canvas_stream",
            "canonical_raw_sha256": r1.sha256_file(raw_path),
            "device_receipt_sha256": r1.sha256_file(receipt_path),
            "device_manifest_sha256": r1.sha256_file(manifest_path),
            "frames": frames,
        }
        r1.atomic_write_json(ledger_path, ledger)
        successor = {
            "schema": r1.SUCCESSOR_SCHEMA,
            "stage": "R2-L1E",
            "recovery_stage": STAGE,
            "attempt_namespace": ATTEMPT_NAMESPACE,
            "source_id": descriptor["source_id"],
            "sequence_id": descriptor["sequence_id"],
            "frame_mask_sha256": descriptor["frame_mask_sha256"],
            "raw_sha256": r1.sha256_file(raw_path),
            "raw_retained_as_parent_evidence": False,
            "device_receipt_sha256": r1.sha256_file(receipt_path),
            "device_manifest_sha256": r1.sha256_file(manifest_path),
            "compact_ledger_sha256": r1.sha256_file(ledger_path),
            "frame_count": len(frames),
            "wall_time_seconds": time.monotonic() - started,
            "projected_raw_bytes": len(rows) * r1.RAW_BYTES_PER_FRAME,
            "validated": True,
        }
        r1.atomic_write_json(successor_path, successor)
        if not r1.validate_compact_ledger(
            ledger_path, successor_path, descriptor, rows
        ):
            raise MaterializationError("r3_compact_successor_validation_failed")
        raw_sha = r1.sha256_file(raw_path)
        raw_path.unlink()
        host_receipt = {
            "schema": (
                "blindassist_ustrf_l1e_materialization_recovery_r3_"
                "one_shard_receipt"
            ),
            "stage": STAGE,
            "status": "FIRST_CROWDBOT_SHARD_MATERIALIZED",
            "config_sha256": r1.sha256_file(overlay_path),
            "source_id": descriptor["source_id"],
            "sequence_id": descriptor["sequence_id"],
            "frame_count": len(frames),
            "raw_sha256": raw_sha,
            "raw_retained": False,
            "compact_ledger_sha256": r1.sha256_file(ledger_path),
            "successor_sha256": r1.sha256_file(successor_path),
            "readiness_memory_samples": memory_samples,
            "post_input_validation_memory_bytes": post_input_memory,
            "pre_instrument_memory_bytes": pre_instrument_memory,
            "device_free_bytes_before_staging": device_free,
            "authority": overlay["authority"],
        }
        host_receipt_path = attempt_dir / "host-materialization-receipt.json"
        r1.atomic_write_json(host_receipt_path, host_receipt)
        print(json.dumps(host_receipt, sort_keys=True))
        return 0
    finally:
        recovery.run(
            recovery.adb_command(
                adb, serial, "shell", "rm", "-rf", shell_root
            ),
            timeout=120,
            check=False,
        )
        recovery.run(
            recovery.adb_command(
                adb, serial, "shell", "run-as", TARGET_PACKAGE,
                "rm", "-rf", app_root
            ),
            timeout=120,
            check=False,
        )


if __name__ == "__main__":
    raise SystemExit(main())
