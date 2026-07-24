#!/usr/bin/env python3
"""Run the outcome-unseen R3 target-private transport canary."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import exploratory_profiles_r2_l1 as r1
from r2_l1x_l2p import build_context


STAGE = "R2-L1E-RECOVERY-B1"
ATTEMPT_NAMESPACE = "r2-l1e-recovery-b1"
TARGET_PACKAGE = "com.linnan.blindassist"
INSTRUMENTATION_COMPONENT = (
    "com.linnan.blindassist.benchmark/androidx.test.runner.AndroidJUnitRunner"
)
TEST_METHOD = (
    "com.linnan.blindassist.benchmark."
    "UstrfR2L1MaterializationRecoveryR3DeviceTest"
    "#verifyTargetPrivateTransportCanary"
)
SAFE_STAGE = re.compile(r"r2-l1e-recovery-b1/canary-[0-9a-f]{16}")
EXPECTED_PARENT_KEYS = {
    "base_prereg",
    "original_r2_terminal",
    "a1_overlay",
    "a1_merged_prereg",
    "a1_terminal",
    "a1_validation",
    "a1_resource_guard",
    "a1_attempt_001_manifest",
    "a1_attempt_001_stdout",
    "a1_attempt_002_manifest",
    "a1_attempt_002_stdout",
    "l2_prereg",
    "l3_template",
    "r1_terminal",
}


class CanaryError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CanaryError(f"expected JSON object: {path}")
    return value


def verify_binding(repo: Path, binding: dict[str, Any], label: str) -> Path:
    if set(binding) != {"path", "sha256"}:
        raise CanaryError(f"{label}_binding_shape_drift")
    path = (repo / binding["path"]).resolve()
    try:
        path.relative_to(repo.resolve())
    except ValueError as error:
        raise CanaryError(f"{label}_binding_escapes_repo") from error
    if not path.is_file() or r1.sha256_file(path) != binding["sha256"]:
        raise CanaryError(f"{label}_binding_mismatch")
    return path


def verify_config(repo: Path, path: Path) -> dict[str, Any]:
    config = load_json(path)
    if set(config) != {
        "schema",
        "stage",
        "status",
        "frozen_on",
        "baseline_commit",
        "output_root",
        "parent_bindings",
        "transport_canary",
        "resource_guards",
        "implementation_bindings",
        "authority",
    }:
        raise CanaryError("r3_config_key_roster_drift")
    if config["schema"] != "blindassist_ustrf_l1e_materialization_recovery_r3":
        raise CanaryError("r3_config_schema_drift")
    if config["stage"] != STAGE or config["status"] != "FROZEN_BEFORE_R3_DEVICE_OUTPUT":
        raise CanaryError("r3_stage_or_status_drift")
    if set(config["parent_bindings"]) != EXPECTED_PARENT_KEYS:
        raise CanaryError("r3_parent_binding_roster_drift")
    resolved = {
        label: verify_binding(repo, binding, label)
        for label, binding in config["parent_bindings"].items()
    }
    if load_json(resolved["original_r2_terminal"])["terminal_state"] != (
        "FAIL_CLOSED_EXECUTION_ABORTED"
    ):
        raise CanaryError("original_r2_terminal_rewritten")
    if load_json(resolved["a1_terminal"])["terminal_state"] != (
        "FAIL_CLOSED_EXECUTION_ABORTED"
    ):
        raise CanaryError("a1_terminal_rewritten")
    if config["transport_canary"] != {
        "source": "first_missing_crowdbot_ledger",
        "initial_attempts": 1,
        "bounded_retries": 1,
        "candidate_execution_forbidden": True,
        "tflite_load_forbidden": True,
        "verify_every_manifest_image_sha256": True,
        "ingress": "adb_push_data_local_tmp_then_run_as_target_internal_files",
        "target_package": TARGET_PACKAGE,
        "attempt_namespace": ATTEMPT_NAMESPACE,
    }:
        raise CanaryError("transport_canary_contract_drift")
    guards = config["resource_guards"]
    if guards != {
        "minimum_system_available_physical_memory_bytes": 6 * 1024**3,
        "readiness_sample_count": 6,
        "readiness_sample_interval_seconds": 5,
        "minimum_device_free_bytes": 5 * 1024**3,
        "old_attempts_count_toward_r3": False,
    }:
        raise CanaryError("r3_resource_guard_contract_drift")
    expected_authority = {
        "candidate_execution": False,
        "candidate_trace": False,
        "candidate_profile": False,
        "selection": False,
        "ranking": False,
        "recommendation": False,
        "provisional_selection": False,
        "l2_execution": False,
        "l3_execution": False,
        "android_shadow": False,
        "h2": False,
        "human_outcome": False,
        "independent_walking_safety": False,
        "production": False,
    }
    if config["authority"] != expected_authority:
        raise CanaryError("r3_canary_authority_drift")
    for label, binding in config["implementation_bindings"].items():
        verify_binding(repo, binding, f"implementation_{label}")
    return config


def run(
    command: Sequence[str],
    *,
    timeout: int,
    check: bool = True,
    binary: bool = False,
) -> subprocess.CompletedProcess[Any]:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=not binary,
        encoding=None if binary else "utf-8",
        errors=None if binary else "replace",
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        stderr = (
            completed.stderr.decode("utf-8", "replace")
            if isinstance(completed.stderr, bytes)
            else completed.stderr
        )
        raise CanaryError(
            f"command_failed:{completed.returncode}:{' '.join(command[:5])}:{stderr.strip()}"
        )
    return completed


def adb_command(adb: Path, serial: str, *args: str) -> list[str]:
    return [str(adb), "-s", serial, *args]


def select_serial(adb: Path) -> str:
    listed = run([str(adb), "devices", "-l"], timeout=30).stdout
    devices = [
        line.split()[0]
        for line in listed.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]
    requested = os.environ.get("ANDROID_SERIAL")
    if requested:
        if requested not in devices:
            raise CanaryError("requested_android_serial_not_online")
        return requested
    if len(devices) != 1:
        raise CanaryError(f"expected_one_android_device:found={len(devices)}")
    return devices[0]


def readiness_samples(config: dict[str, Any]) -> list[int]:
    guards = config["resource_guards"]
    samples = []
    for index in range(int(guards["readiness_sample_count"])):
        samples.append(r1.available_memory_bytes())
        if index + 1 < int(guards["readiness_sample_count"]):
            time.sleep(int(guards["readiness_sample_interval_seconds"]))
    required = int(guards["minimum_system_available_physical_memory_bytes"])
    if any(value < required for value in samples):
        raise CanaryError(
            "readiness_memory_guard:"
            + ",".join(map(str, samples))
            + f":required={required}"
        )
    return samples


def require_memory(config: dict[str, Any], checkpoint: str) -> int:
    available = r1.available_memory_bytes()
    required = int(
        config["resource_guards"]["minimum_system_available_physical_memory_bytes"]
    )
    if available < required:
        raise CanaryError(
            f"{checkpoint}_memory_guard:available={available}:required={required}"
        )
    return available


def device_free_bytes(adb: Path, serial: str) -> int:
    output = run(
        adb_command(adb, serial, "shell", "df", "-k", "/data"),
        timeout=30,
    ).stdout
    rows = [line.split() for line in output.splitlines() if line.strip()]
    if len(rows) < 2:
        raise CanaryError("device_free_space_unavailable")
    for row in reversed(rows[1:]):
        if len(row) >= 4 and row[-1].startswith("/data"):
            try:
                return int(row[-3]) * 1024
            except ValueError as error:
                raise CanaryError("device_free_space_unparseable") from error
    raise CanaryError("device_data_mount_missing")


def first_missing_crowdbot(
    repo: Path, base_prereg_path: Path
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    Path,
    list[dict[str, Any]],
]:
    _, context = build_context(repo, base_prereg_path)
    for descriptor, rows in context["groups"]:
        if descriptor["source_id"].startswith("crowdbot_"):
            bundle, image_rows = r1.load_crowdbot_images(
                repo, context["base_config"], descriptor, rows
            )
            return descriptor, rows, context, bundle, image_rows
    raise CanaryError("no_crowdbot_ledger")


def derived_manifest(
    repo: Path,
    config: dict[str, Any],
    descriptor: dict[str, Any],
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    bundle_path: Path,
    image_rows: list[dict[str, Any]],
    path: Path,
    memory_samples: list[int],
) -> dict[str, Any]:
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
        "readiness_memory_samples": memory_samples,
        "frames": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in image_rows
        ],
        "authority": {
            "transport_canary_input_only": True,
            **config["authority"],
        },
    }
    r1.atomic_write_json(path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    config = verify_config(repo, config_path)
    output_root = (repo / config["output_root"]).resolve()
    local_artifacts = (repo / "artifacts.local").resolve()
    try:
        output_root.relative_to(local_artifacts)
    except ValueError as error:
        raise CanaryError("r3_output_root_escapes_artifacts_local") from error
    output_root.mkdir(parents=True, exist_ok=True)
    memory_samples = readiness_samples(config)
    base_prereg_path = repo / config["parent_bindings"]["base_prereg"]["path"]
    descriptor, rows, context, bundle_path, image_rows = first_missing_crowdbot(
        repo, base_prereg_path
    )
    post_input_memory = require_memory(config, "post_input_validation")
    canary_id = f"canary-{secrets.token_hex(8)}"
    remote_relative = f"{ATTEMPT_NAMESPACE}/{canary_id}"
    if not SAFE_STAGE.fullmatch(remote_relative):
        raise CanaryError("unsafe_canary_relative_path")
    attempt_dir = output_root / "transport-canary" / canary_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = attempt_dir / "manifest.json"
    derived_manifest(
        repo,
        config,
        descriptor,
        rows,
        context,
        bundle_path,
        image_rows,
        manifest_path,
        memory_samples,
    )
    manifest_sha = r1.sha256_file(manifest_path)
    image_directory = Path(image_rows[0]["_host_image_path"]).parent.resolve()
    if any(Path(row["_host_image_path"]).parent.resolve() != image_directory for row in image_rows):
        raise CanaryError("canary_images_span_multiple_directories")
    adb = r1.locate_adb(repo)
    serial = select_serial(adb)
    free_bytes = device_free_bytes(adb, serial)
    required_device_free = int(
        config["resource_guards"]["minimum_device_free_bytes"]
    )
    if free_bytes < required_device_free:
        raise CanaryError(
            f"device_free_space_guard:available={free_bytes}:"
            f"required={required_device_free}"
        )
    shell_root = f"/data/local/tmp/blindassist-{canary_id}"
    app_root = f"files/{remote_relative}"
    receipt_relative = f"{remote_relative}/canary-receipt.json"
    instrumentation_path = attempt_dir / "instrumentation.stdout.txt"
    receipt_path = attempt_dir / "canary-receipt.json"
    shell_stat = ""
    app_stat = ""
    try:
        run(adb_command(adb, serial, "shell", "mkdir", "-p", f"{shell_root}/images"), timeout=60)
        run(adb_command(adb, serial, "push", str(manifest_path), f"{shell_root}/manifest.json"), timeout=300)
        run(
            adb_command(adb, serial, "push", f"{image_directory}{os.sep}.", f"{shell_root}/images/"),
            timeout=1800,
        )
        run(
            adb_command(
                adb,
                serial,
                "shell",
                "run-as",
                TARGET_PACKAGE,
                "mkdir",
                "-p",
                f"{app_root}/images",
            ),
            timeout=60,
        )
        run(
            adb_command(
                adb,
                serial,
                "shell",
                "run-as",
                TARGET_PACKAGE,
                "cp",
                f"{shell_root}/manifest.json",
                f"{app_root}/manifest.json",
            ),
            timeout=60,
        )
        run(
            adb_command(
                adb,
                serial,
                "shell",
                "run-as",
                TARGET_PACKAGE,
                "cp",
                "-R",
                f"{shell_root}/images/.",
                f"{app_root}/images/",
            ),
            timeout=1800,
        )
        shell_stat = run(
            adb_command(adb, serial, "shell", "ls", "-lZ", f"{shell_root}/manifest.json"),
            timeout=30,
        ).stdout.strip()
        app_stat = run(
            adb_command(
                adb,
                serial,
                "shell",
                "run-as",
                TARGET_PACKAGE,
                "ls",
                "-l",
                f"{app_root}/manifest.json",
            ),
            timeout=30,
        ).stdout.strip()
        pre_instrument_memory = require_memory(config, "pre_instrument")
        instrument = adb_command(
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
            "ustrfR2L1eRecoveryR3CanaryRequired",
            "true",
            "-e",
            "ustrfR2L1eRecoveryR3CanaryInput",
            f"{remote_relative}/manifest.json",
            "-e",
            "ustrfR2L1eRecoveryR3ExpectedManifestSha256",
            manifest_sha,
            "-e",
            "ustrfR2L1eRecoveryR3CanaryOutput",
            receipt_relative,
            INSTRUMENTATION_COMPONENT,
        )
        completed = run(instrument, timeout=1800, check=False)
        transcript = completed.stdout + "\n" + completed.stderr
        instrumentation_path.write_text(transcript, encoding="utf-8")
        if (
            completed.returncode != 0
            or "FAILURES!!!" in transcript
            or "OK (1 test)" not in transcript
        ):
            raise CanaryError("transport_canary_instrumentation_failed")
        pulled = run(
            adb_command(
                adb,
                serial,
                "exec-out",
                "run-as",
                TARGET_PACKAGE,
                "cat",
                f"files/{receipt_relative}",
            ),
            timeout=120,
            binary=True,
        )
        receipt_path.write_bytes(bytes(pulled.stdout))
        receipt = load_json(receipt_path)
        expected = {
            "stage": STAGE,
            "attempt_namespace": ATTEMPT_NAMESPACE,
            "status": "TARGET_PRIVATE_TRANSPORT_CANARY_PASS",
            "error": None,
            "target_package": TARGET_PACKAGE,
            "resolved_storage": "target_context_internal_files",
            "manifest_sha256": manifest_sha,
            "source_id": descriptor["source_id"],
            "sequence_id": descriptor["sequence_id"],
            "verified_image_count": len(rows),
            "candidate_execution": {
                "tflite_loaded": False,
                "c1_c2_c3_loaded": False,
                "candidate_output_count": 0,
            },
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise CanaryError(f"transport_canary_receipt_drift:{key}")
        host_receipt = {
            "schema": "blindassist_ustrf_l1e_materialization_recovery_r3_host_canary_receipt",
            "stage": STAGE,
            "status": "TARGET_PRIVATE_TRANSPORT_CANARY_PASS",
            "config_sha256": r1.sha256_file(config_path),
            "manifest_sha256": manifest_sha,
            "device_receipt_sha256": r1.sha256_file(receipt_path),
            "instrumentation_stdout_sha256": r1.sha256_file(instrumentation_path),
            "serial": serial,
            "shell_manifest_stat": shell_stat,
            "app_manifest_stat": app_stat,
            "readiness_memory_samples": memory_samples,
            "post_input_validation_memory_bytes": post_input_memory,
            "pre_instrument_memory_bytes": pre_instrument_memory,
            "device_free_bytes_before_staging": free_bytes,
            "verified_image_count": len(rows),
            "candidate_output_count": 0,
            "authority": config["authority"],
        }
        host_receipt_path = attempt_dir / "host-canary-receipt.json"
        r1.atomic_write_json(host_receipt_path, host_receipt)
        print(json.dumps(host_receipt, sort_keys=True))
        return 0
    finally:
        run(
            adb_command(adb, serial, "shell", "rm", "-rf", shell_root),
            timeout=120,
            check=False,
        )
        run(
            adb_command(
                adb,
                serial,
                "shell",
                "run-as",
                TARGET_PACKAGE,
                "rm",
                "-rf",
                app_root,
            ),
            timeout=120,
            check=False,
        )


if __name__ == "__main__":
    raise SystemExit(main())
