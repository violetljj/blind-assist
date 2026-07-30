#!/usr/bin/env python3
"""Fail-closed activation gate for the one-shot device producer."""

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(repo_root: Path, command: list[str]) -> str:
    return subprocess.check_output(
        command,
        cwd=repo_root,
        text=True,
        encoding="utf-8",
        stderr=subprocess.STDOUT,
    ).strip()


def is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def installed_base_apk_sha256(repo_root: Path, device_serial: str, package_name: str) -> str:
    package_output = run(
        repo_root,
        ["adb", "-s", device_serial, "shell", "pm", "path", package_name],
    )
    package_lines = [
        line.removeprefix("package:").strip()
        for line in package_output.splitlines()
        if line.startswith("package:")
    ]
    if len(package_lines) != 1 or not package_lines[0].endswith("/base.apk"):
        raise ValueError(f"unexpected installed APK path for {package_name}")
    digest_output = run(
        repo_root,
        ["adb", "-s", device_serial, "shell", "sha256sum", package_lines[0]],
    )
    digest = digest_output.split()[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"invalid installed APK SHA-256 for {package_name}")
    return digest


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--implementation-review", type=Path, required=True)
    parser.add_argument("--device-serial", default="R5CX10M8Y8X")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    lock_path = args.implementation_lock.resolve()
    review_path = args.implementation_review.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED":
        raise ValueError("implementation lock is not LOCKED")
    if lock.get("protocol_id") != PROTOCOL_ID or lock.get("implementation_id") != IMPLEMENTATION_ID:
        raise ValueError("implementation lock identity mismatch")
    for relative, expected_hash in lock["source_sha256"].items():
        if sha256_file(repo_root / relative) != expected_hash:
            raise ValueError(f"source identity drift: {relative}")
    for apk_key in ("app_apk", "test_apk"):
        apk = Path(lock[apk_key]["path"])
        if sha256_file(apk) != lock[apk_key]["sha256"]:
            raise ValueError(f"{apk_key} identity drift")
    prestart = Path(lock["device_prestart"]["path"])
    if sha256_file(prestart) != lock["device_prestart"]["sha256"]:
        raise ValueError("device prestart receipt drift")

    review_lines = {
        line.strip()
        for line in review_path.read_text(encoding="utf-8").splitlines()
    }
    required_review_lines = {
        "STATUS: PASS",
        "IMPLEMENTATION_LOCK_VALID: true",
        "FORMAL_EXECUTION_AUTHORIZED: true",
        f"PROTOCOL_ID: {PROTOCOL_ID}",
        f"IMPLEMENTATION_LOCK_SHA256: {sha256_file(lock_path)}",
        f"IMPLEMENTATION_GIT_COMMIT: {lock['git_commit']}",
    }
    if not required_review_lines.issubset(review_lines):
        raise ValueError("implementation review does not authorize formal execution")
    if run(repo_root, ["git", "status", "--short"]):
        raise ValueError("activation requires a clean worktree")
    head = run(repo_root, ["git", "rev-parse", "HEAD"])
    if head != run(repo_root, ["git", "rev-parse", "origin/master"]):
        raise ValueError("activation requires HEAD == origin/master")
    if not is_ancestor(repo_root, lock["git_commit"], head):
        raise ValueError("implementation lock commit is not an ancestor of activation HEAD")

    evidence_root = (
        repo_root
        / "artifacts.local"
        / "evidence"
        / "dual-loop"
        / "production-temporal-geometry-factorial-ab-r0"
    )
    for directory in ("device-producer", "sealed-producer", "evaluation"):
        if (evidence_root / directory).exists():
            raise ValueError(f"candidate output namespace already exists: {directory}")

    state = run(repo_root, ["adb", "-s", args.device_serial, "get-state"])
    if state != "device":
        raise ValueError("formal device is unavailable")
    model = run(repo_root, ["adb", "-s", args.device_serial, "shell", "getprop", "ro.product.model"])
    soc = run(repo_root, ["adb", "-s", args.device_serial, "shell", "getprop", "ro.soc.model"])
    if model != "SM-S9280" or soc != "SM8650":
        raise ValueError(f"formal device identity mismatch: {model}/{soc}")
    installed_app_sha256 = installed_base_apk_sha256(
        repo_root,
        args.device_serial,
        "com.linnan.blindassist",
    )
    installed_test_sha256 = installed_base_apk_sha256(
        repo_root,
        args.device_serial,
        "com.linnan.blindassist.benchmark",
    )
    if installed_app_sha256 != lock["app_apk"]["sha256"]:
        raise ValueError("installed production app APK identity drift")
    if installed_test_sha256 != lock["test_apk"]["sha256"]:
        raise ValueError("installed instrumentation APK identity drift")
    remote_base = (
        "/sdcard/Android/data/com.linnan.blindassist/files/"
        "dual_loop_production_temporal_ab_r0"
    )
    remote_state = run(
        repo_root,
        [
            "adb",
            "-s",
            args.device_serial,
            "shell",
            f"if [ -e '{remote_base}/formal_start.json' ] || "
            f"[ -e '{remote_base}/output' ] || "
            f"[ -e '{remote_base}/authorization' ]; "
            "then echo EXISTS; else echo ABSENT; fi",
        ],
    )
    if remote_state != "ABSENT":
        raise ValueError("remote formal marker or output already exists")

    receipt = {
        "schema_version": "blindassist.production_temporal_ab_activation.v1",
        "protocol_id": PROTOCOL_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "status": "ACTIVATED",
        "git_commit": head,
        "implementation_lock_path": str(lock_path),
        "implementation_lock_sha256": sha256_file(lock_path),
        "implementation_review_path": str(review_path),
        "implementation_review_sha256": sha256_file(review_path),
        "device_serial": args.device_serial,
        "device_model": model,
        "soc_model": soc,
        "installed_app_apk_sha256": installed_app_sha256,
        "installed_test_apk_sha256": installed_test_sha256,
        "candidate_output_namespace_absent": True,
        "formal_execution_authorized": True,
        "truth_join_authorized_after_validated_seal_only": True,
        "confirmation_authorized": False,
        "production_behavior_change_authorized": False,
    }
    atomic_json(args.output.resolve(), receipt)
    print(json.dumps({"status": "ACTIVATED", "git_commit": head}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
