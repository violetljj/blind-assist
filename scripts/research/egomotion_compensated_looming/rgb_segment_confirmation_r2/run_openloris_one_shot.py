from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import urlsplit
import urllib.request
import zlib

from activation_preflight import (
    ActivationPreflightFailure,
    exclusive_claim,
    process_peak_memory,
    secret_scan,
    sha256_file,
    validate_resource_probe,
)
from diagnostic_transport import (
    AppendOnlyLedger,
    DiagnosticRemoteRange,
    EvidencePersistenceFailure,
    write_json_atomic,
)


EXPECTED_CONFIG_KEYS = {
    "schema_version",
    "protocol_id",
    "source",
    "solid_folder",
    "window",
    "transport",
    "phase_order",
    "bindings",
    "claim_namespace",
    "authority",
    "status",
    "execution_authority",
}
EXPECTED_PHASE_ORDER = [
    "ACTIVATION_VALIDATED",
    "CLAIM_CREATED",
    "ARCHIVE_HEADER_AND_DIRECTORY",
    "SOLID_FOLDER_PACK_AND_TARGET_EMISSION",
    "TARGET_IDENTITY_VALIDATION",
    "RUNTIME_SECRET_SCAN",
    "SUCCESS_COMMIT",
]
ACTIVATION_CANDIDATE_PATH = (
    "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/"
    "openloris_activation_candidate.v3.json"
)


class RemoteFileAdapter(io.RawIOBase):
    def __init__(self, remote: DiagnosticRemoteRange) -> None:
        self.remote = remote

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.remote.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.remote.position + offset
        elif whence == io.SEEK_END:
            target = self.remote.length + offset
        else:
            raise ValueError("UNSUPPORTED_WHENCE")
        return self.remote.seek(target)

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            raise io.UnsupportedOperation("NEGATIVE_SIZE_READ_FORBIDDEN")
        return self.remote.read(size)

    def writable(self) -> bool:
        return False


class IdentityBoundOpener:
    def __init__(
        self,
        *,
        allowed_final_hosts: set[str],
        maximum_attempts_per_identical_range: int,
        opener=urllib.request.urlopen,
    ) -> None:
        self.allowed_final_hosts = allowed_final_hosts
        self.maximum_attempts = maximum_attempts_per_identical_range
        self.opener = opener
        self.attempts: dict[str, int] = {}

    def __call__(self, request, *, timeout: int):
        request.add_header("Accept-Encoding", "identity")
        range_value = request.get_header("Range")
        if not isinstance(range_value, str):
            raise OSError("RANGE_HEADER_MISSING")
        count = self.attempts.get(range_value, 0)
        if count >= self.maximum_attempts:
            raise OSError("CLAIM_IDENTICAL_RANGE_ATTEMPT_LIMIT")
        self.attempts[range_value] = count + 1
        response = self.opener(request, timeout=timeout)
        final = urlsplit(response.geturl())
        content_encoding = response.headers.get("Content-Encoding")
        if (
            final.scheme != "https"
            or final.hostname not in self.allowed_final_hosts
            or content_encoding not in (None, "", "identity")
        ):
            response.close()
            raise OSError("SOURCE_RESPONSE_IDENTITY")
        return response


def require_exact_file(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = repo / binding.get("path", "")
    if not path.is_file() or sha256_file(path) != binding.get("sha256"):
        raise ActivationPreflightFailure(f"BINDING_MISMATCH:{label}")
    return path


def validate_config(repo: Path, config: dict[str, Any]) -> None:
    if set(config) != EXPECTED_CONFIG_KEYS:
        raise ActivationPreflightFailure("CONFIG_TOP_LEVEL_KEYS")
    if (
        config.get("schema_version")
        != "rcle_rgb_segment_confirmation_r2_openloris_runner_config.v1"
        or config.get("protocol_id") != "RCLE_RGB_SEGMENT_CONFIRMATION_R2"
    ):
        raise ActivationPreflightFailure("CONFIG_IDENTITY")
    if config.get("source") != {
        "source_family_id": "OPENLORIS_CORRIDOR",
        "capture_id": "corridor1-1",
        "window_id": "corridor1-1:w004",
        "url": "https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/main/package/corridor1-1.7z",
        "object_length": 13_853_763_765,
        "object_sha256": "c7ff1a472ca54da82198521eda8c18f2065691075a05e706880f7fb58fda8415",
    }:
        raise ActivationPreflightFailure("CONFIG_SOURCE")
    if config.get("solid_folder") != {
        "folder_index": 2,
        "pack_bytes": 3_946_335_545,
        "target_folder_membership": "ALL_302_TARGETS_BOUND_TO_FOLDER_2",
    }:
        raise ActivationPreflightFailure("CONFIG_SOLID_FOLDER")
    if config.get("window") != {
        "half_open_window_s": ["1560000043.537699", "1560000053.537699"],
        "selected_frame_count": 300,
        "guard_frame_count": 2,
        "target_member_count": 302,
    }:
        raise ActivationPreflightFailure("CONFIG_WINDOW")
    if config.get("transport") != {
        "remote_byte_hard_cap": 3_947_000_000,
        "maximum_attempts_per_identical_range": 3,
        "network_chunk_bytes": 8_388_608,
        "timeout_seconds": 90,
        "accept_encoding": "identity",
        "allowed_final_url_hosts": [
            "huggingface.co",
            "cdn-lfs.hf.co",
            "cas-bridge.xethub.hf.co",
        ],
        "final_identity_closure": (
            "ALLOWED_HTTPS_HOST_PLUS_EXACT_OBJECT_TOTAL_PLUS_"
            "EXACT_302_MEMBER_SIZE_CRC32"
        ),
        "maximum_network_attempts_per_identical_range_for_entire_claim": 3,
        "full_source_fallback": False,
        "r1_retry_or_resume": False,
    }:
        raise ActivationPreflightFailure("CONFIG_TRANSPORT")
    if config.get("phase_order") != EXPECTED_PHASE_ORDER:
        raise ActivationPreflightFailure("CONFIG_PHASE_ORDER")
    if config.get("claim_namespace") != (
        "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/"
        "openloris_identity_run_v2"
    ):
        raise ActivationPreflightFailure("CONFIG_CLAIM_NAMESPACE")
    if config.get("authority") != {
        "openloris_identity_extraction": False,
        "rgb_decode": False,
        "rgb_algorithm_execution": False,
        "performance_qualification": False,
        "host_offline_replay": False,
        "android": False,
    }:
        raise ActivationPreflightFailure("CONFIG_AUTHORITY")
    if (
        config.get("status") != "ACTIVATION_REVIEW_REQUIRED"
        or config.get("execution_authority") is not False
    ):
        raise ActivationPreflightFailure("CONFIG_NOT_ACTIVATED")
    bindings = config["bindings"]
    require_exact_file(repo, bindings["r1_preaccess_lock"], "R1_LOCK")
    runtime_path = require_exact_file(
        repo, bindings["r1_runtime_lock"], "R1_RUNTIME"
    )
    r2_runtime_path = require_exact_file(
        repo, bindings["r2_runtime_lock"], "R2_RUNTIME"
    )
    if (
        bindings.get("r2_contract_sha256")
        != "ee7285c021460b25bc3f1c1a668c7e3e4181427da291b3aa599b92fc3a2bb177"
        or bindings.get("r2_design_review_sha256")
        != "191a4bda6d3b6dbc78fcfdfe8a499efe2cb86d3f65defbdc251fd4e852cefe7f"
        or bindings.get("r2_user_authorization_sha256")
        != "e3404747d77f21dfa0068a88bac6a417bd087055500ae6cf34a4d9e27d6a427c"
    ):
        raise ActivationPreflightFailure("CONFIG_R2_BINDINGS")
    validate_runtime(runtime_path)
    validate_parent_runtime_fully(repo, runtime_path)
    validate_r2_runtime(repo, r2_runtime_path)


def validate_runtime(runtime_path: Path) -> None:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    python = runtime["python"]
    executable = Path(sys.executable).resolve()
    if str(executable).casefold() != str(
        Path(python["venv_executable"]).resolve()
    ).casefold():
        raise ActivationPreflightFailure("RUNTIME_EXECUTABLE")
    if sha256_file(executable) != python["venv_executable_sha256"]:
        raise ActivationPreflightFailure("RUNTIME_EXECUTABLE_SHA256")
    if ".".join(map(str, sys.version_info[:3])) != python["version"]:
        raise ActivationPreflightFailure("RUNTIME_PYTHON_VERSION")
    if importlib.metadata.version("py7zr") != "1.0.0":
        raise ActivationPreflightFailure("RUNTIME_PY7ZR_VERSION")
    validate_resource_probe(process_peak_memory())


def validate_parent_runtime_fully(repo: Path, runtime_path: Path) -> None:
    implementation = (
        repo
        / "scripts/research/egomotion_compensated_looming/"
        "rgb_segment_confirmation_r1/opaque_transport.py"
    )
    if (
        not implementation.is_file()
        or sha256_file(implementation)
        != "bc507980baa2ee6bffb9ffa515c2d79c8847aadd9387e537fad4e30bb41ee78f"
    ):
        raise ActivationPreflightFailure("R1_RUNTIME_VALIDATOR_BINDING")
    spec = importlib.util.spec_from_file_location(
        "r1_opaque_transport_runtime_validator", implementation
    )
    if spec is None or spec.loader is None:
        raise ActivationPreflightFailure("R1_RUNTIME_VALIDATOR_LOAD")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        module.validate_runtime_lock(repo, runtime_path)
    except Exception as error:
        raise ActivationPreflightFailure("R1_RUNTIME_FULL_VALIDATION") from error


def validate_r2_runtime(repo: Path, runtime_path: Path) -> None:
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if (
        runtime.get("schema_version")
        != "rcle_rgb_segment_confirmation_r2_openloris_runtime_lock.v1"
        or runtime.get("protocol_id") != "RCLE_RGB_SEGMENT_CONFIRMATION_R2"
    ):
        raise ActivationPreflightFailure("R2_RUNTIME_IDENTITY")
    python = runtime["python"]
    executable = Path(sys.executable).resolve()
    base_executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    if (
        str(executable).casefold()
        != str(Path(python["venv_executable"]).resolve()).casefold()
        or sha256_file(executable) != python["venv_executable_sha256"]
        or str(base_executable).casefold()
        != str(Path(python["base_executable"]).resolve()).casefold()
        or sha256_file(base_executable) != python["base_executable_sha256"]
    ):
        raise ActivationPreflightFailure("R2_RUNTIME_PYTHON")
    for distribution, version in runtime["distributions"].items():
        if importlib.metadata.version(distribution) != version:
            raise ActivationPreflightFailure("R2_RUNTIME_DISTRIBUTION")
    module_root = Path(__file__).resolve().parent
    for name, expected_hash in runtime["r2_modules"].items():
        if sha256_file(module_root / name) != expected_hash:
            raise ActivationPreflightFailure("R2_RUNTIME_MODULE")
    if os.environ.get("PYTHONPATH"):
        raise ActivationPreflightFailure("R2_RUNTIME_PYTHONPATH")
    validate_resource_probe(process_peak_memory())


def validate_activation(
    repo: Path,
    config_path: Path,
    activation_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    candidate_path = repo / ACTIVATION_CANDIDATE_PATH
    if not candidate_path.is_file():
        raise ActivationPreflightFailure("ACTIVATION_CANDIDATE_MISSING")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    expected_candidate_keys = {
        "schema_version",
        "status",
        "decision",
        "source_family_id",
        "capture_id",
        "window_id",
        "target_member_count",
        "remote_byte_hard_cap",
        "maximum_network_attempts_per_identical_range_for_entire_claim",
        "claim_namespace",
        "command",
        "bindings",
        "authority",
        "execution_authority",
    }
    if (
        set(candidate) != expected_candidate_keys
        or candidate.get("schema_version")
        != "rcle_rgb_segment_confirmation_r2_openloris_activation_candidate.v3"
        or candidate.get("status") != "ACTIVATION_REVIEW_REQUIRED"
        or candidate.get("decision") != "EXECUTION_NOT_AUTHORIZED"
        or candidate.get("source_family_id") != "OPENLORIS_CORRIDOR"
        or candidate.get("capture_id") != "corridor1-1"
        or candidate.get("window_id") != "corridor1-1:w004"
        or candidate.get("target_member_count") != 302
        or candidate.get("remote_byte_hard_cap") != 3_947_000_000
        or candidate.get(
            "maximum_network_attempts_per_identical_range_for_entire_claim"
        )
        != 3
        or candidate.get("claim_namespace")
        != "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/openloris_identity_run_v2"
        or candidate.get("execution_authority") is not False
    ):
        raise ActivationPreflightFailure("ACTIVATION_CANDIDATE_NOT_EXACT")
    expected_false_authority = {
        "openloris_identity_extraction": False,
        "rgb_decode": False,
        "rgb_algorithm_execution": False,
        "performance_qualification": False,
        "host_offline_replay": False,
        "android": False,
    }
    if candidate.get("authority") != expected_false_authority:
        raise ActivationPreflightFailure("ACTIVATION_CANDIDATE_AUTHORITY")
    binding_paths = {
        "contract": "docs/research/rcle/RCLE_RGB_SEGMENT_CONFIRMATION_R2_TRANSPORT_REPAIR_CONTRACT_2026-07-28.json",
        "design_review": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/design_independent_review.v1.json",
        "user_authorization": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/user_authorization.v1.json",
        "r1_lock": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r1/preaccess_lock.v11.json",
        "runner": runner_path.relative_to(repo).as_posix(),
        "diagnostic_transport": "scripts/research/egomotion_compensated_looming/rgb_segment_confirmation_r2/diagnostic_transport.py",
        "activation_preflight": "scripts/research/egomotion_compensated_looming/rgb_segment_confirmation_r2/activation_preflight.py",
        "config": config_path.relative_to(repo).as_posix(),
        "runtime_lock": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/openloris_runtime_lock.v1.json",
        "test_receipt": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/openloris_runner_test_receipt.v2.json",
        "static_secret_scan": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/openloris_secret_scan.v2.json",
    }
    expected_candidate_bindings = {
        name: {
            "path": relative,
            "sha256": sha256_file(repo / relative),
        }
        for name, relative in binding_paths.items()
    }
    if candidate.get("bindings") != expected_candidate_bindings:
        raise ActivationPreflightFailure("ACTIVATION_CANDIDATE_BINDINGS")
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "rcle_rgb_segment_confirmation_r2_openloris_activation.v1",
        "decision": "OPENLORIS_R2_ONE_SHOT_EXECUTION_AUTHORIZED",
        "execution_authority": True,
        "claim_namespace": candidate["claim_namespace"],
        "remote_byte_hard_cap": 3_947_000_000,
        "maximum_network_attempts_per_identical_range_for_entire_claim": 3,
        "bindings": {
            "candidate": {
                "path": ACTIVATION_CANDIDATE_PATH,
                "sha256": sha256_file(candidate_path),
            },
            **expected_candidate_bindings,
        },
        "authority": {
            "openloris_identity_extraction": True,
            "rgb_decode": False,
            "rgb_algorithm_execution": False,
            "performance_qualification": False,
            "host_offline_replay": False,
            "android": False,
        },
    }
    if activation != expected:
        raise ActivationPreflightFailure("ACTIVATION_RECEIPT_NOT_EXACT")
    return activation


def targets_from_lock(lock: dict[str, Any]) -> list[str]:
    segment = lock["segments"][0]
    if (
        segment.get("source_family_id") != "OPENLORIS_CORRIDOR"
        or segment.get("capture_id") != "corridor1-1"
        or segment.get("window_id") != "corridor1-1:w004"
        or segment.get("half_open_window_s")
        != ["1560000043.537699", "1560000053.537699"]
        or len(segment.get("frame_inventory", [])) != 300
    ):
        raise ActivationPreflightFailure("LOCKED_SEGMENT_IDENTITY")
    targets = [
        segment["guard_before"]["path"],
        *[row["rgb_member_path"] for row in segment["frame_inventory"]],
        segment["guard_after"]["path"],
    ]
    if len(targets) != 302 or len(set(targets)) != 302:
        raise ActivationPreflightFailure("TARGET_SET_NOT_EXACT_302")
    target_hash = hashlib.sha256("\n".join(targets).encode("utf-8")).hexdigest()
    identity_rows = [
        segment["guard_before"],
        *segment["frame_inventory"],
        segment["guard_after"],
    ]
    identity_hash = hashlib.sha256(
        json.dumps(
            identity_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if (
        target_hash
        != "612f55b4cf6973380e1233bc516efd9127f3cc44ea3a8e9f3813fa4d5aabc493"
        or identity_hash
        != "1092779c66922e9d15454b5df6f92482a7c823bf75219cd6ec7b8b4e9a8bbb06"
    ):
        raise ActivationPreflightFailure("TARGET_IDENTITY_MANIFEST")
    for target in targets:
        candidate = Path(target)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ActivationPreflightFailure("TARGET_PATH_ESCAPE")
    return targets


def append_phase(
    ledger: AppendOnlyLedger,
    phase: str,
    status: str,
    ordinal: int,
    remote: DiagnosticRemoteRange | None = None,
) -> dict[str, Any]:
    snapshot = remote.snapshot() if remote is not None else None
    return ledger.append(
        {
            "schema_version": "r2_openloris_phase.v1",
            "phase": phase,
            "status": status,
            "ordinal": ordinal,
            "monotonic_ns": time.monotonic_ns(),
            "transport": snapshot,
            "range_ledger_head_sha256": (
                remote.last_request.get("row_sha256")
                if remote is not None and remote.last_request is not None
                else None
            ),
            "resource": validate_resource_probe(process_peak_memory()),
        }
    )


def audit_orphan(namespace: Path) -> dict[str, Any]:
    claim = namespace / "claim.json"
    if not claim.is_file():
        return {"decision": "NO_CLAIM", "mutated": False}
    terminal_names = (
        "SUCCESS.json",
        "PARTIAL_QUARANTINED.json",
        "transport_failure.json",
        "ORPHAN_CONSUMED_NOT_EVALUABLE.json",
    )
    present = [name for name in terminal_names if (namespace / name).is_file()]
    if present:
        return {
            "decision": "TERMINAL_ALREADY_PRESENT",
            "terminal_files": present,
            "mutated": False,
        }
    phase_path = namespace / "phase_ledger.jsonl"
    range_path = namespace / "range_ledger.jsonl"
    receipt = {
        "schema_version": "r2_openloris_orphan_terminal.v1",
        "decision": "ORPHAN_CONSUMED_NOT_EVALUABLE",
        "claim_sha256": sha256_file(claim),
        "phase_ledger_sha256": (
            sha256_file(phase_path) if phase_path.is_file() else None
        ),
        "range_ledger_sha256": (
            sha256_file(range_path) if range_path.is_file() else None
        ),
        "resource": validate_resource_probe(process_peak_memory()),
        "resume_or_retry_authority": False,
        "execution_authority": False,
    }
    write_json_atomic(
        namespace / "ORPHAN_CONSUMED_NOT_EVALUABLE.json",
        receipt,
        exclusive=True,
    )
    return {**receipt, "mutated": True}


def run(
    repo: Path,
    config_path: Path,
    activation_path: Path,
) -> dict[str, Any]:
    import py7zr

    runner_path = Path(__file__).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(repo, config)
    validate_activation(repo, config_path, activation_path, runner_path)
    namespace = repo / config["claim_namespace"]
    if namespace.exists():
        audit_orphan(namespace)
        raise ActivationPreflightFailure("CLAIM_NAMESPACE_ALREADY_EXISTS")
    claim_path = namespace / "claim.json"
    output = namespace / "payloads"
    exclusive_claim(
        claim_path,
        {
            "protocol_id": config["protocol_id"],
            "source_family_id": "OPENLORIS_CORRIDOR",
            "window_id": "corridor1-1:w004",
            "config_sha256": sha256_file(config_path),
            "activation_sha256": sha256_file(activation_path),
            "runner_sha256": sha256_file(runner_path),
            "remote_byte_hard_cap": 3_947_000_000,
        },
    )
    phase_ledger = AppendOnlyLedger(namespace / "phase_ledger.jsonl")
    append_phase(phase_ledger, "ACTIVATION_VALIDATED", "END", 0)
    append_phase(phase_ledger, "CLAIM_CREATED", "END", 1)

    lock_path = repo / config["bindings"]["r1_preaccess_lock"]["path"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    targets = targets_from_lock(lock)
    segment = lock["segments"][0]
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        namespace / "ordered_target_manifest.json",
        {
            "schema_version": "r2_openloris_ordered_targets.v1",
            "count": len(targets),
            "ordered_targets": targets,
            "ordered_targets_sha256": hashlib.sha256(
                "\n".join(targets).encode("utf-8")
            ).hexdigest(),
        },
        exclusive=True,
    )
    bound_opener = IdentityBoundOpener(
        allowed_final_hosts=set(
            config["transport"]["allowed_final_url_hosts"]
        ),
        maximum_attempts_per_identical_range=3,
    )
    remote = DiagnosticRemoteRange(
        url=config["source"]["url"],
        length=config["source"]["object_length"],
        budget=config["transport"]["remote_byte_hard_cap"],
        ledger_path=namespace / "range_ledger.jsonl",
        progress_path=namespace / "progress.json",
        failure_receipt_path=namespace / "transport_failure.json",
        phase="ARCHIVE_HEADER_AND_DIRECTORY",
        user_agent="BlindAssist-RCLE-RGB-Segment-Confirmation-R2",
        maximum_attempts=3,
        timeout_seconds=90,
        resource_probe=process_peak_memory,
        opener=bound_opener,
    )
    adapter = RemoteFileAdapter(remote)
    append_phase(
        phase_ledger,
        "ARCHIVE_HEADER_AND_DIRECTORY",
        "BEGIN",
        2,
        remote,
    )
    try:
        with py7zr.SevenZipFile(adapter, mode="r") as archive:
            append_phase(
                phase_ledger,
                "ARCHIVE_HEADER_AND_DIRECTORY",
                "END",
                3,
                remote,
            )
            remote.phase = "SOLID_FOLDER_PACK_AND_TARGET_EMISSION"
            append_phase(
                phase_ledger,
                "SOLID_FOLDER_PACK_AND_TARGET_EMISSION",
                "BEGIN",
                4,
                remote,
            )
            archive.extract(path=output, targets=targets)
        append_phase(
            phase_ledger,
            "SOLID_FOLDER_PACK_AND_TARGET_EMISSION",
            "END",
            5,
            remote,
        )
        actual = sorted(
            path.relative_to(output).as_posix()
            for path in output.rglob("*")
            if path.is_file()
        )
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ActivationPreflightFailure("TARGET_LINK_FORBIDDEN")
        if actual != sorted(targets):
            raise ActivationPreflightFailure("TARGET_SET_MISMATCH")
        remote.phase = "TARGET_IDENTITY_VALIDATION"
        append_phase(
            phase_ledger,
            "TARGET_IDENTITY_VALIDATION",
            "BEGIN",
            6,
            remote,
        )
        expected_rows = {
            segment["guard_before"]["path"]: segment["guard_before"],
            segment["guard_after"]["path"]: segment["guard_after"],
            **{
                row["rgb_member_path"]: {
                    "uncompressed_bytes": row["rgb_member_uncompressed_bytes"],
                    "crc32": row["rgb_member_crc32"],
                }
                for row in segment["frame_inventory"]
            },
        }
        members = []
        for relative in targets:
            path = output / Path(relative)
            raw = path.read_bytes()
            expected = expected_rows[relative]
            if len(raw) != int(expected["uncompressed_bytes"]):
                raise ActivationPreflightFailure("TARGET_SIZE_MISMATCH")
            if f"{zlib.crc32(raw) & 0xffffffff:08x}" != expected["crc32"]:
                raise ActivationPreflightFailure("TARGET_CRC32_MISMATCH")
            members.append(
                {
                    "path": relative,
                    "bytes": len(raw),
                    "crc32": expected["crc32"],
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        append_phase(
            phase_ledger,
            "TARGET_IDENTITY_VALIDATION",
            "END",
            7,
            remote,
        )
        remote.phase = "RUNTIME_SECRET_SCAN"
        append_phase(
            phase_ledger,
            "RUNTIME_SECRET_SCAN",
            "BEGIN",
            8,
            remote,
        )
        runtime_scan_paths = [
            activation_path,
            repo / ACTIVATION_CANDIDATE_PATH,
            config_path,
            claim_path,
            namespace / "ordered_target_manifest.json",
            namespace / "phase_ledger.jsonl",
            namespace / "range_ledger.jsonl",
            namespace / "progress.json",
        ]
        runtime_scan_result = secret_scan(runtime_scan_paths)
        if runtime_scan_result["decision"] != "PASS":
            raise ActivationPreflightFailure("RUNTIME_SECRET_SCAN")
        runtime_scan_path = namespace / "runtime_secret_scan.json"
        write_json_atomic(
            runtime_scan_path,
            {
                "schema_version": "r2_openloris_runtime_secret_scan.v1",
                **runtime_scan_result,
                "scanned_files": [
                    {
                        "path": path.relative_to(repo).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for path in runtime_scan_paths
                ],
            },
            exclusive=True,
        )
        append_phase(
            phase_ledger,
            "RUNTIME_SECRET_SCAN",
            "END",
            9,
            remote,
        )
        remote.phase = "SUCCESS_COMMIT"
        success_commit_phase = append_phase(
            phase_ledger,
            "SUCCESS_COMMIT",
            "BEGIN",
            10,
            remote,
        )
        final_runtime_scan_result = secret_scan(runtime_scan_paths)
        if final_runtime_scan_result["decision"] != "PASS":
            raise ActivationPreflightFailure("FINAL_RUNTIME_SECRET_SCAN")
        write_json_atomic(
            runtime_scan_path,
            {
                "schema_version": "r2_openloris_runtime_secret_scan.v1",
                **final_runtime_scan_result,
                "scanned_files": [
                    {
                        "path": path.relative_to(repo).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for path in runtime_scan_paths
                ],
                "phase_ledger_final_head_sha256": success_commit_phase[
                    "row_sha256"
                ],
            },
            exclusive=False,
        )
        result = {
            "schema_version": "rcle_rgb_segment_confirmation_r2_openloris_terminal.v1",
            "decision": "OPENLORIS_EXACT_TARGET_IDENTITY_CLOSED",
            "member_count": len(members),
            "members": members,
            "transport": remote.snapshot(),
            "phase_ledger_head_sha256": success_commit_phase["row_sha256"],
            "runtime_secret_scan": {
                "path": runtime_scan_path.relative_to(repo).as_posix(),
                "sha256": sha256_file(runtime_scan_path),
            },
            "pixel_decode_calls": 0,
            "rgb_algorithm_calls": 0,
            "execution_authority": False,
        }
        write_json_atomic(namespace / "SUCCESS.json", result, exclusive=True)
        return result
    except BaseException:
        # Partial outputs remain under the consumed claim namespace and have no
        # evidentiary authority. The guard process records the terminal.
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-orphan", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo / args.config).resolve()
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(repo, config)
    if args.audit_orphan:
        result = audit_orphan(repo / config["claim_namespace"])
        print(json.dumps(result))
        return 0
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "decision": "PREFLIGHT_PASS_EXECUTION_NOT_AUTHORIZED",
                    "config_sha256": sha256_file(config_path),
                    "resource": validate_resource_probe(process_peak_memory()),
                }
            )
        )
        return 0
    if args.activation is None:
        raise ActivationPreflightFailure("ACTIVATION_RECEIPT_REQUIRED")
    activation_path = (
        args.activation.resolve()
        if args.activation.is_absolute()
        else (repo / args.activation).resolve()
    )
    try:
        result = run(repo, config_path, activation_path)
    except BaseException as error:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        namespace = repo / config["claim_namespace"]
        if (namespace / "claim.json").is_file():
            partial = []
            payload_root = namespace / "payloads"
            if payload_root.is_dir():
                for path in sorted(payload_root.rglob("*")):
                    if path.is_file():
                        partial.append(
                            {
                                "path": path.relative_to(payload_root).as_posix(),
                                "bytes": path.stat().st_size,
                                "sha256": sha256_file(path),
                            }
                        )
            marker = {
                "schema_version": "r2_openloris_partial_quarantine.v1",
                "decision": "NOT_EVALUABLE_PARTIAL_QUARANTINED",
                "error_type": type(error).__name__,
                "claim_sha256": sha256_file(namespace / "claim.json"),
                "phase_ledger_sha256": (
                    sha256_file(namespace / "phase_ledger.jsonl")
                    if (namespace / "phase_ledger.jsonl").is_file()
                    else None
                ),
                "range_ledger_sha256": (
                    sha256_file(namespace / "range_ledger.jsonl")
                    if (namespace / "range_ledger.jsonl").is_file()
                    else None
                ),
                "progress": (
                    json.loads(
                        (namespace / "progress.json").read_text(encoding="utf-8")
                    )
                    if (namespace / "progress.json").is_file()
                    else None
                ),
                "resource": validate_resource_probe(process_peak_memory()),
                "partial_file_count": len(partial),
                "partial_files": partial,
                "rgb_decode_calls": 0,
                "rgb_algorithm_calls": 0,
                "resume_or_retry_authority": False,
                "execution_authority": False,
            }
            marker_path = namespace / "PARTIAL_QUARANTINED.json"
            if not marker_path.exists():
                write_json_atomic(marker_path, marker, exclusive=True)
        print(
            json.dumps(
                {
                    "decision": "NOT_EVALUABLE",
                    "error_type": type(error).__name__,
                }
            )
        )
        return 1
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "member_count": result["member_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
