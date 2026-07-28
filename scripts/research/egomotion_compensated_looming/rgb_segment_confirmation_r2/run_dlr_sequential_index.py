from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
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
    write_json_atomic,
)
from dlr_streaming_bag_index import StreamingBagIndexer


ACTIVATION_CANDIDATE_PATH = (
    "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/"
    "dlr_index_activation_candidate.v2.json"
)
ACTIVATION_REVIEW_PATH = (
    "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/"
    "dlr_index_activation_independent_review.v1.json"
)


class SourceMemberIdentityFailure(RuntimeError):
    pass


def failure_decision(error: BaseException) -> str:
    return (
        "INVALID_SOURCE_OR_MEMBER_IDENTITY"
        if isinstance(error, SourceMemberIdentityFailure)
        else "DLR_INDEX_NOT_EVALUABLE"
    )


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
        encoding = response.headers.get("Content-Encoding")
        if (
            final.scheme != "https"
            or final.hostname not in self.allowed_final_hosts
            or encoding not in (None, "", "identity")
        ):
            response.close()
            raise OSError("SOURCE_RESPONSE_IDENTITY")
        return response


def require_exact_file(repo: Path, binding: dict[str, Any], label: str) -> Path:
    path = repo / binding.get("path", "")
    if not path.is_file() or sha256_file(path) != binding.get("sha256"):
        raise ActivationPreflightFailure(f"BINDING_MISMATCH:{label}")
    return path


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
        "dlr_r1_runtime_validator", implementation
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


def validate_config(repo: Path, config: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "source",
        "member",
        "window",
        "transport",
        "mode",
        "claim_namespace",
        "bindings",
        "pixel_firewall",
        "status",
        "execution_authority",
    }
    if set(config) != expected_keys:
        raise ActivationPreflightFailure("CONFIG_KEYS")
    if (
        config["schema_version"]
        != "rcle_rgb_segment_confirmation_r2_dlr_index_runner_config.v1"
        or config["protocol_id"] != "RCLE_RGB_SEGMENT_CONFIRMATION_R2"
        or config["mode"] != "SEQUENTIAL_INDEX_ONLY_NO_RANDOM_ACCESS"
        or config["claim_namespace"]
        != "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/dlr_index_run_v2"
        or config["status"] != "ACTIVATION_REVIEW_REQUIRED"
        or config["execution_authority"] is not False
    ):
        raise ActivationPreflightFailure("CONFIG_IDENTITY")
    if config["source"] != {
        "source_family_id": "DLR_RGBD_VICON",
        "capture_id": "extreme_geometry/hexagon_01",
        "window_id": "extreme_geometry/hexagon_01:w001",
        "url": "https://zenodo.org/api/records/10453700/files/realsense.zip/content",
        "outer_object_bytes": 45_718_173_762,
    }:
        raise ActivationPreflightFailure("CONFIG_SOURCE")
    if config["member"] != {
        "name": "extreme_geometry/hexagon_01.bag",
        "local_header_offset": 5_626_409_581,
        "compressed_data_start": 5_626_409_670,
        "compressed_bytes": 3_633_353_215,
        "uncompressed_bytes": 3_655_946_650,
        "compression_method": 8,
        "crc32": "bb79b456",
        "uncompressed_sha256": "c70984b06766e553d84703b3da62bdb210b9e8bb997542a260f26ddca03e2a99",
    }:
        raise ActivationPreflightFailure("CONFIG_MEMBER")
    if config["window"] != {
        "start_ns": 1_634_201_323_995_618_343,
        "end_ns": 1_634_201_333_995_618_343,
    }:
        raise ActivationPreflightFailure("CONFIG_WINDOW")
    if config["transport"] != {
        "remote_byte_hard_cap": 3_633_353_305,
        "retry_headroom": 0,
        "size_plus_one_final_preauthorization_margin": 1,
        "network_chunk_bytes": 8_388_608,
        "maximum_network_attempts_per_identical_range_for_entire_claim": 3,
        "accept_encoding": "identity",
        "allowed_final_url_hosts": ["zenodo.org", "files.zenodo.org"],
        "full_source_fallback": False,
        "r1_retry_or_resume": False,
    }:
        raise ActivationPreflightFailure("CONFIG_TRANSPORT")
    zero_firewall = {
        "rgb_payload_files_written": 0,
        "rgb_payload_bytes_retained": 0,
        "rgb_per_frame_payload_hash_calls": 0,
        "image_decode_calls": 0,
        "image_visualization_calls": 0,
        "rgb_algorithm_calls": 0,
    }
    if config["pixel_firewall"] != zero_firewall:
        raise ActivationPreflightFailure("CONFIG_PIXEL_FIREWALL")
    bindings = config["bindings"]
    require_exact_file(repo, bindings["r1_lock"], "R1_LOCK")
    require_exact_file(repo, bindings["amendment"], "AMENDMENT")
    require_exact_file(repo, bindings["amendment_review"], "AMENDMENT_REVIEW")
    runtime_path = repo / bindings["runtime_lock_path"]
    if not runtime_path.is_file():
        raise ActivationPreflightFailure("RUNTIME_LOCK_MISSING")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if (
        runtime["schema_version"]
        != "rcle_rgb_segment_confirmation_r2_dlr_index_runtime_lock.v1"
        or runtime["python"]["version"] != ".".join(map(str, sys.version_info[:3]))
        or sha256_file(Path(sys.executable))
        != runtime["python"]["venv_executable_sha256"]
        or sha256_file(Path(getattr(sys, "_base_executable", sys.executable)))
        != runtime["python"]["base_executable_sha256"]
    ):
        raise ActivationPreflightFailure("RUNTIME_IDENTITY")
    module_root = Path(__file__).resolve().parent
    for name, expected in runtime["r2_modules"].items():
        if sha256_file(module_root / name) != expected:
            raise ActivationPreflightFailure("RUNTIME_MODULE")
    if os.environ.get("PYTHONPATH"):
        raise ActivationPreflightFailure("RUNTIME_PYTHONPATH")
    parent_runtime = repo / runtime["parent_runtime_lock"]["path"]
    if (
        not parent_runtime.is_file()
        or sha256_file(parent_runtime)
        != runtime["parent_runtime_lock"]["sha256"]
    ):
        raise ActivationPreflightFailure("PARENT_RUNTIME_BINDING")
    validate_parent_runtime_fully(repo, parent_runtime)
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
    if (
        candidate.get("schema_version")
        != "rcle_rgb_segment_confirmation_r2_dlr_index_activation_candidate.v2"
        or candidate.get("status") != "ACTIVATION_REVIEW_REQUIRED"
        or candidate.get("decision") != "EXECUTION_NOT_AUTHORIZED"
        or candidate.get("claim_namespace")
        != "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/dlr_index_run_v2"
        or candidate.get("remote_byte_hard_cap") != 3_633_353_305
        or candidate.get("execution_authority") is not False
    ):
        raise ActivationPreflightFailure("ACTIVATION_CANDIDATE")
    binding_paths = {
        "amendment": "docs/research/rcle/RCLE_RGB_SEGMENT_CONFIRMATION_R2_DLR_SEQUENTIAL_INDEX_AMENDMENT_2026-07-28.json",
        "amendment_review": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/dlr_sequential_index_amendment_review.v1.json",
        "user_authorization": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/user_authorization.v1.json",
        "r1_lock": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r1/preaccess_lock.v11.json",
        "runner": runner_path.relative_to(repo).as_posix(),
        "streaming_index": "scripts/research/egomotion_compensated_looming/rgb_segment_confirmation_r2/dlr_streaming_bag_index.py",
        "diagnostic_transport": "scripts/research/egomotion_compensated_looming/rgb_segment_confirmation_r2/diagnostic_transport.py",
        "activation_preflight": "scripts/research/egomotion_compensated_looming/rgb_segment_confirmation_r2/activation_preflight.py",
        "config": config_path.relative_to(repo).as_posix(),
        "runtime_lock": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/dlr_index_runtime_lock.v1.json",
        "test_receipt": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/dlr_index_test_receipt.v1.json",
        "secret_scan": "artifacts.local/evidence/rcle_rgb_segment_confirmation_r2/dlr_index_secret_scan.v1.json",
    }
    expected_bindings = {
        name: {"path": path, "sha256": sha256_file(repo / path)}
        for name, path in binding_paths.items()
    }
    if candidate.get("bindings") != expected_bindings:
        raise ActivationPreflightFailure("ACTIVATION_CANDIDATE_BINDINGS")
    review_path = repo / ACTIVATION_REVIEW_PATH
    if not review_path.is_file():
        raise ActivationPreflightFailure("INDEPENDENT_ACTIVATION_REVIEW_MISSING")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if (
        review.get("schema_version")
        != "rcle_rgb_segment_confirmation_r2_dlr_index_activation_independent_review.v1"
        or review.get("protocol_id") != "RCLE_RGB_SEGMENT_CONFIRMATION_R2"
        or review.get("review_mode") != "READ_ONLY_NO_NETWORK"
        or review.get("candidate")
        != {
            "path": ACTIVATION_CANDIDATE_PATH,
            "sha256": sha256_file(candidate_path),
        }
        or review.get("decision") != "DLR_INDEX_ACTIVATION_REVIEW_PASS"
        or review.get("execution_authority") is not False
    ):
        raise ActivationPreflightFailure("INDEPENDENT_ACTIVATION_REVIEW")
    review_binding = {
        "independent_activation_review": {
            "path": ACTIVATION_REVIEW_PATH,
            "sha256": sha256_file(review_path),
        }
    }
    expected = {
        "schema_version": "rcle_rgb_segment_confirmation_r2_dlr_index_activation.v1",
        "decision": "DLR_R2_SEQUENTIAL_INDEX_ONE_SHOT_EXECUTION_AUTHORIZED",
        "execution_authority": True,
        "claim_namespace": candidate["claim_namespace"],
        "remote_byte_hard_cap": 3_633_353_305,
        "bindings": {
            "candidate": {
                "path": ACTIVATION_CANDIDATE_PATH,
                "sha256": sha256_file(candidate_path),
            },
            **expected_bindings,
            **review_binding,
        },
        "authority": {
            "dlr_sequential_index_only": True,
            "rgb_payload_retention": False,
            "rgb_decode": False,
            "rgb_algorithm_execution": False,
            "exact_window_rgb_extraction": False,
            "host_offline_replay": False,
            "android": False,
        },
    }
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    if activation != expected:
        raise ActivationPreflightFailure("ACTIVATION_RECEIPT")
    return activation


def ledger_rows(
    path: Path,
    schema_version: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ledger = AppendOnlyLedger(path)
    head = None
    for row in rows:
        head = ledger.append({"schema_version": schema_version, **row})
    return {
        "path": path,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "head_sha256": head["row_sha256"] if head else None,
    }


def summarize_append_only_ledger(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "path": path,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "head_sha256": rows[-1]["row_sha256"] if rows else None,
        "accounted_byte_sum": sum(
            int(row.get("accounted_bytes", 0)) for row in rows
        ),
        "logical_request_count": len(
            {row.get("logical_request_id") for row in rows}
        ),
        "attempt_count": len(rows),
        "retry_count": sum(int(row.get("attempt", 1)) > 1 for row in rows),
    }


def stream_deflate_member(
    remote: DiagnosticRemoteRange,
    *,
    compressed_bytes: int,
    network_chunk_bytes: int,
    parser: StreamingBagIndexer,
) -> dict[str, Any]:
    decompressor = zlib.decompressobj(-15)
    compressed_hash = hashlib.sha256()
    bag_hash = hashlib.sha256()
    bag_crc = 0
    compressed_count = 0
    uncompressed_count = 0
    observations: list[dict[str, Any]] = []
    remaining = compressed_bytes
    while remaining:
        size = min(network_chunk_bytes, remaining)
        compressed = remote.read(size)
        if len(compressed) != size:
            raise ActivationPreflightFailure("COMPRESSED_MEMBER_TRUNCATED")
        compressed_hash.update(compressed)
        compressed_count += len(compressed)
        remaining -= len(compressed)
        pending = compressed
        emitted_this_input = 0
        while pending:
            output = decompressor.decompress(pending, 1 << 20)
            pending = decompressor.unconsumed_tail
            if output:
                bag_hash.update(output)
                bag_crc = zlib.crc32(output, bag_crc)
                uncompressed_count += len(output)
                emitted_this_input += len(output)
                parser.feed(output)
            if not pending:
                break
        observations.append(
            {
                "compressed_bytes_fed": compressed_count,
                "uncompressed_bytes_emitted": uncompressed_count,
                "input_uncompressed_bytes_emitted": emitted_this_input,
                "outer_next_offset": remote.position,
                "restartable": False,
            }
        )
    tail = decompressor.flush()
    if tail:
        bag_hash.update(tail)
        bag_crc = zlib.crc32(tail, bag_crc)
        uncompressed_count += len(tail)
        parser.feed(tail)
    return {
        "decompressor": decompressor,
        "compressed_count": compressed_count,
        "compressed_sha256": compressed_hash.hexdigest(),
        "uncompressed_count": uncompressed_count,
        "bag_crc32": f"{bag_crc & 0xffffffff:08x}",
        "bag_sha256": bag_hash.hexdigest(),
        "observations": observations,
    }


def run(repo: Path, config_path: Path, activation_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runner_path = Path(__file__).resolve()
    validate_config(repo, config)
    validate_activation(repo, config_path, activation_path, runner_path)
    namespace = repo / config["claim_namespace"]
    if namespace.exists():
        raise ActivationPreflightFailure("CLAIM_NAMESPACE_ALREADY_EXISTS")
    claim_path = namespace / "claim.json"
    exclusive_claim(
        claim_path,
        {
            "protocol_id": config["protocol_id"],
            "mode": config["mode"],
            "member": config["member"]["name"],
            "config_sha256": sha256_file(config_path),
            "activation_sha256": sha256_file(activation_path),
            "runner_sha256": sha256_file(runner_path),
            "remote_byte_hard_cap": 3_633_353_305,
        },
    )
    opener = IdentityBoundOpener(
        allowed_final_hosts=set(
            config["transport"]["allowed_final_url_hosts"]
        ),
        maximum_attempts_per_identical_range=3,
    )
    remote = DiagnosticRemoteRange(
        url=config["source"]["url"],
        length=config["source"]["outer_object_bytes"],
        budget=config["transport"]["remote_byte_hard_cap"],
        ledger_path=namespace / "range_ledger.jsonl",
        progress_path=namespace / "progress.json",
        failure_receipt_path=namespace / "transport_failure.json",
        phase="ZIP_LOCAL_HEADER",
        user_agent="BlindAssist-RCLE-DLR-Sequential-Index-R2",
        opener=opener,
        resource_probe=process_peak_memory,
    )
    member = config["member"]
    remote.seek(member["local_header_offset"])
    fixed = remote.read(30)
    if len(fixed) != 30 or fixed[:4] != b"PK\x03\x04":
        raise SourceMemberIdentityFailure("ZIP_LOCAL_HEADER")
    unpacked = struct.unpack("<IHHHHHIIIHH", fixed)
    method = unpacked[3]
    name_length = unpacked[9]
    extra_length = unpacked[10]
    if method != 8 or name_length + extra_length != 59:
        raise SourceMemberIdentityFailure("ZIP_LOCAL_HEADER_IDENTITY")
    variable = remote.read(name_length + extra_length)
    if variable[:name_length].decode("utf-8") != member["name"]:
        raise SourceMemberIdentityFailure("ZIP_MEMBER_NAME")
    if remote.position != member["compressed_data_start"]:
        raise SourceMemberIdentityFailure("ZIP_COMPRESSED_DATA_START")

    remote.phase = "SEQUENTIAL_MEMBER_INDEX"
    parser = StreamingBagIndexer(
        start_ns=config["window"]["start_ns"],
        end_ns=config["window"]["end_ns"],
    )
    scan_result = stream_deflate_member(
        remote,
        compressed_bytes=member["compressed_bytes"],
        network_chunk_bytes=config["transport"]["network_chunk_bytes"],
        parser=parser,
    )
    decompressor = scan_result["decompressor"]
    compressed_count = scan_result["compressed_count"]
    uncompressed_count = scan_result["uncompressed_count"]
    observations = scan_result["observations"]
    terminal = parser.terminal()
    member_identity_complete = (
        decompressor.eof
        and not decompressor.unused_data
        and compressed_count == member["compressed_bytes"]
        and uncompressed_count == member["uncompressed_bytes"]
        and scan_result["bag_crc32"] == member["crc32"]
        and scan_result["bag_sha256"] == member["uncompressed_sha256"]
        and terminal["magic_complete"]
        and terminal["partial_buffer_length"] == 0
    )
    identity_complete = member_identity_complete and all(
        value == 0 for value in terminal["pixel_firewall"].values()
    )

    top_binding = ledger_rows(
        namespace / "top_level_records.jsonl",
        "r2_dlr_top_record.v1",
        parser.top_level_records,
    )
    connection_rows = [
        value for _, value in sorted(parser.connections.items())
    ]
    connection_binding = ledger_rows(
        namespace / "connections.jsonl",
        "r2_dlr_connection.v1",
        connection_rows,
    )
    time_rows = []
    for connection_id, window in sorted(parser.windows.items()):
        time_rows.append(
            {
                "connection_id": connection_id,
                "message_count": window.count,
                "minimum_bag_timestamp_ns": window.minimum_bag_timestamp_ns,
                "maximum_bag_timestamp_ns": window.maximum_bag_timestamp_ns,
                "before": window.before,
                "selected": window.selected,
                "after": window.after,
            }
        )
    chunk_binding = ledger_rows(
        namespace / "chunk_index.jsonl",
        "r2_dlr_chunk_index.v1",
        parser.chunk_records,
    )
    time_binding = ledger_rows(
        namespace / "connection_time_windows.jsonl",
        "r2_dlr_connection_time.v1",
        time_rows,
    )
    observation_binding = ledger_rows(
        namespace / "offset_observations.jsonl",
        "r2_dlr_nonrestartable_offset_observation.v1",
        observations,
    )
    candidates = terminal["candidate_color_connections"]
    unique_complete_candidate = (
        len(candidates) == 1
        and candidates[0]["before"] is not None
        and bool(candidates[0]["selected"])
        and candidates[0]["after"] is not None
    )
    if not member_identity_complete:
        decision = "INVALID_SOURCE_OR_MEMBER_IDENTITY"
    elif identity_complete and unique_complete_candidate:
        decision = "DLR_SEQUENTIAL_INDEX_COMPLETE_EXECUTION_NOT_AUTHORIZED"
    else:
        decision = "DLR_INDEX_NOT_EVALUABLE"
    range_binding = summarize_append_only_ledger(
        namespace / "range_ledger.jsonl"
    )
    result = {
        "schema_version": "rcle_rgb_segment_confirmation_r2_dlr_index_terminal.v1",
        "decision": decision,
        "identity_complete": identity_complete,
        "member": member,
        "scan": {
            "compressed_bytes_read": compressed_count,
            "compressed_sha256": scan_result["compressed_sha256"],
            "uncompressed_bytes_emitted": uncompressed_count,
            "deflate_eof": decompressor.eof,
            "bag_crc32": scan_result["bag_crc32"],
            "bag_sha256": scan_result["bag_sha256"],
            "remote": remote.snapshot(),
        },
        "rosbag": terminal,
        "ledgers": {
            "top_level_records": {
                **top_binding,
                "path": top_binding["path"].relative_to(repo).as_posix(),
            },
            "connections": {
                **connection_binding,
                "path": connection_binding["path"].relative_to(repo).as_posix(),
            },
            "chunk_index": {
                **chunk_binding,
                "path": chunk_binding["path"].relative_to(repo).as_posix(),
            },
            "connection_time_windows": {
                **time_binding,
                "path": time_binding["path"].relative_to(repo).as_posix(),
            },
            "offset_observations": {
                **observation_binding,
                "path": observation_binding["path"].relative_to(repo).as_posix(),
            },
            "range": {
                **range_binding,
                "path": range_binding["path"].relative_to(repo).as_posix(),
            },
        },
        "pixel_firewall": terminal["pixel_firewall"],
        "resumability": terminal["resumability"],
        "execution_authority": False,
    }
    scan_paths = [
        activation_path,
        repo / ACTIVATION_CANDIDATE_PATH,
        config_path,
        claim_path,
        namespace / "range_ledger.jsonl",
        namespace / "progress.json",
        namespace / "top_level_records.jsonl",
        namespace / "connections.jsonl",
        namespace / "chunk_index.jsonl",
        namespace / "connection_time_windows.jsonl",
        namespace / "offset_observations.jsonl",
    ]
    scan = secret_scan(scan_paths)
    if scan["decision"] != "PASS":
        raise ActivationPreflightFailure("RUNTIME_SECRET_SCAN")
    runtime_scan_path = namespace / "runtime_secret_scan.json"
    write_json_atomic(
        runtime_scan_path,
        {
            "schema_version": "r2_dlr_runtime_secret_scan.v1",
            **scan,
            "scanned_files": [
                {
                    "path": path.relative_to(repo).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in scan_paths
            ],
        },
        exclusive=True,
    )
    result["runtime_secret_scan"] = {
        "path": runtime_scan_path.relative_to(repo).as_posix(),
        "sha256": sha256_file(runtime_scan_path),
    }
    write_json_atomic(namespace / "TERMINAL.json", result, exclusive=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--activation", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo / args.config).resolve()
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(repo, config)
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
        raise ActivationPreflightFailure("ACTIVATION_REQUIRED")
    activation_path = (
        args.activation.resolve()
        if args.activation.is_absolute()
        else (repo / args.activation).resolve()
    )
    try:
        result = run(repo, config_path, activation_path)
    except BaseException as error:
        namespace = repo / config["claim_namespace"]
        if (namespace / "claim.json").is_file():
            decision = failure_decision(error)
            range_path = namespace / "range_ledger.jsonl"
            failure = {
                "schema_version": "r2_dlr_index_failure.v1",
                "decision": decision,
                "error_type": type(error).__name__,
                "claim_sha256": sha256_file(namespace / "claim.json"),
                "range_ledger_sha256": (
                    sha256_file(namespace / "range_ledger.jsonl")
                    if (namespace / "range_ledger.jsonl").is_file()
                    else None
                ),
                "resource": validate_resource_probe(process_peak_memory()),
                "pixel_firewall": config["pixel_firewall"],
                "resume_or_retry_authority": False,
                "execution_authority": False,
            }
            if range_path.is_file():
                binding = summarize_append_only_ledger(range_path)
                failure["range_ledger"] = {
                    **binding,
                    "path": binding["path"].relative_to(repo).as_posix(),
                }
            failure_path = namespace / "FAILURE.json"
            if not failure_path.exists():
                write_json_atomic(failure_path, failure, exclusive=True)
            terminal_path = namespace / "TERMINAL.json"
            if not terminal_path.exists():
                write_json_atomic(terminal_path, failure, exclusive=True)
        print(
            json.dumps(
                {
                    "decision": failure_decision(error),
                    "error_type": type(error).__name__,
                }
            )
        )
        return 1
    print(json.dumps({"decision": result["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
