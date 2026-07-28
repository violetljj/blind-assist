from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any

CANDIDATE_PATH = (
    "artifacts.local/evidence/rcle_rgb_segment_confirmation_mvsec_r2/"
    "candidate_lock.v1.json"
)
REVIEW_PATH = (
    "artifacts.local/evidence/rcle_rgb_segment_confirmation_mvsec_r2/"
    "candidate_independent_review.v1.json"
)
AUTHORIZATION_PATH = (
    "artifacts.local/evidence/rcle_rgb_segment_confirmation_mvsec_r2/"
    "new_network_claim_authorization.v1.json"
)


class BootstrapFailure(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_bootstrap_bindings(
    repo: Path,
    config: dict[str, Any],
) -> None:
    bindings = config.get("bootstrap_bindings", {})
    for name in (
        "capture_runner",
        "evidence_module",
        "core_runner",
        "parser",
    ):
        binding = bindings.get(name)
        if (
            not isinstance(binding, dict)
            or not (repo / binding.get("path", "")).is_file()
            or sha256_file(repo / binding["path"]) != binding.get("sha256")
        ):
            raise BootstrapFailure("R2_BOOTSTRAP_BINDING")


def import_verified_modules(
    repo: Path,
    config: dict[str, Any],
):
    bindings = config["bootstrap_bindings"]
    r1_directory = (repo / bindings["core_runner"]["path"]).parent
    r2_directory = (repo / bindings["capture_runner"]["path"]).parent
    sys.path.insert(0, str(r1_directory))
    sys.path.insert(0, str(r2_directory))
    evidence = importlib.import_module("mvsec_identity_evidence_r2")
    capture_module = importlib.import_module("mvsec_identity_capture_r2")
    core = importlib.import_module("run_mvsec_identity")
    return core, evidence, capture_module


def validate_authority(
    repo: Path,
    config_path: Path,
    activation_path: Path,
    *,
    core: Any,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    candidate_path = repo / CANDIDATE_PATH
    review_path = repo / REVIEW_PATH
    authorization_path = repo / AUTHORIZATION_PATH
    if (
        not candidate_path.is_file()
        or not review_path.is_file()
        or not authorization_path.is_file()
    ):
        raise core.IdentityFailure("R2_REVIEW_GATE_MISSING")
    candidate = load_json(candidate_path)
    review = load_json(review_path)
    authorization = load_json(authorization_path)
    candidate_sha = sha256_file(candidate_path)
    resolved_runner = (
        runner_path.resolve()
        if runner_path is not None
        else Path(__file__).resolve()
    )
    runtime_bindings = {
        "config": {
            "path": config_path.relative_to(repo).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "runner": {
            "path": resolved_runner.relative_to(repo).as_posix(),
            "sha256": sha256_file(resolved_runner),
        },
    }
    for name, expected in runtime_bindings.items():
        if candidate.get("bindings", {}).get(name) != expected:
            raise core.IdentityFailure("R2_CANDIDATE_RUNTIME_BINDING")
    for name in (
        "capture_runner",
        "diagnostics",
        "core_runner",
        "parser",
        "tests",
    ):
        binding = candidate.get("bindings", {}).get(name)
        if (
            not isinstance(binding, dict)
            or not (repo / binding.get("path", "")).is_file()
            or sha256_file(repo / binding["path"]) != binding.get("sha256")
        ):
            raise core.IdentityFailure("R2_CANDIDATE_RUNTIME_BINDING")
    if (
        candidate.get("decision") != "R2_IDENTITY_EXTRACTION_NOT_AUTHORIZED"
        or candidate.get("execution_authority") is not False
        or review.get("decision")
        != "MVSEC_RGB_IDENTITY_R2_CANDIDATE_REVIEW_PASS"
        or review.get("candidate_sha256") != candidate_sha
        or review.get("execution_authority") is not False
        or authorization.get("decision")
        != "MVSEC_RGB_IDENTITY_R2_NEW_NETWORK_CLAIM_AUTHORIZED"
        or authorization.get("candidate_sha256") != candidate_sha
        or authorization.get("new_network_claim_authority") is not True
        or authorization.get("retry_or_resume_r1_authority") is not False
        or authorization.get("full_bag_authority") is not False
    ):
        raise core.IdentityFailure("R2_REVIEW_GATE")
    expected = {
        "schema_version": "rcle_mvsec_rgb_identity_r2_activation.v1",
        "decision": "MVSEC_RGB_IDENTITY_R2_ONE_SHOT_AUTHORIZED",
        "execution_authority": True,
        "bindings": {
            "candidate": {
                "path": CANDIDATE_PATH,
                "sha256": candidate_sha,
            },
            "review": {
                "path": REVIEW_PATH,
                "sha256": sha256_file(review_path),
            },
            "new_network_claim_authorization": {
                "path": AUTHORIZATION_PATH,
                "sha256": sha256_file(authorization_path),
            },
            **runtime_bindings,
            "capture_runner": candidate["bindings"]["capture_runner"],
            "diagnostics": candidate["bindings"]["diagnostics"],
            "core_runner": candidate["bindings"]["core_runner"],
            "parser": candidate["bindings"]["parser"],
            "tests": candidate["bindings"]["tests"],
        },
        "authority": {
            "bag_index_and_target_chunks": True,
            "image_metadata_ledger": True,
            "selected_mono8_payload_materialization": True,
            "image_decode": False,
            "rectification": False,
            "rgb_algorithm": False,
            "android": False,
        },
    }
    if load_json(activation_path) != expected:
        raise core.IdentityFailure("R2_ACTIVATION_IDENTITY")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--activation", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repo / args.config).resolve()
    )
    activation_path = (
        args.activation.resolve()
        if args.activation.is_absolute()
        else (repo / args.activation).resolve()
    )
    config = load_json(config_path)
    if (
        config.get("schema_version") != "rcle_mvsec_rgb_identity_r2_config.v1"
        or config.get("protocol_id")
        != "RCLE_RGB_SEGMENT_CONFIRMATION_MVSEC_R2"
        or config.get("execution_authority") is not False
        or config.get("claim_namespace")
        != "artifacts.local/evidence/rcle_rgb_segment_confirmation_mvsec_r2/identity_run_v1"
    ):
        raise BootstrapFailure("R2_CONFIG_IDENTITY")
    verify_bootstrap_bindings(repo, config)
    core, evidence, capture_module = import_verified_modules(repo, config)
    base_config = core.load_bound(repo, config["base_config"])
    if (
        base_config.get("schema_version")
        != "rcle_mvsec_rgb_identity_config.v1"
        or base_config.get("protocol_id")
        != "RCLE_RGB_SEGMENT_CONFIRMATION_MVSEC_R1"
    ):
        raise core.IdentityFailure("R2_BASE_CONFIG_IDENTITY")
    consumed = config.get("consumed_r1", {})
    for name in (
        "candidate",
        "review",
        "activation",
        "claim",
        "failure",
        "range_ledger",
        "core_runner",
        "parser",
    ):
        binding = consumed.get(name)
        if (
            not isinstance(binding, dict)
            or not (repo / binding.get("path", "")).is_file()
            or sha256_file(repo / binding["path"]) != binding.get("sha256")
        ):
            raise core.IdentityFailure("R2_CONSUMED_R1_BINDING")
    r1_failure = load_json(repo / consumed["failure"]["path"])
    if (
        r1_failure.get("decision") != "MVSEC_RGB_IDENTITY_NOT_EVALUABLE"
        or r1_failure.get("retry_or_resume_authority") is not False
        or r1_failure.get("algorithm_execution_authority") is not False
    ):
        raise core.IdentityFailure("R2_CONSUMED_R1_TERMINAL_IDENTITY")
    calibration_path = repo / base_config["camera_relation"]["calibration_path"]
    if (
        not calibration_path.is_file()
        or sha256_file(calibration_path)
        != base_config["camera_relation"]["calibration_sha256"]
    ):
        raise core.IdentityFailure("CALIBRATION_BINDING")
    validate_authority(
        repo,
        config_path,
        activation_path,
        core=core,
    )
    namespace = repo / config["claim_namespace"]
    if namespace.exists():
        raise FileExistsError("R2_CLAIM_NAMESPACE_EXISTS")
    core.write_json_exclusive(
        namespace / "claim.json",
        {
            "schema_version": "rcle_mvsec_rgb_identity_r2_claim.v1",
            "config_sha256": sha256_file(config_path),
            "activation_sha256": sha256_file(activation_path),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "diagnostics_sha256": sha256_file(
                Path(__file__).with_name("mvsec_identity_evidence_r2.py")
            ),
            "capture_runner_sha256": sha256_file(
                Path(__file__).with_name("mvsec_identity_capture_r2.py")
            ),
            "core_runner_sha256": sha256_file(Path(core.__file__).resolve()),
            "retry_or_resume_authority": False,
        },
    )
    try:
        captures = []
        for capture in base_config["captures"]:
            geometry_source = core.load_bound(repo, capture["geometry_source"])
            geometry_result = core.load_bound(repo, capture["geometry_result"])
            core.validate_geometry_window(
                geometry_result,
                window=capture["window"],
            )
            captures.append(
                capture_module.inspect_capture_r2(
                    core,
                    repo=repo,
                    namespace=namespace,
                    capture=capture,
                    geometry_source=geometry_source,
                )
            )
        terminal = {
            "schema_version": "rcle_mvsec_rgb_identity_r2_terminal.v1",
            "decision": "MVSEC_RGB_EXACT_CAPTURE_IDENTITY_COMPLETE",
            "captures": captures,
            "camera_relation": base_config["camera_relation"],
            "adapter": base_config["adapter"],
            "payload_scope": {
                "selected_windows": 2,
                "guard_frames_per_window": 2,
                "full_bag_downloaded": False,
                "non_target_payload_files": 0,
            },
            "image_decode_calls": 0,
            "rectification_calls": 0,
            "rgb_algorithm_calls": 0,
            "algorithm_execution_authority": False,
            "android_authority": False,
        }
        core.write_json_exclusive(namespace / "TERMINAL.json", terminal)
        print(json.dumps({"decision": terminal["decision"]}))
        return 0
    except BaseException as error:
        terminal = {
            "schema_version": "rcle_mvsec_rgb_identity_r2_failure.v1",
            "decision": core.terminal_decision(error),
            **evidence.classify_exception(error),
            "claim_sha256": sha256_file(namespace / "claim.json"),
            "evidence_bindings": evidence.collect_evidence_bindings(
                repo,
                namespace,
            ),
            "retry_or_resume_authority": False,
            "algorithm_execution_authority": False,
            "android_authority": False,
        }
        core.write_json_exclusive(namespace / "FAILURE.json", terminal)
        print(json.dumps(terminal))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
