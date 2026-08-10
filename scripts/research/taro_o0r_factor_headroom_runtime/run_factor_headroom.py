#!/usr/bin/env python3
"""Run the hash-bound TARO O0R DepthART factor-headroom one-shot."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil

from scripts.research.taro_o0r_factor_headroom_runtime.candidate_phase import (
    load_sealed_candidate_frame,
    run_candidate_phase,
)
from scripts.research.taro_o0r_factor_headroom_runtime.depthart_runner import (
    bind_sealed_candidate_to_source,
    load_official_depthart,
    validate_depthart_inference_receipt,
)
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import (
    FactorEvidenceError,
    FactorEvidenceWriter,
)
from scripts.research.taro_o0r_factor_headroom_runtime.factor_evaluator import (
    CRITICAL_STRATA,
    evaluate_admitted_frame,
)
from scripts.research.taro_o0r_factor_headroom_runtime.factor_canary import (
    CLAIM_CEILING as FACTOR_CANARY_CLAIM_CEILING,
    build_factor_canary_record,
    summarize_factor_canary,
)
from scripts.research.taro_o0r_factor_headroom_runtime.statistics import (
    evaluate_factor_diagnostics,
    evaluate_factor_headroom,
)
from scripts.research.taro_o0r_factor_headroom_runtime.truth_recompute import recompute_committed_truth
from scripts.research.taro_o0r_factor_headroom_runtime.uncertainty_loader import load_factory_bound_uncertainty_model
from scripts.research.taro_o0r_factor_headroom_runtime.uncertainty_refit import refit_and_verify_uncertainty_model
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o0r_truth_materializer_runtime import run_truth_only as truth_runner


EXECUTION_LOCK_SCHEMA = "blindassist.taro.o0r.factor_headroom_r3_execution_lock.v1"
EXECUTION_LOCK_ID = "TARO_O0R_ARKITSCENES_FACTOR_HEADROOM_R3_ONE_SHOT_EXECUTION_LOCK"
RESULT_SCHEMA = "blindassist.taro.o0r.factor_headroom_r3_result.v1"
MANIFEST_SCHEMA = "blindassist.taro.o0r.factor_headroom_manifest.v1"
EXPECTED_ROOTS = {
    "SOURCE": "artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r3",
    "TRUTH_EVIDENCE": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3",
    "FACTOR_EVIDENCE": "artifacts.local/evidence/taro/o0r-arkitscenes-factor-headroom-r3",
}
EXPECTED_AUTHORITY = {
    "depthart_inference": True,
    "factorial_execution": True,
    "descriptive_partial_factor_canary": True,
    "source_cache_reuse": True,
    "truth_recomputation_after_candidate_seal": True,
    "training": False,
    "network": False,
    "device": False,
    "product": False,
    "safety": False,
}
EXPECTED_BUDGET = {
    "wall_seconds": 28_800,
    "peak_rss_bytes": 17_179_869_184,
    "maximum_evidence_bytes": 2_147_483_648,
    "maximum_cuda_allocated_bytes": 8_500_000_000,
    "training_steps": 0,
    "network_requests": 0,
}
EXPECTED_RUNTIME = {
    "python": "3.11.9",
    "numpy": "2.1.3",
    "opencv": "4.10.0",
    "torch": "2.11.0+cu128",
    "torch_cuda": "12.8",
    "cuda_device": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "psutil": "7.2.2",
    "pillow": "12.2.0",
}
EXPECTED_BINDING_PATHS = {
    "FACTOR_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_FACTOR_HEADROOM_R3_IMPLEMENTATION_LOCK_2026-08-10.json",
    "R3_RESULT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/result.json",
    "R3_COMPLETION": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/completion-receipt.json",
    "R3_MANIFEST": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/manifest.json",
    "R3_EXACT_FRAME_PLAN": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/exact-frame-plan.json.gz",
    "R3_DOWNLOAD_RECEIPTS": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/download-receipts.json.gz",
    "R3_UNCERTAINTY_RECEIPT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-receipt.json",
    "R3_UNCERTAINTY_ARTIFACT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-artifact.json.gz",
    "TRUTH_PREFLIGHT": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_ONLY_PREFLIGHT_R1_LOCK_2026-08-10.json",
    "R3_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R3_IMPLEMENTATION_LOCK_2026-08-10.json",
    "R3_EXECUTION_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_ONLY_R3_EXECUTION_LOCK_2026-08-10.json",
    "SOURCE_ADAPTER": "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py",
    "MATERIALIZER": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "TRUTH_RUNNER": "scripts/research/taro_o0r_truth_materializer_runtime/run_truth_only.py",
    "DEPTHART_RUNNER": "scripts/research/taro_o0r_factor_headroom_runtime/depthart_runner.py",
    "CANDIDATE_INPUTS": "scripts/research/taro_o0r_factor_headroom_runtime/candidate_inputs.py",
    "CANDIDATE_PHASE": "scripts/research/taro_o0r_factor_headroom_runtime/candidate_phase.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "UNCERTAINTY_LOADER": "scripts/research/taro_o0r_factor_headroom_runtime/uncertainty_loader.py",
    "UNCERTAINTY_REFIT": "scripts/research/taro_o0r_factor_headroom_runtime/uncertainty_refit.py",
    "TRUTH_RECOMPUTE": "scripts/research/taro_o0r_factor_headroom_runtime/truth_recompute.py",
    "FACTOR_HEADROOM": "scripts/research/taro_o0r_factor_headroom_runtime/factor_headroom.py",
    "FACTOR_CANARY": "scripts/research/taro_o0r_factor_headroom_runtime/factor_canary.py",
    "FACTOR_EVALUATOR": "scripts/research/taro_o0r_factor_headroom_runtime/factor_evaluator.py",
    "STATISTICS": "scripts/research/taro_o0r_factor_headroom_runtime/statistics.py",
    "FACTOR_RUNNER": "scripts/research/taro_o0r_factor_headroom_runtime/run_factor_headroom.py",
}
STRUCTURALLY_NOT_APPLICABLE_STRATA = {
    "orientation": "SOURCE_CONTRACT_FIXES_ALL_REGISTERED_RASTERS_TO_LANDSCAPE_1440X1920_NO_PORTRAIT_LEVEL_EXISTS",
}


class FactorExecutionError(RuntimeError):
    """Stable fail-closed factor one-shot error."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FactorExecutionError(code, message, **context)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT.resolve(), relative)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise FactorExecutionError("BOUND_JSON_INVALID", "bound JSON cannot be decoded", path=str(path)) from error


def _load_json_gzip(path: Path) -> Any:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    except Exception as error:
        raise FactorExecutionError("BOUND_GZIP_JSON_INVALID", "bound gzip JSON cannot be decoded", path=str(path)) from error


def validate_execution_lock(lock_path: Path) -> dict[str, Any]:
    lock = _load_json(lock_path.resolve())
    require(lock.get("schema") == EXECUTION_LOCK_SCHEMA and lock.get("lock_id") == EXECUTION_LOCK_ID, "FACTOR_EXECUTION_LOCK_IDENTITY_DRIFT", "factor execution lock schema/id drift")
    require(lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "FACTOR_ONE_SHOT_ALREADY_CONSUMED", "factor execution lock is not authorized and unconsumed")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "FACTOR_EXECUTION_AUTHORITY_DRIFT", "factor execution authority drift")
    require(lock.get("roots") == EXPECTED_ROOTS, "FACTOR_EXECUTION_ROOT_DRIFT", "factor execution root drift")
    require(lock.get("resource_budget") == EXPECTED_BUDGET, "FACTOR_EXECUTION_BUDGET_DRIFT", "factor execution budget drift")
    require(lock.get("runtime") == EXPECTED_RUNTIME, "FACTOR_EXECUTION_RUNTIME_DRIFT", "factor execution runtime lock drift")
    import cv2
    import PIL
    import torch

    observed_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cuda_device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CUDA_UNAVAILABLE",
        "psutil": psutil.__version__,
        "pillow": PIL.__version__,
    }
    require(observed_runtime == EXPECTED_RUNTIME, "FACTOR_EXECUTION_RUNTIME_DRIFT", "installed factor execution runtime differs from lock", observed=observed_runtime)
    require(lock.get("overwrite") is False and lock.get("rerun") is False, "FACTOR_ONE_SHOT_POLICY_DRIFT", "factor execution must forbid overwrite/rerun")
    required_environment = lock.get("required_environment")
    require(isinstance(required_environment, dict), "FACTOR_EXECUTION_ENVIRONMENT_MISSING", "factor execution environment is missing")
    for key, expected in required_environment.items():
        require(os.environ.get(key) == str(expected), "FACTOR_EXECUTION_ENVIRONMENT_DRIFT", "factor execution environment differs from lock", key=key)
    actual_argv = [
        Path(sys.argv[0]).resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "--execution-lock",
        lock_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
    ]
    require(lock.get("argv") == actual_argv, "FACTOR_EXECUTION_ARGV_DRIFT", "factor execution argv differs from lock", actual=actual_argv)
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDING_PATHS), "FACTOR_EXECUTION_BINDINGS_MISSING", "factor execution binding cardinality drift")
    verified: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        require(isinstance(binding, dict) and set(binding) == {"role", "path", "bytes", "sha256"}, "FACTOR_EXECUTION_BINDING_INVALID", "factor binding fields drift")
        role, relative = str(binding["role"]), str(binding["path"])
        require(role not in verified and EXPECTED_BINDING_PATHS.get(role) == relative, "FACTOR_EXECUTION_BINDING_INVALID", "factor binding role/path drift", role=role)
        path = _repo_path(relative)
        require(path.is_file() and path.stat().st_size == binding["bytes"] and materializer.sha256_file(path) == binding["sha256"], "FACTOR_EXECUTION_BINDING_HASH_DRIFT", "factor bound file differs from lock", role=role)
        verified[role] = dict(binding)
    require(set(verified) == set(EXPECTED_BINDING_PATHS), "FACTOR_EXECUTION_BINDING_INVALID", "factor binding role set drift")
    assets = lock.get("depthart_assets")
    expected_asset_keys = {"source_root", "source_git_commit", "checkpoint_path", "checkpoint_bytes", "checkpoint_sha256", "model_id"}
    require(isinstance(assets, dict) and set(assets) == expected_asset_keys, "DEPTHART_ASSET_LOCK_INVALID", "DepthART asset lock fields drift")
    source_root, checkpoint = Path(assets["source_root"]).resolve(), Path(assets["checkpoint_path"]).resolve()
    require(source_root.is_dir() and checkpoint.is_file(), "DEPTHART_ASSET_MISSING", "DepthART source/checkpoint is missing")
    require(assets["source_git_commit"] == "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c" and assets["model_id"] == adapter.BASELINE_MODEL_ID, "DEPTHART_ASSET_LOCK_INVALID", "DepthART source/model identity drift")
    require(checkpoint.stat().st_size == assets["checkpoint_bytes"] and materializer.sha256_file(checkpoint) == assets["checkpoint_sha256"] == adapter.BASELINE_CHECKPOINT_SHA256, "DEPTHART_CHECKPOINT_DRIFT", "DepthART checkpoint differs from lock")
    lock["_verified_bindings"] = verified
    lock["_depthart_source"] = source_root
    lock["_depthart_checkpoint"] = checkpoint
    return lock


def _verify_r3_admission(bindings: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _load_json(_repo_path(bindings["R3_RESULT"]["path"]))
    completion = _load_json(_repo_path(bindings["R3_COMPLETION"]["path"]))
    require(
        result.get("schema") == materializer.TRUTH_RESULT_SCHEMA
        and result.get("model_outputs_absent") is True
        and result.get("depthart_inference_count") == 0
        and result.get("factorial_execution_count") == 0,
        "R3_TERMINAL_INVALID",
        "factor execution requires a valid model-free R3 terminal",
    )
    if result.get("passed") is True:
        require(
            result.get("scientific_status") == "TRUTH_ONLY_ADMISSION_PASS"
            and result.get("terminal") == "TARO_O0R_TRUTH_ONLY_ADMISSION_PASS",
            "R3_TERMINAL_INVALID",
            "R3 PASS identity drift",
        )
    else:
        require(
            result.get("passed") is False
            and result.get("scientific_status") == "NOT_EVALUABLE"
            and result.get("terminal") == "TARO_O0R_NOT_EVALUABLE_SOURCE_TRUTH_OR_INTERFACE",
            "R3_TERMINAL_INVALID",
            "R3 non-PASS terminal is not the exact retained NOT_EVALUABLE outcome",
        )
    require(
        completion.get("passed") is result["passed"]
        and completion.get("terminal") == result["terminal"]
        and completion.get("one_shot_consumed") is True,
        "R3_COMPLETION_INVALID",
        "R3 completion receipt does not match its consumed terminal",
    )
    return result, completion


def _verify_r3_manifest(truth_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    require(manifest.get("schema") == materializer.MATERIALIZER_MANIFEST_SCHEMA and isinstance(manifest.get("files"), dict), "R3_MANIFEST_INVALID", "R3 manifest schema/files drift")
    files = manifest["files"]
    require(manifest.get("file_count_before_manifest") == len(files), "R3_MANIFEST_INVALID", "R3 manifest file count drift")
    observed_bytes = 0
    for relative, receipt in files.items():
        require(isinstance(receipt, dict) and receipt.get("path") == relative, "R3_MANIFEST_INVALID", "R3 manifest file receipt drift", path=relative)
        path = materializer.safe_join(truth_root, relative)
        require(path.is_file() and path.stat().st_size == receipt.get("bytes") and materializer.sha256_file(path) == receipt.get("sha256"), "R3_MANIFEST_FILE_DRIFT", "R3 evidence file differs from manifest", path=relative)
        observed_bytes += path.stat().st_size
    actual = {
        path.relative_to(truth_root).as_posix()
        for path in truth_root.rglob("*")
        if path.is_file() and path.resolve() != manifest_path.resolve()
    }
    require(actual == set(files) and observed_bytes == manifest.get("bytes_before_manifest"), "R3_MANIFEST_COVERAGE_DRIFT", "R3 manifest does not exactly cover its evidence root")
    return manifest


def _prepared_parents(frame_plan: list[dict[str, Any]], bindings: Mapping[str, Mapping[str, Any]], source_root: Path) -> list[Any]:
    preflight = _load_json(_repo_path(bindings["TRUTH_PREFLIGHT"]["path"]))
    plan_rows = materializer.expanded_asset_plan(preflight)
    downloads = _load_json_gzip(_repo_path(bindings["R3_DOWNLOAD_RECEIPTS"]["path"]))
    require(len(plan_rows) == len(downloads), "R3_DOWNLOAD_RECEIPT_CARDINALITY", "source plan/download receipt cardinality drift")
    for row, receipt in zip(plan_rows, downloads, strict=True):
        relative = row.get("relative_path")
        require(isinstance(relative, str) and bool(relative), "R3_SOURCE_PATH_INVALID", "source plan contains an invalid relative path")
        materializer.verify_bound_container(materializer.safe_join(source_root, relative), receipt)
    prepared = truth_runner._prepare_downloaded_parents(preflight, plan_rows, downloads, source_root=source_root)
    recomputed = [
        {
            "parent": parent.parent,
            "frame_plan": parent.frame_plan,
            "materialized_bytes": parent.materialized_bytes,
            "container_receipts": parent.container_receipts,
        }
        for parent in prepared
    ]
    require(adapter.canonical_sha256(recomputed) == adapter.canonical_sha256(frame_plan), "R3_FRAME_PLAN_RECOMPUTE_MISMATCH", "source cannot reproduce the bound R3 exact frame plan")
    return prepared


def _decode(prepared: Any, token: str) -> dict[str, Any]:
    return truth_runner._decode_prepared_frame(prepared, token, materializer.decode_source_frame)


def _failure_code(error: Exception) -> str:
    return str(getattr(error, "code", type(error).__name__))


def _emit_progress(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True), flush=True)


def _build_descriptive_factor_canary_frame(
    decoded_source_frame: Mapping[str, Any],
    verified_truth: Mapping[str, Any],
    candidate_depth_m: np.ndarray,
    candidate_output_receipt: Mapping[str, Any],
    uncertainty_model: Any,
) -> dict[str, Any]:
    """Build nine threshold-free factor records, retaining extractor failures."""

    source = adapter._validate_base_receipt(dict(verified_truth["source_frame_receipt"]))
    queries = list(verified_truth["query_receipts"])
    truth_frames = list(verified_truth["truth_factor_frames"])
    lookups = list(verified_truth["uncertainty_lookups"])
    require(len(queries) == len(truth_frames) == len(lookups) == 9, "FACTOR_CANARY_QUERY_CARDINALITY", "descriptive canary requires nine aligned query inputs")
    matrix = np.asarray(source["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
    faro = np.asarray(decoded_source_frame["highres_faro_depth_mm"])
    candidate = np.asarray(candidate_depth_m)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (query, truth_frame, lookup) in enumerate(zip(queries, truth_frames, lookups, strict=True)):
        require(query["query_id"] == truth_frame["query_id"] == lookup["query_id"], "FACTOR_CANARY_QUERY_ALIGNMENT", "descriptive canary query inputs are misaligned", query_index=index)
        try:
            candidate_frame = adapter.build_candidate_query_factor_frame(
                candidate,
                matrix,
                source["gravity_up_camera_xyz"],
                source,
                query,
                truth_frame["base_geometry"],
                uncertainty_model,
                dict(candidate_output_receipt),
                confidence_value=lookup["confidence_value"],
                range_m=lookup["range_m"],
            )
        except adapter.AdapterError as error:
            failures.append(
                {
                    "query_index": index,
                    "query_id": query["query_id"],
                    "error_code": error.code,
                    "candidate_output_receipt_sha256": candidate_output_receipt["content_sha256"],
                }
            )
            continue
        records.append(
            build_factor_canary_record(
                source["parent_id"],
                truth_frame,
                candidate_frame,
                faro,
                candidate,
            )
        )
    require(len(records) + len(failures) == 9, "FACTOR_CANARY_OUTPUT_CARDINALITY", "descriptive canary did not account for all nine queries")
    payload = {
        "schema": "blindassist.taro.o0r.descriptive_factor_canary_frame.v1",
        "parent_id": source["parent_id"],
        "physical_frame_id": source["physical_frame_id"],
        "source_frame_receipt_sha256": source["content_sha256"],
        "candidate_output_receipt_sha256": candidate_output_receipt["content_sha256"],
        "truth_commitment_record_sha256": verified_truth["compact_truth_record_sha256"],
        "query_attempt_count": 9,
        "record_count": len(records),
        "failure_count": len(failures),
        "records": records,
        "candidate_extractor_failures": failures,
        "threshold_or_pass_fail_decision_applied": False,
        "dense_factor_arrays_persisted": False,
    }
    payload["content_sha256"] = adapter.canonical_sha256(payload)
    return payload


def execute_factor_headroom(
    lock_path: Path,
    *,
    model_loader: Callable[..., tuple[Any, dict[str, Any]]] = load_official_depthart,
    candidate_phase_fn: Callable[..., dict[str, Any]] = run_candidate_phase,
) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    bindings = lock["_verified_bindings"]
    r3_result, _ = _verify_r3_admission(bindings)
    locked_r3 = lock.get("r3_truth_admission")
    require(
        isinstance(locked_r3, dict)
        and locked_r3.get("terminal") == r3_result["terminal"]
        and locked_r3.get("formal_headroom_authorized") is r3_result["passed"]
        and locked_r3.get("descriptive_partial_factor_canary_authorized") is True
        and locked_r3.get("result_sha256") == bindings["R3_RESULT"]["sha256"]
        and locked_r3.get("completion_sha256") == bindings["R3_COMPLETION"]["sha256"]
        and locked_r3.get("manifest_sha256") == bindings["R3_MANIFEST"]["sha256"],
        "R3_LOCK_ADMISSION_MISMATCH",
        "execution lock R3 authority differs from its bound terminal",
    )
    source_root = _repo_path(lock["roots"]["SOURCE"])
    truth_root = _repo_path(lock["roots"]["TRUTH_EVIDENCE"])
    factor_root = _repo_path(lock["roots"]["FACTOR_EVIDENCE"])
    require(source_root.is_dir() and truth_root.is_dir() and not factor_root.exists(), "FACTOR_ROOT_PREFLIGHT_INVALID", "source/truth roots must exist and factor root must be absent")
    frame_plan = _load_json_gzip(_repo_path(bindings["R3_EXACT_FRAME_PLAN"]["path"]))
    # Source-only preparation validates every bound container and reproduces the
    # exact frame plan before model load or irreversible factor-root creation.
    prepared = _prepared_parents(frame_plan, bindings, source_root)
    started = time.monotonic()
    process = psutil.Process()
    writer = FactorEvidenceWriter(factor_root, lock["resource_budget"]["maximum_evidence_bytes"])

    def guard() -> None:
        require(time.monotonic() - started <= lock["resource_budget"]["wall_seconds"], "FACTOR_EXECUTION_TIMEOUT", "factor execution exceeded its wall-time budget")
        require(process.memory_info().rss <= lock["resource_budget"]["peak_rss_bytes"], "FACTOR_EXECUTION_RSS_EXCEEDED", "factor execution exceeded its RSS budget")

    try:
        model, runtime_identity = model_loader(lock["_depthart_source"], lock["_depthart_checkpoint"], device="cuda", seed=0)
        guard()
        writer.activate(
            {
                "schema": "blindassist.taro.o0r.factor_headroom_execution_start.v1",
                "execution_lock_sha256": materializer.sha256_file(lock_path),
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "argv": lock["argv"],
                "verified_binding_sha256s": {role: row["sha256"] for role, row in sorted(bindings.items())},
                "candidate_truth_firewall": "ALL_EXACT_CANDIDATES_SEALED_BEFORE_PER_FRAME_TRUTH_READ",
                "one_shot_consumed_on_factor_root_creation": True,
            }
        )
        completion = candidate_phase_fn(
            frame_plan,
            source_root,
            writer=writer,
            model=model,
            runtime_identity=runtime_identity,
            guard_fn=guard,
            progress_fn=_emit_progress,
        )
        guard()
        try:
            import torch

            cuda_peak = int(torch.cuda.max_memory_allocated())
            del model
            torch.cuda.empty_cache()
        except Exception:
            cuda_peak = 0
        require(cuda_peak <= lock["resource_budget"]["maximum_cuda_allocated_bytes"], "FACTOR_EXECUTION_CUDA_BUDGET_EXCEEDED", "DepthART inference exceeded its CUDA allocation budget", observed=cuda_peak)

        _verify_r3_manifest(truth_root, _repo_path(bindings["R3_MANIFEST"]["path"]))
        guard()
        model_package = _load_json_gzip(_repo_path(bindings["R3_UNCERTAINTY_ARTIFACT"]["path"]))
        hydrated_artifact = materializer.hydrate_content_addressed_artifact(model_package, lambda relative: materializer.safe_join(truth_root, relative).read_bytes())
        persisted_model_receipt = _load_json(_repo_path(bindings["R3_UNCERTAINTY_RECEIPT"]["path"]))
        audit_model = load_factory_bound_uncertainty_model(
            hydrated_artifact,
            persisted_model_receipt,
            expected_artifact_canonical_sha256=model_package["artifact_canonical_sha256"],
            expected_model_sha256=persisted_model_receipt["content_sha256"],
        )
        require(audit_model.content_sha256 == persisted_model_receipt["content_sha256"], "UNCERTAINTY_AUDIT_MODEL_MISMATCH", "hydrated uncertainty audit model hash drift")
        del audit_model
        uncertainty_model = refit_and_verify_uncertainty_model(
            prepared,
            candidate_phase_completion=completion,
            persisted_receipt=persisted_model_receipt,
            persisted_artifact_canonical_sha256=model_package["artifact_canonical_sha256"],
            decode_fn=_decode,
        )
        writer.write_json(
            "uncertainty-refit-verification.json",
            {
                "schema": "blindassist.taro.o0r.uncertainty_refit_verification.v1",
                "model_sha256": uncertainty_model.content_sha256,
                "fit_frame_count": len(uncertainty_model.source_receipt_sha256s),
                "persisted_artifact_canonical_sha256": model_package["artifact_canonical_sha256"],
                "pre_rounding_model_refit_required_for_exact_factor_recomputation": True,
                "candidate_phase_completion_sha256": completion["content_sha256"],
            },
        )

        formal_headroom_authorized = r3_result["passed"] is True
        statistics_rows: list[dict[str, Any]] = []
        canary_records: list[dict[str, Any]] = []
        canary_failures: list[dict[str, Any]] = []
        evaluated_frames = 0
        canary_frames = 0
        canary_query_attempts = 0
        skipped_incomplete_truth = 0
        missing_truth_packages = 0
        evaluated_parent_counts: dict[str, int] = {}
        canary_parent_counts: dict[str, int] = {}
        for prepared_parent in (item for item in prepared if item.parent["role"] == "O0R_EVAL_CANDIDATE"):
            parent_id, video_id = prepared_parent.parent["visit_id"], prepared_parent.parent["video_id"]
            for token in prepared_parent.frame_plan["exact_timestamp_tokens"]:
                guard()
                truth_path = truth_root / "truth-frames" / parent_id / video_id / f"{token}.json.gz"
                if not truth_path.is_file():
                    missing_truth_packages += 1
                    continue
                truth_package = _load_json_gzip(truth_path)
                compact_truth = materializer.hydrate_content_addressed_artifact(truth_package, lambda relative: materializer.safe_join(truth_root, relative).read_bytes())
                compact_truth = materializer.validate_eval_truth_commitment_record(compact_truth)
                truth_complete = compact_truth["query_bundle"]["complete_factor_query_truth"] is True
                if not truth_complete:
                    skipped_incomplete_truth += 1
                decoded = _decode(prepared_parent, token)
                verified_truth = recompute_committed_truth(decoded, uncertainty_model, compact_truth)
                sealed_candidate = load_sealed_candidate_frame(factor_root, parent_id, video_id, token)
                inference = validate_depthart_inference_receipt(sealed_candidate["candidate_frame_record"]["inference_receipt"])
                candidate_depth, candidate_output_receipt = bind_sealed_candidate_to_source(
                    inference_receipt=inference,
                    native_depth_m=sealed_candidate["native_depth_m"],
                    source_frame_receipt=decoded["source_frame_receipt"],
                )
                canary_frame = _build_descriptive_factor_canary_frame(
                    decoded,
                    verified_truth,
                    candidate_depth,
                    candidate_output_receipt,
                    uncertainty_model,
                )
                writer.write_json_gzip(f"factor-canary/{parent_id}/{video_id}/{token}.json.gz", canary_frame)
                canary_records.extend(canary_frame["records"])
                canary_failures.extend(
                    {
                        "parent_id": parent_id,
                        "video_id": video_id,
                        "timestamp_token": token,
                        "physical_frame_id": canary_frame["physical_frame_id"],
                        **failure,
                    }
                    for failure in canary_frame["candidate_extractor_failures"]
                )
                canary_frames += 1
                canary_query_attempts += canary_frame["query_attempt_count"]
                canary_parent_counts[parent_id] = canary_parent_counts.get(parent_id, 0) + 1
                _emit_progress(
                    {
                        "phase": "DESCRIPTIVE_FACTOR_CANARY",
                        "canary_frames": canary_frames,
                        "canary_records": len(canary_records),
                        "canary_failures": len(canary_failures),
                        "parent_id": parent_id,
                        "physical_frame_id": decoded["source_frame_receipt"]["physical_frame_id"],
                    }
                )
                if formal_headroom_authorized and truth_complete:
                    evaluated = evaluate_admitted_frame(decoded, verified_truth, candidate_depth, candidate_output_receipt, uncertainty_model)
                    writer.write_json_gzip(f"factor-evaluations/{parent_id}/{video_id}/{token}.json.gz", evaluated["compact_evaluation"])
                    statistics_rows.extend(evaluated["statistics_rows"])
                    evaluated_frames += 1
                    evaluated_parent_counts[parent_id] = evaluated_parent_counts.get(parent_id, 0) + 1

        require(canary_frames > 0 and canary_query_attempts == len(canary_records) + len(canary_failures), "FACTOR_CANARY_INCOMPLETE", "descriptive factor canary emitted no frames or failed to account for every query")
        require(bool(canary_records), "FACTOR_CANARY_NO_RECORDS", "descriptive factor canary produced no validated factor record")
        canary_summary = summarize_factor_canary(canary_records)
        writer.write_json_gzip("descriptive-factor-canary-records.json.gz", canary_records)
        writer.write_json_gzip("descriptive-factor-canary-failures.json.gz", canary_failures)
        writer.write_json("descriptive-factor-canary-summary.json", canary_summary)

        primary: dict[str, Any] | None = None
        diagnostics: dict[str, Any] | None = None
        formal_headroom_passed: bool | None = None
        if formal_headroom_authorized:
            require(evaluated_frames > 0 and len(statistics_rows) == evaluated_frames * 9 * len(adapter.ARMS) * len(adapter.ORACLE_MODES), "FACTOR_EVALUATION_INCOMPLETE", "formal factor evaluation emitted no frames or wrong row cardinality")
            writer.write_json_gzip("statistics-rows.json.gz", statistics_rows)
            primary = evaluate_factor_headroom(
                statistics_rows,
                critical_strata=CRITICAL_STRATA,
                structurally_not_applicable_strata=STRUCTURALLY_NOT_APPLICABLE_STRATA,
            )
            diagnostics = evaluate_factor_diagnostics(statistics_rows)
            writer.write_json("primary-statistics.json", primary)
            writer.write_json("factor-diagnostics-holm.json", diagnostics)
            formal_headroom_passed = bool(primary["gates"]["passed"])
        else:
            require(evaluated_frames == 0 and not statistics_rows, "FORMAL_FACTOR_EVALUATION_UNAUTHORIZED", "formal factorial results were emitted after a retained R3 NOT_EVALUABLE terminal")

        elapsed = time.monotonic() - started
        passed = formal_headroom_passed is True
        if formal_headroom_authorized:
            terminal = "TARO_O0R_FACTOR_HEADROOM_PASS" if passed else "TARO_O0R_NO_ADMISSIBLE_FACTOR_HEADROOM"
            scientific_status = "FACTOR_HEADROOM_PASS" if passed else "NO_ADMISSIBLE_HEADROOM"
        else:
            terminal = "TARO_O0R_PARTIAL_FACTOR_CANARY_COMPLETE"
            scientific_status = "POST_HOC_DESCRIPTIVE_PARTIAL_FACTOR_CANARY_ONLY"
        result = {
            "schema": RESULT_SCHEMA,
            "terminal": terminal,
            "passed": passed,
            "execution_valid": True,
            "scientific_status": scientific_status,
            "formal_headroom_authorized": formal_headroom_authorized,
            "formal_headroom_passed": formal_headroom_passed,
            "descriptive_partial_factor_canary_complete": True,
            "primary_comparison": primary["comparison"] if primary is not None else None,
            "primary_gates": primary["gates"] if primary is not None else None,
            "primary_failure_codes": primary["failure_codes"] if primary is not None else ["R3_TRUTH_ADMISSION_NOT_EVALUABLE_FORMAL_HEADROOM_WITHHELD"],
            "evaluated_frames": evaluated_frames,
            "evaluated_parent_counts": dict(sorted(evaluated_parent_counts.items())),
            "statistics_rows": len(statistics_rows),
            "descriptive_canary_frames": canary_frames,
            "descriptive_canary_parent_counts": dict(sorted(canary_parent_counts.items())),
            "descriptive_canary_query_attempts": canary_query_attempts,
            "descriptive_canary_record_count": len(canary_records),
            "descriptive_canary_failure_count": len(canary_failures),
            "descriptive_canary_summary_sha256": canary_summary["content_sha256"],
            "missing_truth_packages": missing_truth_packages,
            "skipped_incomplete_truth_frames": skipped_incomplete_truth,
            "candidate_inference_count": completion["candidate_frame_count"],
            "factorial_reduction_count": len(statistics_rows),
            "uncertainty_model_sha256": uncertainty_model.content_sha256,
            "resource_receipt": {
                "elapsed_seconds": elapsed,
                "peak_rss_bytes_observed_at_completion": process.memory_info().rss,
                "peak_cuda_allocated_bytes": cuda_peak,
                "evidence_bytes_before_result": writer.bytes_written,
            },
            "claim_ceiling": "Exact frozen landscape-only ARKitScenes WILD_LAB DepthART factor-headroom only; no portrait, wearable, active-observation, device, product, deployment or safety claim.",
            "descriptive_canary_claim_ceiling": FACTOR_CANARY_CLAIM_CEILING,
        }
        writer.write_json("result.json", result)
        writer.write_json(
            "completion-receipt.json",
            {
                "schema": "blindassist.taro.o0r.factor_headroom_completion.v1",
                "terminal": result["terminal"],
                "passed": passed,
                "execution_valid": True,
                "formal_headroom_authorized": formal_headroom_authorized,
                "formal_headroom_passed": formal_headroom_passed,
                "descriptive_partial_factor_canary_complete": True,
                "one_shot_consumed": True,
                "elapsed_seconds": elapsed,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "evidence_bytes_before_completion": writer.bytes_written,
            },
        )
        writer.write_json(
            "manifest.json",
            {
                "schema": MANIFEST_SCHEMA,
                "files": {key: value for key, value in sorted(writer.file_receipts.items())},
                "file_count_before_manifest": len(writer.file_receipts),
                "bytes_before_manifest": writer.bytes_written,
                "factor_root_consumed": True,
            },
        )
        return result
    except Exception as error:
        if not factor_root.exists():
            raise
        failure = {
            "schema": RESULT_SCHEMA,
            "terminal": "TARO_O0R_FACTOR_EXECUTION_INVALID",
            "passed": False,
            "execution_valid": False,
            "scientific_status": "EXECUTION_INVALID_NO_SCIENTIFIC_INTERPRETATION",
            "failure_code": _failure_code(error),
            "message": str(error),
            "one_shot_consumed": True,
        }
        try:
            writer.write_json("failure.json", failure)
            writer.write_json(
                "manifest.json",
                {
                    "schema": MANIFEST_SCHEMA,
                    "files": {key: value for key, value in sorted(writer.file_receipts.items())},
                    "file_count_before_manifest": len(writer.file_receipts),
                    "bytes_before_manifest": writer.bytes_written,
                    "factor_root_consumed": True,
                },
            )
        except Exception:
            pass
        return failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute_factor_headroom(args.execution_lock.resolve())
    except Exception as error:
        print(json.dumps({"status": "EXECUTION_NOT_STARTED", "error_code": _failure_code(error), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "execution_valid": result.get("execution_valid", False)}, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("execution_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
