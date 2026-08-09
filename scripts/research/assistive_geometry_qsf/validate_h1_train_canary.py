#!/usr/bin/env python3
"""Validate the frozen AG-QSF H1-only TRAIN canary lock and runtime resources."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import psutil

from scripts.research.assistive_geometry.download_b0_arkitscenes_assets import sha256_file
from scripts.research.assistive_geometry_qsf.h1_survival import (
    HAZARD_BIN_EDGES_M,
    OCCUPANCY_HORIZONS_M,
    h1_parameter_budget,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_RELATIVE = PurePosixPath(
    "docs/research/assistive-geometry-qsf/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_H1_TRAIN_CANARY_PROTOCOL_2026-08-09.json"
)
ROUTE_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_QSF_R0"
PROTOCOL_SCHEMA = "blindassist.assistive_geometry_qsf.h1_train_canary_protocol.v1"
EXPECTED_FIT_PARENTS = (
    "41159448",
    "42445086",
    "42445790",
    "42897547",
    "42898024",
    "42898242",
    "44358617",
    "47333249",
    "47334163",
    "47430327",
    "47204445",
    "47331226",
)
EXPECTED_EVAL_PARENTS = ("47430531", "47431114", "47895508", "47334948")
FOREIGN_GPU_MARKERS = (
    "train_b1_a0_formal.py",
    "train_b1_additive_arm.py",
    "assistive-geometry-b1-a0-formal-train",
)


class ValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def _repo_path(value: str) -> Path:
    logical = PurePosixPath(value)
    _require(
        not logical.is_absolute() and ".." not in logical.parts and "\\" not in value,
        f"repository path must be a clean POSIX relative path: {value}",
    )
    logical_path = REPO_ROOT / Path(*logical.parts)
    if logical.parts and logical.parts[0].casefold() == "artifacts.local":
        # artifacts.local is the repository's governed F-backed junction.  Keep
        # logical ownership under the checkout while allowing the configured
        # storage target to resolve outside E:\linnan\linnan.
        return logical_path
    path = logical_path.resolve()
    _require(path.is_relative_to(REPO_ROOT), f"repository path escaped checkout: {value}")
    return path


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def validate_protocol(
    protocol: dict[str, Any],
    *,
    verify_inputs: bool = True,
) -> dict[str, Any]:
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "H1 canary schema drift")
    _require(protocol.get("route_id") == ROUTE_ID, "H1 canary route drift")
    _require(protocol.get("stage") == "CANARY", "H1 canary stage drift")
    _require(protocol.get("profile") == "CANARY_LITE", "H1 canary profile drift")
    _require(protocol.get("candidate") == "H1_ONLY", "H1-only candidate drift")
    _require(
        protocol.get("status") == "H1_IMPLEMENTED_TRAIN_CANARY_LOCKED_NOT_RUN",
        "H1 canary status drift",
    )
    authority = protocol.get("execution_authority", {})
    _require(authority.get("real_train_canary") is True, "real TRAIN canary not authorized")
    _require(authority.get("h2_implementation_or_materialization") is False, "H2 authority leaked")
    _require(authority.get("h1_plus_h2_training") is False, "combined training authority leaked")
    _require(authority.get("development_outcome_access") is False, "Development firewall drift")
    _require(authority.get("confirmation_outcome_access") is False, "Confirmation firewall drift")
    _require(authority.get("default_app_change") is False, "default App authority leaked")

    representation = protocol.get("representation", {})
    _require(
        tuple(float(value) for value in representation.get("hazard_bin_edges_m", []))
        == HAZARD_BIN_EDGES_M,
        "hazard grid drift",
    )
    _require(
        tuple(float(value) for value in representation.get("occupancy_horizons_m", []))
        == OCCUPANCY_HORIZONS_M,
        "occupancy horizon drift",
    )
    _require(representation.get("robust_contact_quantile") == 0.02, "q-contact quantile drift")
    _require(
        representation.get("unknown_is_negative") is False,
        "UNKNOWN must not be negative",
    )
    _require(
        representation.get("partial_support_is_right_censor") is False,
        "partial support must not become right censor",
    )
    budget = h1_parameter_budget()
    _require(budget["exact_match"] is True, "H1/direct parameter budget no longer matches")
    _require(
        protocol.get("parameter_budget") == budget,
        "tracked parameter budget does not match implementation",
    )

    roster = protocol.get("roster", {})
    fit = tuple(str(value) for value in roster.get("fit_parent_video_ids", []))
    evaluate = tuple(str(value) for value in roster.get("eval_parent_video_ids", []))
    _require(fit == EXPECTED_FIT_PARENTS, "fit parent roster drift")
    _require(evaluate == EXPECTED_EVAL_PARENTS, "eval parent roster drift")
    _require(set(fit).isdisjoint(evaluate), "fit/eval parent overlap")
    _require(roster.get("frames_per_parent") == 64, "canary frame count drift")
    _require(roster.get("selection") == "SOURCE_ORDER_EVENLY_SPACED", "frame selection drift")

    training = protocol.get("training", {})
    expected_training = {
        "encoder": "FROZEN_DEPTHART_S_INDOOR",
        "trainable_modules": ["band_mlp", "hazard_out", "confidence_out"],
        "seed": 1702943,
        "epochs": 50,
        "batch_size": 64,
        "optimizer": "ADAMW",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "active_losses": ["survival_nll", "false_clear_extra", "confidence_bce"],
    }
    _require(training == expected_training, "H1 training lock drift")

    gates = protocol.get("gates", {})
    expected_gate_keys = {
        "fit_survival_nll_relative_improvement_min",
        "eval_survival_nll_relative_improvement_min",
        "eval_false_clear_rate_increase_max",
        "eval_clearance_mae_increase_max_m",
        "horizon_monotonicity_violations_max",
        "known_coverage_delta_max",
    }
    _require(set(gates) == expected_gate_keys, "H1 gate keyset drift")
    _require(gates["fit_survival_nll_relative_improvement_min"] == 0.10, "fit NLL gate drift")
    _require(gates["eval_survival_nll_relative_improvement_min"] == 0.02, "eval NLL gate drift")
    _require(gates["eval_false_clear_rate_increase_max"] == 0.01, "false-clear gate drift")
    _require(gates["eval_clearance_mae_increase_max_m"] == 0.02, "clearance gate drift")
    _require(gates["horizon_monotonicity_violations_max"] == 0, "monotonicity gate drift")
    _require(gates["known_coverage_delta_max"] == 0.0, "coverage gate drift")

    scheduling = protocol.get("resource_scheduling", {})
    _require(scheduling.get("foreign_formal_train_priority") is True, "foreign priority drift")
    _require(scheduling.get("requires_foreign_gpu_idle") is True, "GPU isolation drift")
    _require(scheduling.get("minimum_free_vram_mib") == 5000, "free VRAM gate drift")
    _require(scheduling.get("maximum_projected_wall_seconds") == 900, "wall-time gate drift")

    embedded_manifest = protocol.get("shared_resource_manifest", {})
    _require(
        embedded_manifest
        == {
            "schema": "blindassist.assistive_geometry_qsf.embedded_shared_resource_manifest.v1",
            "embedded_in_protocol": True,
            "resource_keys": [
                "target_manifest",
                "depthart_source",
                "initialization_checkpoint",
            ],
            "exact_input_keyset_required": True,
            "development_or_confirmation_resources_present": False,
        },
        "embedded shared-resource manifest drift",
    )

    implementation = protocol.get("implementation", {})
    required_paths = {
        "scripts/research/assistive_geometry/assistive_geometry_model.py",
        "scripts/research/assistive_geometry/assistive_geometry_training.py",
        "scripts/research/assistive_geometry/depthart_training_scan.py",
        "scripts/research/assistive_geometry/download_b0_arkitscenes_assets.py",
        "scripts/research/assistive_geometry/smoke_b1_a0_train_execution.py",
        "scripts/research/assistive_geometry/train_b1_a0_formal.py",
        "scripts/research/hftf/deployment/depthart/export_depthart_camera_external.py",
        "scripts/research/assistive_geometry_qsf/h1_survival.py",
        "scripts/research/assistive_geometry_qsf/run_h1_train_canary.py",
        "scripts/research/assistive_geometry_qsf/test_h1_survival.py",
        "scripts/research/assistive_geometry_qsf/test_validate_h1_train_canary.py",
        "scripts/research/assistive_geometry_qsf/validate_h1_train_canary.py",
    }
    _require(set(implementation) == required_paths, "implementation path set drift")
    for logical, expected_sha in implementation.items():
        _require(re.fullmatch(r"[0-9A-F]{64}", str(expected_sha)) is not None, f"bad SHA: {logical}")
        _require(sha256_file(_repo_path(logical)) == expected_sha, f"implementation SHA drift: {logical}")

    inputs = protocol.get("inputs", {})
    _require(
        set(inputs) == {"target_manifest", "depthart_source", "initialization_checkpoint"},
        "shared input keyset drift or protected input leak",
    )
    target = inputs.get("target_manifest", {})
    source = inputs.get("depthart_source", {})
    checkpoint = inputs.get("initialization_checkpoint", {})
    for resource in (target, source, checkpoint):
        for key in (
            "kind",
            "producer_route",
            "access",
            "immutable",
            "provenance",
            "license_scope",
            "data_role",
            "outcome_access",
            "selection_influence",
            "claim_use",
        ):
            _require(key in resource, f"shared input disclosure missing: {key}")
        _require(resource["access"] == "READ_ONLY", "shared input must be READ_ONLY")
        _require(resource["immutable"] is True, "shared input must be immutable")
        _require(resource["selection_influence"] == "NONE", "selection influence drift")
    _require(target.get("data_role") == "TRAIN", "target data role drift")
    _require(target.get("outcome_access") == "CONTENT_INSPECTED", "target content access drift")
    _require(target.get("claim_use") == "TRAIN_TARGET_INPUT_ONLY", "target claim-use drift")
    _require(source.get("data_role") == "NOT_APPLICABLE", "source data role drift")
    _require(source.get("outcome_access") == "NONE", "source outcome access drift")
    _require(checkpoint.get("data_role") == "NOT_APPLICABLE", "checkpoint data role drift")
    _require(checkpoint.get("outcome_access") == "NONE", "checkpoint outcome access drift")
    _require(
        target.get("kind") == "DERIVED_TRAIN_CACHE"
        and target.get("producer_route") == "BLINDASSIST_ASSISTIVE_GEOMETRY",
        "TRAIN target producer/kind drift",
    )
    _require(
        source.get("kind") == "CODE_CONTRACT"
        and source.get("producer_route") == "DEPTHART_R1",
        "DepthART source producer/kind drift",
    )
    _require(
        checkpoint.get("kind") == "INITIALIZATION"
        and checkpoint.get("producer_route") == "DEPTHART_R1",
        "initialization producer/kind drift",
    )

    outputs = protocol.get("outputs", {})
    expected_outputs = {
        "pilot_parent": "artifacts.local/evidence/assistive-geometry-qsf/h1-train-canary-r0",
        "run_parent": "artifacts.local/evidence/assistive-geometry-qsf/h1-train-canary-r0",
        "model_parent": "artifacts.local/models/assistive-geometry-qsf/h1-train-canary-r0",
        "work_parent": "artifacts.local/work/assistive-geometry-qsf/h1-train-canary-r0",
    }
    _require(outputs == expected_outputs, "H1 output namespace drift")

    if verify_inputs:
        target_path = _repo_path(str(target.get("path")))
        _require(target_path.is_file(), "TRAIN target manifest missing")
        _require(sha256_file(target_path) == target.get("sha256"), "TRAIN manifest SHA drift")
        target_manifest = _load_json(target_path)
        _require(
            target_manifest.get("schema")
            == "blindassist_assistive_geometry_b1_train_target_manifest_v1",
            "TRAIN manifest schema drift",
        )
        parents = tuple(str(video["video_id"]) for video in target_manifest.get("videos", []))
        _require(parents == fit + evaluate, "TRAIN manifest parent order/identity drift")
        _require(
            target_manifest.get("development_or_confirmation_content_opened") is False,
            "shared manifest crossed Development/Confirmation firewall",
        )
        _require(target_manifest.get("model_outputs_read") is False, "shared target outcome drift")

        source_root = Path(str(source.get("path"))).resolve()
        _require(source_root.is_dir(), "DepthART source root missing")
        _require(_git_head(source_root) == source.get("git_commit"), "DepthART source commit drift")
        release_manifest = source_root / str(source.get("release_manifest"))
        _require(release_manifest.is_file(), "DepthART release manifest missing")
        _require(
            sha256_file(release_manifest) == source.get("release_manifest_sha256"),
            "DepthART release manifest SHA drift",
        )
        checkpoint_path = Path(str(checkpoint.get("path"))).resolve()
        _require(checkpoint_path.is_file(), "DepthART initialization checkpoint missing")
        _require(
            sha256_file(checkpoint_path) == checkpoint.get("sha256"),
            "DepthART initialization SHA drift",
        )

    return {
        "route_id": ROUTE_ID,
        "status": protocol["status"],
        "candidate": "H1_ONLY",
        "fit_parent_count": len(fit),
        "eval_parent_count": len(evaluate),
        "parameter_budget": budget,
        "terminal": "H1_TRAIN_CANARY_PROTOCOL_VALID",
    }


def find_foreign_gpu_processes(
    processes: Iterable[tuple[int, str]],
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for pid, command in processes:
        lowered = command.casefold()
        if any(marker.casefold() in lowered for marker in FOREIGN_GPU_MARKERS):
            found.append({"pid": int(pid), "role": "FOREIGN_FORMAL_TRAIN"})
    return found


def _live_processes() -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command = " ".join(process.info.get("cmdline") or ())
            rows.append((int(process.info["pid"]), command))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return rows


def _free_vram_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    _require(len(values) == 1, "H1 canary requires exactly one NVIDIA GPU")
    return values[0]


def runtime_preflight(protocol: dict[str, Any]) -> dict[str, Any]:
    foreign = find_foreign_gpu_processes(_live_processes())
    free_vram = _free_vram_mib()
    required = int(protocol["resource_scheduling"]["minimum_free_vram_mib"])
    ready = not foreign and free_vram >= required
    return {
        "status": "READY" if ready else "DEFERRED",
        "foreign_formal_train_processes": foreign,
        "free_vram_mib": free_vram,
        "minimum_free_vram_mib": required,
        "terminal": (
            "H1_CANARY_RUNTIME_READY"
            if ready
            else "H1_CANARY_DEFERRED_RESOURCE_ISOLATION"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / PROTOCOL_RELATIVE)
    parser.add_argument("--runtime-preflight", action="store_true")
    args = parser.parse_args()
    protocol = _load_json(args.protocol.resolve())
    report: dict[str, Any] = {"protocol": validate_protocol(protocol)}
    code = 0
    if args.runtime_preflight:
        report["runtime"] = runtime_preflight(protocol)
        if report["runtime"]["status"] != "READY":
            code = 2
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
