"""Validate the governed AG-DUE SANPO initial static-prescreen result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import validate_due_r0 as r0
from . import validate_due_sanpo_manifest_lock as manifest_lock


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = REPO_ROOT / "docs/research/assistive-geometry-data-upgrade/BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_INITIAL_STATIC_PRESCREEN_RESULT_2026-08-10.json"
LOCK_PATH = manifest_lock.PROTOCOL_PATH

TERMINAL = "AG_DUE_R0_SANPO_INITIAL_STATIC_PRESCREEN_COMPLETE_BOTH_PARTIAL"
SUCCESSOR = "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_SOURCE_SPECIFIC_INTEGRITY_AND_CAPABILITY_AUDIT_PROTOCOL_LOCK"
RESULT_ID = "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0_SANPO_INITIAL_STATIC_PRESCREEN_RESULT_2026-08-10"


class ResultError(ValueError):
    """Raised when the governed result is not an exact replay of the lock."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_result_bytes(result: dict[str, Any]) -> bytes:
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return rendered.encode("utf-8")


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    gap_results = result["gap_results"]
    return {
        "manifest_id": result["manifest_id"],
        "source_id": result["source_id"],
        "decision": result["decision"],
        "hard_rejection_reasons": result["hard_rejection_reasons"],
        "partial_gaps": sorted(name for name, value in gap_results.items() if value["partial"]),
        "screening_match_gaps": sorted(name for name, value in gap_results.items() if value["screening_match"]),
        "upgradeable_gaps": sorted(name for name, value in gap_results.items() if value["upgradeable"]),
        "source_data_support_established": result["source_data_support_established"],
        "supported_for_protocol_lock": result["supported_for_protocol_lock"],
        "next_action": result["next_action"],
        "execution_authorized": result["execution_authorized"],
        "canonical_full_result_bytes": len(canonical_result_bytes(result)),
        "canonical_full_result_sha256": hashlib.sha256(canonical_result_bytes(result)).hexdigest().upper(),
    }


def _expected_results(repo_root: Path, lock_protocol: dict[str, Any]) -> dict[str, Any]:
    bindings = lock_protocol["bindings"]
    due_protocol = load_json(repo_root / bindings["due_r0_protocol"]["path"])
    gap_contract = r0.validate_protocol(due_protocol)
    expected: dict[str, Any] = {}
    for source_name, binding_name in (
        ("SANPO_REAL", "sanpo_real_manifest"),
        ("SANPO_SYNTHETIC", "sanpo_synthetic_manifest"),
    ):
        manifest = load_json(repo_root / bindings[binding_name]["path"])
        expected[source_name] = summarize(r0.evaluate_source_manifest(manifest, gap_contract))
    return expected


def validate_result(result: dict[str, Any], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    require(set(result) == {
        "schema",
        "result_id",
        "route_id",
        "status",
        "terminal",
        "research_mode",
        "execution_profile",
        "scientific_question",
        "bindings",
        "source_results",
        "observations",
        "execution_disclosure",
        "authority",
        "unique_successor",
        "claim_ceiling",
    }, "result field set drift")
    require(result.get("schema") == "blindassist.assistive_geometry_due.sanpo_static_prescreen_result.v1", "result schema drift")
    require(result.get("result_id") == RESULT_ID, "result identity drift")
    require(result.get("route_id") == "BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R0", "route identity drift")
    require(result.get("status") == "SANPO_INITIAL_STATIC_PRESCREEN_EXECUTED", "result status drift")
    require(result.get("terminal") == TERMINAL, "terminal drift")
    require(result.get("research_mode") == "REVERSIBLE_EXPLORATION", "research mode drift")
    require(result.get("execution_profile") == "CANARY_LITE", "execution profile drift")
    require(isinstance(result.get("scientific_question"), str) and result["scientific_question"].strip(), "scientific question missing")

    lock_protocol = load_json(repo_root / LOCK_PATH.relative_to(REPO_ROOT))
    manifest_lock.validate_protocol(lock_protocol, repo_root)
    bindings = result.get("bindings")
    require(isinstance(bindings, dict), "result bindings missing")
    expected_binding_names = {"manifest_lock", "sanpo_real_manifest", "sanpo_synthetic_manifest", "r0_validator"}
    require(set(bindings) == expected_binding_names, "result binding set drift")
    expected_paths = {
        "manifest_lock": str(LOCK_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sanpo_real_manifest": lock_protocol["bindings"]["sanpo_real_manifest"]["path"],
        "sanpo_synthetic_manifest": lock_protocol["bindings"]["sanpo_synthetic_manifest"]["path"],
        "r0_validator": "scripts/research/assistive_geometry_data_upgrade/validate_due_r0.py",
    }
    for name, expected_path in expected_paths.items():
        binding = bindings[name]
        require(binding.get("path") == expected_path, f"result binding path drift: {name}")
        path = repo_root / expected_path
        require(path.is_file(), f"result binding missing: {name}")
        require(binding.get("sha256") == sha256_file(path), f"result binding SHA drift: {name}")

    expected_results = _expected_results(repo_root, lock_protocol)
    require(result.get("source_results") == expected_results, "source result replay mismatch")
    require({item["decision"] for item in expected_results.values()} == {"PARTIAL"}, "unexpected prescreen decision")
    for source_name, item in expected_results.items():
        require(item["hard_rejection_reasons"] == [], f"unexpected hard rejection: {source_name}")
        require(item["source_data_support_established"] is False, f"source support upgraded: {source_name}")
        require(item["supported_for_protocol_lock"] is False, f"protocol lock support upgraded: {source_name}")
        require(item["execution_authorized"] is False, f"execution authority upgraded: {source_name}")

    expected_execution = {
        "formal_source_prescreen_executed": True,
        "manifest_count": 2,
        "metadata_or_network_refresh": False,
        "payload_download_or_open": False,
        "source_specific_integrity_or_count_audit": False,
        "teacher_or_pseudo_label_generation": False,
        "data_materialization": False,
        "model_or_training": False,
        "development_or_confirmation": False,
        "android_or_default_app": False,
    }
    require(result.get("execution_disclosure") == expected_execution, "execution disclosure drift")
    expected_authority = {
        "source_data_support_established": False,
        "dca_pass_established": False,
        "payload_or_source_specific_audit_authorized": False,
        "derivation_or_teacher_authorized": False,
        "materialization_or_training_authorized": False,
        "development_or_confirmation_authorized": False,
        "default_app_product_or_safety_authorized": False,
    }
    require(result.get("authority") == expected_authority, "authority ceiling drift")
    expected_observations = [
        "Both locked manifests pass the static governance and protected-identity checks and therefore have no hard rejection reason.",
        "Both are PARTIAL because relevant R2 F1 and temporal candidates exist only as published source inventory with zero observed frames, CHARACTERIZED_NOT_VALIDATED quality and UNKNOWN camera/upright basis.",
        "Neither source matches any complete gap screen; QSF right-censor, corridor and FCI truth bundles remain absent from both manifests.",
        "SANPO Synthetic is the narrower next audit candidate because its locked manifest names source-native metric-depth and panoptic candidates, whereas SANPO Real does not yet characterize oracle depth as source-native truth. This is only a source-audit priority, not source support or model selection.",
        "SANPO Real privacy verification covers source-level terms and policy only; residual payload PII has not been audited or cleared.",
    ]
    require(result.get("observations") == expected_observations, "observations drift")
    require(result.get("unique_successor") == SUCCESSOR, "successor drift")
    require(result.get("claim_ceiling") == "A deterministic metadata-only prescreen classifies the two locked SANPO manifests as PARTIAL with no hard rejection. No source data support, DCA pass, payload audit, derivation, Teacher, materialization, training, Development, Confirmation, default-App, product or safety authority is established.", "claim ceiling drift")
    return {
        "result_id": result["result_id"],
        "status": "VALID",
        "terminal": TERMINAL,
        "decisions": {name: item["decision"] for name, item in expected_results.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    try:
        require(args.result.resolve() == RESULT_PATH.resolve(), "custom result path is not authorized")
        validated = validate_result(load_json(args.result))
    except (ResultError, manifest_lock.LockError, r0.ContractError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(validated, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
