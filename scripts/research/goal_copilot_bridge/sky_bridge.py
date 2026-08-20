"""Freeze, import, and independently validate BA <-> optimizer bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from evaluator import (
    CandidateContractError,
    EvaluationInfrastructureError,
    evaluate_candidate,
    load_candidate,
)

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]
PROTOCOL_ID = "GOAL-COPILOT-1"
TASK_KIND = "blindassist.search_task_bundle"
CANDIDATE_KIND = "blindassist.candidate_bundle"
TASK_SOURCE_MAP = {
    "README.md": "bundle_README.md",
    "initial_policy.py": "initial_policy.py",
    "protocol.json": "protocol.json",
    "public_scenarios/scenarios.json": "public_scenarios/scenarios.json",
    "task_api.py": "task_api.py",
}
TASK_PAYLOADS = tuple(TASK_SOURCE_MAP)
CANDIDATE_PAYLOADS = (
    "candidate/policy.py",
    "candidate_manifest.json",
    "provenance.json",
    "search_metrics.json",
)


class BundleError(ValueError):
    """A bundle is malformed, modified, out of scope, or bound to another task."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_checksums(directory: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256(directory / Path(name)) for name in sorted(names)}


def content_id(checksums: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json(checksums)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _file_members(directory: Path) -> set[str]:
    return {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }


def _verify_exact_members(directory: Path, allowed: set[str]) -> None:
    actual = _file_members(directory)
    if actual != allowed:
        raise BundleError(f"bundle members must be exactly {sorted(allowed)}; got {sorted(actual)}")


def _publish_directory(staging: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(staging)
        return destination
    os.replace(staging, destination)
    return destination


def export_task(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="goal-copilot-export-", dir=output_root))
    try:
        for exported_name, source_name in TASK_SOURCE_MAP.items():
            destination = staging / Path(exported_name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(MODULE_DIR / Path(source_name), destination)
        checksums = payload_checksums(staging, TASK_PAYLOADS)
        bundle_id = content_id(checksums)
        protocol = json.loads((MODULE_DIR / "protocol.json").read_text(encoding="utf-8"))
        public = json.loads(
            (MODULE_DIR / "public_scenarios/scenarios.json").read_text(encoding="utf-8")
        )
        write_json(staging / "checksums.json", checksums)
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "kind": TASK_KIND,
                "protocol_id": PROTOCOL_ID,
                "source_repository": "violetljj/blind-assist",
                "source_commit": git_value("rev-parse", "HEAD"),
                "bundle_digest": bundle_id,
                "created_at": git_value("show", "-s", "--format=%cI", "HEAD"),
                "candidate_surface": "initial_policy.py:CANDIDATE_FUNCTIONS",
                "evaluator_version": sha256(MODULE_DIR / "evaluator.py"),
                "scenario_version": public["scenario_version"],
                "search_authority": "PROPOSAL_ONLY",
                "acceptance_authority": "BLINDASSIST_ONLY",
                "task_families": protocol["task_families"],
                "payload_files": sorted(TASK_PAYLOADS),
                "sealed_material_exported": False,
            },
        )
        destination = output_root / PROTOCOL_ID / bundle_id
        published = _publish_directory(staging, destination)
        verify_task_bundle(published)
        return published
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_task_bundle(directory: Path) -> dict[str, Any]:
    _verify_exact_members(directory, {*TASK_PAYLOADS, "checksums.json", "manifest.json"})
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != TASK_KIND or manifest.get("protocol_id") != PROTOCOL_ID:
        raise BundleError("wrong SearchTaskBundle kind or protocol")
    if manifest.get("sealed_material_exported") is not False:
        raise BundleError("sealed evaluator material must not be exported")
    recorded = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    actual = payload_checksums(directory, TASK_PAYLOADS)
    if recorded != actual:
        raise BundleError("SearchTaskBundle payload checksum mismatch")
    identity = content_id(actual)
    if manifest.get("bundle_digest") != identity or directory.name != identity:
        raise BundleError("SearchTaskBundle content identity mismatch")
    return manifest


def verify_candidate_bundle(directory: Path) -> dict[str, Any]:
    _verify_exact_members(directory, {*CANDIDATE_PAYLOADS, "checksums.json"})
    manifest = json.loads((directory / "candidate_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != CANDIDATE_KIND or manifest.get("protocol_id") != PROTOCOL_ID:
        raise BundleError("wrong CandidateBundle kind or protocol")
    recorded = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
    actual = payload_checksums(directory, CANDIDATE_PAYLOADS)
    if recorded != actual:
        raise BundleError("CandidateBundle payload checksum mismatch")
    identity = sha256(directory / "candidate/policy.py")
    if manifest.get("candidate_id") != identity or directory.name != identity:
        raise BundleError("CandidateBundle content identity mismatch")
    return manifest


def import_candidate(candidate: Path, task_bundle: Path, output_root: Path) -> Path:
    task_manifest = verify_task_bundle(task_bundle)
    candidate_manifest = verify_candidate_bundle(candidate)
    provenance = json.loads((candidate / "provenance.json").read_text(encoding="utf-8"))
    expected_source = task_manifest["bundle_digest"]
    if provenance.get("source_search_task_bundle_digest") != expected_source:
        raise BundleError("candidate is bound to a different SearchTaskBundle")
    if candidate_manifest.get("source_search_task_bundle_digest") != expected_source:
        raise BundleError("candidate manifest source digest mismatch")
    if candidate_manifest.get("candidate_files") != ["candidate/policy.py"]:
        raise BundleError("candidate surface exceeds the policy allowlist")
    load_candidate(candidate / "candidate/policy.py")

    destination = output_root / PROTOCOL_ID / expected_source / candidate.name
    if destination.exists():
        verify_candidate_bundle(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidate, destination)
    verify_candidate_bundle(destination)
    return destination


def validate_candidate(candidate: Path, task_bundle: Path, output_root: Path) -> Path:
    task_manifest = verify_task_bundle(task_bundle)
    candidate_manifest = verify_candidate_bundle(candidate)
    source_bundle = task_manifest["bundle_digest"]
    if candidate_manifest.get("source_search_task_bundle_digest") != source_bundle:
        raise BundleError("candidate and SearchTaskBundle do not match")
    try:
        evaluation = evaluate_candidate(candidate / "candidate/policy.py")
    except EvaluationInfrastructureError as exc:
        evaluation = {
            "assessment": "NOT_EVALUABLE",
            "reason": str(exc),
            "hard_gate_pass": False,
            "metrics": None,
            "outcomes": [],
            "claim_ceiling": "no_scientific_verdict",
        }

    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": evaluation["assessment"],
        "reason": evaluation["reason"],
        "mode": "BLINDASSIST_INDEPENDENT_MOCK_VALIDATION",
        "source_search_task_bundle_digest": source_bundle,
        "candidate_id": candidate_manifest["candidate_id"],
        "sky_search_metrics_authority": "PROVENANCE_ONLY_NOT_ACCEPTANCE",
        "evaluator_sha256": sha256(MODULE_DIR / "evaluator.py"),
        "sealed_scenarios_sha256": sha256(MODULE_DIR / "sealed_scenarios.json"),
        "claim_ceiling": "bridge_mechanics_only_no_model_or_scientific_result",
        "evaluation": evaluation,
    }
    destination = (
        output_root
        / PROTOCOL_ID
        / source_bundle
        / candidate_manifest["candidate_id"]
        / "assessment.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(receipt)
    if destination.exists() and destination.read_bytes() != encoded:
        raise BundleError("existing independent assessment differs")
    destination.write_bytes(encoded)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--protocol", default=PROTOCOL_ID, choices=[PROTOCOL_ID])
    export_parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "artifacts.local/sky_exports"
    )

    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--candidate", type=Path, required=True)
    import_parser.add_argument("--task-bundle", type=Path, required=True)
    import_parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "artifacts.local/sky_imports"
    )

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--candidate", type=Path, required=True)
    validate_parser.add_argument("--task-bundle", type=Path, required=True)
    validate_parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "artifacts.local/sky_validations"
    )

    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export_task(args.output_root.resolve())
        elif args.command == "import":
            result = import_candidate(
                args.candidate.resolve(), args.task_bundle.resolve(), args.output_root.resolve()
            )
        else:
            result = validate_candidate(
                args.candidate.resolve(), args.task_bundle.resolve(), args.output_root.resolve()
            )
    except (BundleError, CandidateContractError) as exc:
        print(json.dumps({"status": "IMPORT_REJECTED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
