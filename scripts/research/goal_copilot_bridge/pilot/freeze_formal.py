"""Create the create-once formal protocol seal after all zero-model gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from evaluator import DEV_SCENARIOS, evaluate_scenarios
from export_bundle import verify_bundle

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
BASELINE = HERE / "initial_policy.py"
PROTOCOL = HERE / "pilot_protocol.json"
FRESH_MANIFEST = HERE / "fresh_cohort_manifest.json"
FRESH_ENVELOPE = HERE / "fresh_scenarios.enc.json"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def freeze(
    output_root: Path, bundle: Path, sky_root: Path, provider_identity_path: Path,
    verification_receipt_path: Path,
) -> Path:
    if output_root.exists():
        raise FileExistsError(f"formal root already exists: {output_root}")
    bundle_manifest = verify_bundle(bundle)
    protocol = json.loads(PROTOCOL.read_text())
    provider = json.loads(provider_identity_path.read_text())
    verification = json.loads(verification_receipt_path.read_text())
    ba_commit = git_head(REPO_ROOT)
    sky_commit = git_head(sky_root)
    if bundle_manifest["blindassist_commit"] != ba_commit:
        raise RuntimeError("SearchTaskBundle was not exported from current BA commit")
    if provider.get("preflight") != "PASS" or provider.get("authenticated_via") != "ChatGPT":
        raise RuntimeError("provider identity was not mechanically qualified")
    required_gates = {
        "bridge_tests", "pilot_tests", "candidate_isolation_tests",
        "dev_baseline_calibration", "provider_preflight", "fresh_leakage_audit",
    }
    if set(verification.get("gates", {})) != required_gates or not all(verification["gates"].values()):
        raise RuntimeError("pre-model checklist is incomplete")
    baseline = evaluate_scenarios(BASELINE, DEV_SCENARIOS)
    bm = baseline["metrics"]
    if not (
        bm["hard_gate_pass"] and bm["unsafe_guidance"] == 0
        and bm["premature_completion"] == 0 and 4 <= bm["completion_count"] <= 9
        and all(1 <= value <= 3 for value in bm["family_completion_counts"].values())
    ):
        raise RuntimeError("frozen baseline no longer passes the calibration gate")

    output_root.mkdir(parents=True)
    shutil.copy2(FRESH_MANIFEST, output_root / "fresh_cohort_manifest.json")
    shutil.copy2(provider_identity_path, output_root / "provider_identity.json")
    shutil.copy2(verification_receipt_path, output_root / "pre_model_checklist.json")
    (output_root / "scenario_manifest.json").write_bytes(canonical({
        "dev_scenarios_sha256": sha256(DEV_SCENARIOS),
        "dev_scenario_count": 12,
        "fresh_envelope_sha256": sha256(FRESH_ENVELOPE),
        "fresh_plaintext_sha256": json.loads(FRESH_MANIFEST.read_text())["plaintext_sha256"],
        "fresh_scenario_count": 6,
        "fresh_plaintext_present_during_search": False,
    }))
    (output_root / "baseline_assessment.json").write_bytes(canonical(baseline))
    (output_root / "search_budget.json").write_bytes(canonical({
        "generation_attempts_per_replicate": 16,
        "generation_attempts_total": 32,
        "replicate_count": 2,
        "provider_retries": 0,
        "evaluator_retries": 0,
        "total_token_ceiling_per_replicate": protocol["total_token_ceiling_per_replicate"],
        "in_doubt_semantics": "started_without_terminal_journal_record_consumes_attempt",
        "resume_semantics": "no_resume_interruption_yields_NOT_EVALUABLE_INCOMPLETE_FORMAL_RUN",
    }))
    sealed_payload = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "GOAL_COPILOT_1_SKY_PILOT_MODEL_CALLS_AUTHORIZED_NOT_STARTED",
        "source_commits": {"blindassist": ba_commit, "skydiscover": sky_commit},
        "search_task_bundle_digest": bundle_manifest["bundle_digest"],
        "fresh_cohort_digest": json.loads(FRESH_MANIFEST.read_text())["plaintext_sha256"],
        "provider_identity_sha256": sha256(provider_identity_path),
        "protocol_sha256": sha256(PROTOCOL),
        "sky_config_sha256": sha256(sky_root / "benchmarks/blindassist_goal_copilot_pilot/config.yaml"),
        "dev_scenarios_sha256": sha256(DEV_SCENARIOS),
        "fresh_envelope_sha256": sha256(FRESH_ENVELOPE),
        "baseline_sha256": sha256(BASELINE),
        "replicates": protocol["replicates"],
        "search_budget": {
            "generation_attempts_per_replicate": 16,
            "generation_attempts_total": 32,
            "total_token_ceiling_per_replicate": protocol["total_token_ceiling_per_replicate"],
        },
        "authorities": protocol["authorities"],
        "candidate_isolation": protocol["candidate_isolation"],
        "winner_selection": protocol["winner_selection"],
        "fresh_admission": protocol["fresh_admission"],
        "fresh_pass": protocol["fresh_pass"],
        "fresh_plaintext_present_during_search": False,
        "resume_authorized": False,
    }
    seal_digest = hashlib.sha256(canonical(sealed_payload)).hexdigest()
    sealed_payload["protocol_seal_digest"] = seal_digest
    seal_path = output_root / "formal_protocol_seal.json"
    seal_path.write_bytes(canonical(sealed_payload))
    return seal_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sky-root", type=Path, required=True)
    parser.add_argument("--provider-identity", type=Path, required=True)
    parser.add_argument("--verification-receipt", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(
        args.output_root.resolve(), args.bundle.resolve(), args.sky_root.resolve(),
        args.provider_identity.resolve(), args.verification_receipt.resolve(),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
