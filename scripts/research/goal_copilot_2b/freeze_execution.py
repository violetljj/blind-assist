"""Create the formal GC2-B run seal after all zero-model gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_2b.export_bundle import verify_bundle

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PROTOCOL = HERE / "protocol.json"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def freeze(
    output_root: Path,
    bundle: Path,
    heldout_root: Path,
    design_seal_path: Path,
    sky_root: Path,
    provider_path: Path,
    checklist_path: Path,
) -> Path:
    if output_root.exists():
        raise FileExistsError(f"formal run root already exists: {output_root}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    bundle_manifest = verify_bundle(bundle)
    heldout = json.loads((heldout_root / "heldout_manifest.json").read_text(encoding="utf-8"))
    design = json.loads(design_seal_path.read_text(encoding="utf-8"))
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    checklist = json.loads(checklist_path.read_text(encoding="utf-8"))
    ba_commit = git_head(REPO_ROOT)
    sky_commit = git_head(sky_root)
    if design["status"] != "GOAL_COPILOT_2B_PROTOCOL_DESIGN_FROZEN_SEARCH_NOT_AUTHORIZED":
        raise RuntimeError("GC2-B design seal status mismatch")
    if bundle_manifest["blindassist_commit"] != ba_commit:
        raise RuntimeError("public bundle commit drift")
    if bundle_manifest["starting_policy_sha256"] != protocol["frozen_starting_policy"]["sha256"]:
        raise RuntimeError("starting policy drift")
    if heldout["status"] != "SEALED_BA_ONLY_NOT_EXPOSED_TO_SKY":
        raise RuntimeError("held-out envelope is not sealed")
    if heldout["schedule_seeds"] != protocol["heldout_validation"]["schedule_seeds"]:
        raise RuntimeError("held-out schedule seeds drift")
    if provider.get("preflight") != "PASS" or provider.get("authenticated_via") != "ChatGPT":
        raise RuntimeError("native Codex provider is not qualified")
    required = {
        "ba_focused_tests",
        "sky_focused_tests",
        "bundle_integrity",
        "heldout_encryption_roundtrip",
        "heldout_leakage_audit",
        "provider_preflight",
        "zero_model_transport_canary",
    }
    if set(checklist.get("gates", {})) != required or not all(checklist["gates"].values()):
        raise RuntimeError("pre-model checklist incomplete")
    budget = protocol["search_budget"]
    payload = {
        "schema_version": 1,
        "protocol_id": "GOAL-COPILOT-2B",
        "status": "GOAL_COPILOT_2B_MODEL_CALLS_AUTHORIZED_NOT_STARTED",
        "source_commits": {"blindassist": ba_commit, "skydiscover": sky_commit},
        "design_seal_digest": design["design_seal_digest"],
        "protocol_sha256": sha256(PROTOCOL),
        "search_task_bundle_digest": bundle_manifest["bundle_digest"],
        "heldout_envelope_sha256": heldout["envelope_sha256"],
        "heldout_plaintext_sha256": heldout["plaintext_sha256"],
        "heldout_plaintext_present_during_search": False,
        "provider_identity_sha256": sha256(provider_path),
        "sky_config_sha256": sha256(
            sky_root / "benchmarks" / "blindassist_goal_copilot_2b" / "config.yaml"
        ),
        "replicates": budget["replicates"],
        "search_budget": {
            "generation_attempts_per_replicate": budget["generation_attempts_per_replicate"],
            "generation_attempts_total": budget["generation_attempts_total"],
            "total_token_ceiling_per_replicate": budget["total_token_ceiling_per_replicate"],
            "generation_retries": 0,
            "evaluator_retries": 0,
        },
        "provider": {
            "executable": provider["executable"],
            "version": provider["version"],
            "executable_sha256": provider["executable_sha256"],
            "authenticated_via": "ChatGPT",
        },
        "authorities": protocol["authorities"],
        "winner_selection": protocol["winner_selection"],
        "heldout_admission": protocol["heldout_admission"],
        "heldout_pass": protocol["heldout_validation"]["pass"],
        "in_doubt_semantics": budget["started_only_dispatch"],
        "resume_authorized": False,
        "claim_ceiling": "symbolic_consumed_task_noise_robust_search_signal_only",
    }
    payload["protocol_seal_digest"] = hashlib.sha256(canonical(payload)).hexdigest()
    output_root.mkdir(parents=True)
    path = output_root / "formal_protocol_seal.json"
    with path.open("xb") as stream:
        stream.write(canonical(payload))
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--design-seal", type=Path, required=True)
    parser.add_argument("--sky-root", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--checklist", type=Path, required=True)
    args = parser.parse_args()
    print(
        freeze(
            args.output_root.resolve(),
            args.bundle.resolve(),
            args.heldout_root.resolve(),
            args.design_seal.resolve(),
            args.sky_root.resolve(),
            args.provider.resolve(),
            args.checklist.resolve(),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
