#!/usr/bin/env python3
"""Build deterministic risk-profile/lifecycle supervision from reviewed episodes.

This is an offline data-contract adapter, not a trainer.  It refuses incomplete
or unreviewed episode collections and never authorizes a production replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from validate_sanpo_counterfactual_episodes import ContractError, _load_json, validate


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build(config: dict[str, Any], manifest: dict[str, Any], *, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation = validate(config, manifest, root=root, require_complete=True)
    if not validation["training_eligible"]:
        raise ContractError("complete episode validation did not produce eligible supervision")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ContractError("manifest episodes must be a list")
    targets: list[dict[str, Any]] = []
    for episode in sorted(
        episodes,
        key=lambda row: (str(row["session_id"]), str(row["scene_id"]), str(row["matched_pair_id"]), str(row["pair_role"])),
    ):
        targets.append({
            "format": "blindassist_risk_lifecycle_target_v1",
            "episode_id": episode["episode_id"],
            "session_id": episode["session_id"],
            "scene_id": episode["scene_id"],
            "matched_pair_id": episode["matched_pair_id"],
            "pair_role": episode["pair_role"],
            "risk_event_id": episode["risk_event_id"],
            "duration_ms": episode["duration_ms"],
            "expected_should_alert": episode["expected_should_alert"],
            "risk_profile": episode["risk_profile"],
            "lifecycle_intervals_ms": episode["lifecycle_intervals_ms"],
            "event_anchor_ms": {
                "first_visible_ms": episode["first_visible_ms"],
                "alertable_start_ms": episode["alertable_start_ms"],
                "passed_or_cleared_ms": episode["passed_or_cleared_ms"],
            },
            "pixel_supervision_role": "auxiliary_only",
            "source_receipt_id": episode["source_receipt_id"],
        })
    target_hash = canonical_sha256(targets)
    report = {
        "format": "blindassist_risk_lifecycle_target_report_v1",
        "episode_count": len(targets),
        "target_sha256": target_hash,
        "episode_manifest_sha256": canonical_sha256(manifest),
        "validated_collection": validation,
        "training_execution_authorized": False,
        "production_model_replacement_authorized": False,
        "pixel_supervision_role": "auxiliary_only",
    }
    return targets, report


def write_outputs(targets: list[dict[str, Any]], report: dict[str, Any], *, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    targets_path = output_dir / "risk_lifecycle_targets.jsonl"
    report_path = output_dir / "risk_lifecycle_target_report.json"
    if targets_path.exists() or report_path.exists():
        raise ContractError("refusing to overwrite existing target outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in targets), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = _load_json(args.config)
        manifest = _load_json(args.manifest)
        targets, report = build(config, manifest, root=args.manifest.parent)
        write_outputs(targets, report, output_dir=args.output_dir)
    except (ContractError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir.resolve()), "target_sha256": report["target_sha256"], "training_execution_authorized": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
