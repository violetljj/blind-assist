#!/usr/bin/env python3
"""Audit matched public-silver coverage separately by risk mechanism.

An overall counterfactual-pair count can hide heterogeneous tasks. This gate
requires independent matched sources for dynamic-agent approach and static
corridor narrowing separately before a shared or mechanism-routed head may be
treated as data-ready.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_public_video_silver_labels import load_json, sha256_file, validate


SCHEMA = "blindassist_public_silver_mechanism_coverage_v1"
DEFAULT_MECHANISMS = ("dynamic_agent_approach", "static_corridor_narrowing")


def reject_independent_direction(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    if "secondary-corridor-causal" in normalized:
        raise ValueError(f"independent model direction is outside this audit's scope: {path}")


def pair_id(episode: dict[str, Any]) -> str | None:
    value = episode.get("counterfactual_pair_id")
    if not isinstance(value, str) or not value.strip():
        profile = episode.get("risk_profile")
        value = profile.get("counterfactual_pair_id") if isinstance(profile, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def infer_mechanism(episode: dict[str, Any]) -> str | None:
    profile = episode.get("risk_profile")
    if not isinstance(profile, dict):
        return None
    explicit = profile.get("risk_mechanism")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    text = " ".join(
        str(profile.get(field, "")).lower()
        for field in ("primary_hazard_type", "corridor_relation")
    )
    if any(token in text for token in ("static", "furniture", "trash", "sign", "pole", "bike_rack", "step", "curb")):
        return "static_corridor_narrowing"
    if any(token in text for token in ("pedestrian", "stroller", "animal", "vehicle", "crossing", "person")):
        return "dynamic_agent_approach"
    return None


def collect(package_root: Path) -> list[dict[str, Any]]:
    reject_independent_direction(package_root)
    rows: list[dict[str, Any]] = []
    silver_paths = sorted(package_root.glob("*/silver_labels_v2.json"))
    if not silver_paths:
        raise ValueError(f"no v2 public-silver packages found under {package_root}")
    for silver_path in silver_paths:
        source_path = silver_path.parent / "source_manifest_v2.json"
        silver = load_json(silver_path)
        validate(silver, source_manifest_path=source_path)
        source_id = silver["source"]["source_id"]
        for episode in silver["episodes"]:
            verdict = episode["silver_should_alert"]
            identifier = pair_id(episode)
            if verdict == "abstain" or identifier is None:
                continue
            rows.append({
                "episode_id": episode["episode_id"],
                "source_id": source_id,
                "verdict": verdict,
                "counterfactual_pair_id": identifier,
                "mechanism": infer_mechanism(episode),
                "confidence": float(episode.get("confidence", 0.0)),
            })
    return rows


def evaluate(
    rows: list[dict[str, Any]],
    *,
    required_mechanisms: tuple[str, ...],
    minimum_pairs_per_mechanism: int,
    minimum_sources_per_mechanism: int,
    minimum_pair_confidence: float,
) -> dict[str, Any]:
    if minimum_pairs_per_mechanism < 2 or minimum_sources_per_mechanism < 2:
        raise ValueError("mechanism coverage thresholds must require at least two pairs and two sources")
    if not 0.0 <= minimum_pair_confidence <= 1.0:
        raise ValueError("minimum pair confidence must be in [0, 1]")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["counterfactual_pair_id"], []).append(row)

    matched: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for identifier, pair_rows in sorted(grouped.items()):
        if {row["verdict"] for row in pair_rows} != {"candidate_no_alert", "candidate_alert"}:
            continue
        mechanisms = sorted({row["mechanism"] for row in pair_rows if row["mechanism"]})
        if len(mechanisms) != 1:
            conflicts.append({
                "counterfactual_pair_id": identifier,
                "mechanisms": mechanisms,
                "episode_ids": sorted(row["episode_id"] for row in pair_rows),
            })
            continue
        matched.append({
            "counterfactual_pair_id": identifier,
            "mechanism": mechanisms[0],
            "source_ids": sorted({row["source_id"] for row in pair_rows}),
            "episode_ids": sorted(row["episode_id"] for row in pair_rows),
            "minimum_episode_confidence": min(row["confidence"] for row in pair_rows),
        })

    coverage: dict[str, Any] = {}
    failures: list[str] = []
    for mechanism in required_mechanisms:
        all_mechanism_pairs = [row for row in matched if row["mechanism"] == mechanism]
        mechanism_pairs = [
            row for row in all_mechanism_pairs
            if row["minimum_episode_confidence"] >= minimum_pair_confidence
        ]
        source_ids = sorted({source for row in mechanism_pairs for source in row["source_ids"]})
        pair_passed = len(mechanism_pairs) >= minimum_pairs_per_mechanism
        source_passed = len(source_ids) >= minimum_sources_per_mechanism
        coverage[mechanism] = {
            "all_matched_pair_count": len(all_mechanism_pairs),
            "matched_pair_count": len(mechanism_pairs),
            "excluded_low_confidence_pair_ids": [
                row["counterfactual_pair_id"]
                for row in all_mechanism_pairs
                if row["minimum_episode_confidence"] < minimum_pair_confidence
            ],
            "independent_source_count": len(source_ids),
            "independent_source_ids": source_ids,
            "counterfactual_pair_ids": [row["counterfactual_pair_id"] for row in mechanism_pairs],
            "minimum_pairs_passed": pair_passed,
            "minimum_sources_passed": source_passed,
        }
        if not pair_passed:
            failures.append(f"{mechanism}:insufficient_matched_pairs")
        if not source_passed:
            failures.append(f"{mechanism}:insufficient_independent_sources")
    passed = not failures and not conflicts
    if conflicts:
        failures.append("counterfactual_pair_mechanism_conflict")
    return {
        "required_mechanisms": list(required_mechanisms),
        "thresholds": {
            "minimum_matched_pairs_per_mechanism": minimum_pairs_per_mechanism,
            "minimum_independent_sources_per_mechanism": minimum_sources_per_mechanism,
            "minimum_episode_confidence_per_pair": minimum_pair_confidence,
        },
        "coverage": coverage,
        "matched_pairs": matched,
        "mechanism_conflicts": conflicts,
        "mechanism_coverage_gate": {
            "passed": passed,
            "failure_reasons": failures,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    output = args.output.resolve()
    reject_independent_direction(output)
    required_mechanisms = tuple(args.required_mechanism or DEFAULT_MECHANISMS)
    result = evaluate(
        collect(package_root),
        required_mechanisms=required_mechanisms,
        minimum_pairs_per_mechanism=args.minimum_pairs_per_mechanism,
        minimum_sources_per_mechanism=args.minimum_sources_per_mechanism,
        minimum_pair_confidence=args.minimum_pair_confidence,
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        **result,
        "isolation_contract": {
            "public_video_mainline_only": True,
            "independent_model_direction_data_used": False,
            "independent_model_direction_metrics_used_as_gate": False,
        },
        "evidence_limit": "Mechanism-stratified model-silver data sufficiency only; not human truth, calibration, blind evaluation, or production authorization.",
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(output) + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-mechanism", action="append")
    parser.add_argument("--minimum-pairs-per-mechanism", type=int, default=3)
    parser.add_argument("--minimum-sources-per-mechanism", type=int, default=3)
    parser.add_argument("--minimum-pair-confidence", type=float, default=0.65)
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["mechanism_coverage_gate"], "coverage": report["coverage"]}, ensure_ascii=False))
    return 0 if report["mechanism_coverage_gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
