#!/usr/bin/env python3
"""Audit whether provisional public-video supervision is ready for a head run.

This is a dataset gate, not a label validator or a production authorization.
It revalidates every v2 package, verifies the bound image bytes, counts
independent source groups per class, detects cross-source frame duplication,
and requires explicit positive/negative counterfactual pair IDs.  A frozen
feature probe may be supplied, but a passing probe cannot compensate for a
failed dataset gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_public_video_silver_labels import load_json, validate


SCHEMA = "blindassist_public_silver_training_readiness_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counterfactual_pair_id(episode: dict[str, Any]) -> str | None:
    value = episode.get("counterfactual_pair_id")
    if value is None:
        profile = episode.get("risk_profile")
        value = profile.get("counterfactual_pair_id") if isinstance(profile, dict) else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def collect(package_root: Path) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    abstentions: list[dict[str, str]] = []
    packages: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    frame_sources: dict[str, set[str]] = {}

    silver_paths = sorted(package_root.glob("*/silver_labels_v2.json"))
    if not silver_paths:
        raise ValueError(f"no v2 silver-label packages found under {package_root}")

    for silver_path in silver_paths:
        source_path = silver_path.parent / "source_manifest_v2.json"
        silver = load_json(silver_path)
        source = load_json(source_path)
        validation = validate(silver, source_manifest_path=source_path)
        if validation.get("training_execution_authorized") is not True:
            raise ValueError(f"package is not authorized for provisional training: {silver_path}")

        promotion = source.get("promotion")
        if not isinstance(promotion, dict) or not isinstance(promotion.get("image_root"), str):
            raise ValueError(f"source manifest has no bound image_root: {source_path}")
        image_root = Path(promotion["image_root"]).resolve()
        if not image_root.is_dir():
            raise ValueError(f"bound image_root is missing: {image_root}")

        source_id = silver["source"]["source_id"]
        source_rows = source.get("frames")
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError(f"source manifest contains no frames: {source_path}")
        by_hash: dict[str, dict[str, Any]] = {}
        for row in source_rows:
            if not isinstance(row, dict) or not isinstance(row.get("sha256"), str) or not isinstance(row.get("file_name"), str):
                raise ValueError(f"invalid frame row in {source_path}")
            expected_hash = row["sha256"]
            image_path = (image_root / row["file_name"]).resolve()
            if not image_path.is_relative_to(image_root):
                raise ValueError(f"frame escapes bound image_root: {image_path}")
            if not image_path.is_file() or sha256_file(image_path) != expected_hash:
                raise ValueError(f"frame bytes do not match bound SHA256: {image_path}")
            by_hash[expected_hash] = row
            frame_sources.setdefault(expected_hash, set()).add(source_id)

        packages.append({
            "source_id": source_id,
            "silver_manifest": str(silver_path.resolve()),
            "silver_sha256": sha256_file(silver_path),
            "source_manifest": str(source_path.resolve()),
            "source_sha256": sha256_file(source_path),
            "license": source.get("source", {}).get("license"),
        })
        for episode in silver["episodes"]:
            episode_id = episode["episode_id"]
            if episode_id in seen_episode_ids:
                raise ValueError(f"duplicate episode_id across packages: {episode_id}")
            seen_episode_ids.add(episode_id)
            verdict = episode["silver_should_alert"]
            if verdict == "abstain":
                abstentions.append({"episode_id": episode_id, "source_id": source_id})
                continue
            evidence = episode["evidence_frame_sha256"]
            if any(value not in by_hash for value in evidence):
                raise ValueError(f"episode references a frame outside its source: {episode_id}")
            episodes.append({
                "episode_id": episode_id,
                "source_id": source_id,
                "verdict": verdict,
                "counterfactual_pair_id": _counterfactual_pair_id(episode),
                "evidence_frame_count": len(evidence),
            })

    cross_source_duplicates = [
        {"frame_sha256": value, "source_ids": sorted(source_ids)}
        for value, source_ids in sorted(frame_sources.items())
        if len(source_ids) > 1
    ]
    return {
        "packages": packages,
        "episodes": episodes,
        "abstentions": abstentions,
        "cross_source_duplicate_frames": cross_source_duplicates,
    }


def evaluate(
    collected: dict[str, Any],
    *,
    minimum_sources_per_class: int,
    minimum_counterfactual_pairs: int,
    probe_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if minimum_sources_per_class < 2 or minimum_counterfactual_pairs < 1:
        raise ValueError("readiness thresholds must require at least two sources per class and one counterfactual pair")
    episodes = collected["episodes"]
    verdicts = ("candidate_no_alert", "candidate_alert")
    class_counts = {verdict: sum(row["verdict"] == verdict for row in episodes) for verdict in verdicts}
    source_ids = {
        verdict: sorted({row["source_id"] for row in episodes if row["verdict"] == verdict})
        for verdict in verdicts
    }
    pair_rows: dict[str, list[dict[str, Any]]] = {}
    for row in episodes:
        if row["counterfactual_pair_id"]:
            pair_rows.setdefault(row["counterfactual_pair_id"], []).append(row)
    matched_pairs = [
        {
            "counterfactual_pair_id": pair_id,
            "episode_ids": sorted(row["episode_id"] for row in rows),
            "source_ids": sorted({row["source_id"] for row in rows}),
        }
        for pair_id, rows in sorted(pair_rows.items())
        if {row["verdict"] for row in rows} == set(verdicts)
    ]

    enough_sources = all(len(source_ids[verdict]) >= minimum_sources_per_class for verdict in verdicts)
    enough_pairs = len(matched_pairs) >= minimum_counterfactual_pairs
    frames_disjoint = not collected["cross_source_duplicate_frames"]
    data_ready = enough_sources and enough_pairs and frames_disjoint

    probe_supplied = probe_report is not None
    probe_passed = bool(probe_report and probe_report.get("linear_separability_gate", {}).get("passed") is True)
    head_ready = data_ready and probe_supplied and probe_passed
    failures: list[str] = []
    if not enough_sources:
        failures.append("insufficient_independent_sources_per_class")
    if not enough_pairs:
        failures.append("insufficient_explicit_matched_counterfactual_pairs")
    if not frames_disjoint:
        failures.append("frame_hash_reused_across_source_ids")
    if not probe_supplied:
        failures.append("frozen_feature_probe_not_supplied")
    elif not probe_passed:
        failures.append("frozen_feature_probe_failed")

    return {
        "episode_count": len(episodes),
        "excluded_abstain_count": len(collected["abstentions"]),
        "class_counts": class_counts,
        "independent_source_counts": {key: len(value) for key, value in source_ids.items()},
        "independent_source_ids": source_ids,
        "matched_counterfactual_pairs": matched_pairs,
        "cross_source_duplicate_frames": collected["cross_source_duplicate_frames"],
        "required_evaluation_group_key": "source_id",
        "gates": {
            "minimum_independent_sources_per_class": minimum_sources_per_class,
            "minimum_matched_counterfactual_pairs": minimum_counterfactual_pairs,
            "independent_sources_per_class_passed": enough_sources,
            "matched_counterfactual_pairs_passed": enough_pairs,
            "cross_source_frame_disjointness_passed": frames_disjoint,
            "dataset_ready_for_linear_probe": data_ready,
            "frozen_feature_probe_supplied": probe_supplied,
            "frozen_feature_probe_passed": probe_passed,
            "head_short_runs_authorized": head_ready,
            "failure_reasons": failures,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    collected = collect(package_root)
    probe = load_json(args.probe_report) if args.probe_report else None
    result = evaluate(
        collected,
        minimum_sources_per_class=args.minimum_sources_per_class,
        minimum_counterfactual_pairs=args.minimum_counterfactual_pairs,
        probe_report=probe,
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "package_count": len(collected["packages"]),
        "packages": collected["packages"],
        **result,
        "probe_report": ({"path": str(args.probe_report.resolve()), "sha256": sha256_file(args.probe_report)} if args.probe_report else None),
        "evidence_limit": "GPT/VLM provisional supervision only; this gate never authorizes calibration, blind evaluation, or production replacement.",
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-sources-per-class", type=int, default=5)
    parser.add_argument("--minimum-counterfactual-pairs", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["gates"]}, ensure_ascii=False))
    return 0 if report["gates"]["head_short_runs_authorized"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
