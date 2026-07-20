#!/usr/bin/env python3
"""Plan new public SANPO boundary-coverage candidates for auxiliary-only use.

The plan intentionally consumes sparse public source-mask evidence only.  It
never creates risk/event labels, never touches a blind manifest, excludes every
session already in the canonical train/dev set, and does not download RGB.
Exact remote mask screening and later privacy/source checks remain mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import train_export_sanpo_segmentation as shared


SCHEMA = "blindassist_sanpo_boundary_aux_candidate_plan_v1"
DEFAULT_AGGREGATE = "artifacts.local/evidence/sanpo-p3-discovery-auto-20260713/aggregate.json"
DEFAULT_CANONICAL = "test-artifacts.local/datasets/sanpo-v4-real-canonical-r3-20260713"
DEFAULT_OUTPUT = "artifacts.local/evidence/sanpo-boundary-aux-candidates-20260715/plan.json"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def canonical_session_ids(dataset_root: Path) -> set[str]:
    manifest = dataset_root / "training_manifest.jsonl"
    records = shared.load_records(manifest)
    return {record.session_id.removeprefix("sanpo_real_v0:") for record in records}


def boundary_score(candidate: dict[str, Any]) -> tuple[int, int, int]:
    evidence = candidate.get("sparse_frame_evidence")
    if not isinstance(evidence, list):
        return (0, 0, 0)
    boundary_pixels: list[int] = []
    for frame in evidence:
        profiles = frame.get("profiles") if isinstance(frame, dict) else None
        if not isinstance(profiles, dict) or profiles.get("step_curb") is not True:
            continue
        best = profiles.get("best_boundary_target")
        if isinstance(best, dict) and isinstance(best.get("pixel_count"), int):
            boundary_pixels.append(int(best["pixel_count"]))
    return (len(boundary_pixels), sum(boundary_pixels), max(boundary_pixels, default=0))


def plan_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    excluded_session_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        session_id = candidate.get("session_id")
        if not isinstance(session_id, str) or not session_id or session_id in excluded_session_ids:
            continue
        if candidate.get("official_split") != "train":
            continue
        if candidate.get("camera") != "camera_chest" or candidate.get("lens") != "left":
            continue
        if candidate.get("selection_profile") != "step_curb":
            continue
        frame_count, total_pixels, max_pixels = boundary_score(candidate)
        if frame_count <= 0 or total_pixels <= 0:
            continue
        start_frame = candidate.get("recommended_start_frame")
        if not isinstance(start_frame, int) or start_frame < 0:
            continue
        eligible.append({
            "session_id": session_id,
            "camera": candidate["camera"],
            "lens": candidate["lens"],
            "start_frame": start_frame,
            "sparse_boundary_frame_count": frame_count,
            "sparse_boundary_pixel_sum": total_pixels,
            "sparse_boundary_pixel_max": max_pixels,
            "source_license": candidate.get("license"),
            "selection_reason": "highest sparse public step/curb boundary coverage among non-canonical train sessions",
        })
    ranked = sorted(
        eligible,
        key=lambda item: (-item["sparse_boundary_frame_count"], -item["sparse_boundary_pixel_sum"], item["session_id"]),
    )
    selected: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    for item in ranked:
        if item["session_id"] in seen_sessions:
            continue
        selected.append(item)
        seen_sessions.add(item["session_id"])
        if len(selected) >= limit:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--canonical-dataset-root", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    aggregate = load_json(args.aggregate)
    if aggregate.get("complete") is not True or not isinstance(aggregate.get("candidates"), list):
        raise ValueError("aggregate must be a complete discovery record with a candidates list")
    canonical_ids = canonical_session_ids(args.canonical_dataset_root)
    selected = plan_candidates(aggregate["candidates"], excluded_session_ids=canonical_ids, limit=args.limit)
    if not selected:
        raise ValueError("no non-canonical public step/curb auxiliary candidates remain")
    payload = {
        "schema": SCHEMA,
        "aggregate": str(args.aggregate.resolve()),
        "aggregate_sha256": shared.sha256_file(args.aggregate),
        "canonical_dataset_root": str(args.canonical_dataset_root.resolve()),
        "excluded_canonical_session_count": len(canonical_ids),
        "selected_count": len(selected),
        "candidates": selected,
        "source_mask_role": "auxiliary_pixel_geometry_only",
        "prohibited_uses": [
            "risk_or_event_label",
            "risk_lifecycle_target",
            "calibration_or_benchmark_truth",
            "default_model_replacement",
        ],
        "next_gate": "remote_mask_only_exact_50_frame_screen; RGB remains undisclosed and undownloaded",
        "training_execution_authorized": False,
        "production_model_replacement_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(shared.sha256_file(args.output) + "\n", encoding="ascii")
    print(json.dumps({"selected": len(selected), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
