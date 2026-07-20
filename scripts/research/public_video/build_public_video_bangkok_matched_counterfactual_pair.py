#!/usr/bin/env python3
"""Materialize the frozen Bangkok same-source matched contrast as train-candidate episodes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_obstacle_aware_route_width_probe as route_width


SCHEMA = "blindassist_public_video_matched_counterfactual_pair_v1"


def episode_rows(pair_id: str, parent_source_id: str, role: str, label: int,
                 start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    if label not in (0, 1) or start_ms > end_ms or start_ms % 1000 or end_ms % 1000:
        raise ValueError("invalid frozen episode interval")
    return [{
        "item_id": f"{pair_id}:{role}:{timestamp_ms}",
        "pair_id": pair_id,
        "parent_source_id": parent_source_id,
        "episode_role": role,
        "provisional_binary_label": label,
        "timestamp_ms": timestamp_ms,
    } for timestamp_ms in range(start_ms, end_ms + 1, 1000)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.output_root):
        mil.reject_independent_direction(path)
    if args.output_root.exists():
        raise ValueError(f"refusing to overwrite output root: {args.output_root}")
    contract = common.load_json(args.contract)
    inputs = contract["bound_inputs"]
    resolved: dict[str, Path] = {}
    for name, binding in inputs.items():
        path = Path(binding["path"])
        path = path if path.is_absolute() else (Path.cwd() / path).resolve()
        if common.sha256_file(path) != binding["sha256"]:
            raise ValueError(f"bound input mismatch: {name}")
        resolved[name] = path
    negative_review = lifecycle.verify_json_sidecar(resolved["negative_review"])
    positive_review = lifecycle.verify_json_sidecar(resolved["positive_review"])
    source_id = contract["parent_source_id"]
    video_sha = inputs["video"]["sha256"]
    for review in (negative_review, positive_review):
        if review["source"]["source_id"] != source_id or review["source"]["source_video_sha256"] != video_sha:
            raise ValueError("review parent source binding differs")
    pair_id = contract["pair_id"]
    specifications = contract["episodes"]
    all_rows: list[dict[str, Any]] = []
    for role, spec in specifications.items():
        rows = episode_rows(pair_id, source_id, role, int(spec["provisional_binary_label"]),
                            int(spec["start_ms"]), int(spec["end_ms"]))
        decoded = route_width.decode_at(resolved["video"], [row["timestamp_ms"] for row in rows])
        if len(decoded) != len(rows):
            raise ValueError(f"decoded frame count differs for {role}")
        role_dir = args.output_root / role
        role_dir.mkdir(parents=True, exist_ok=False)
        for row, image in zip(rows, decoded):
            timestamp = int(row["timestamp_ms"])
            frame_path = role_dir / f"{timestamp:010d}.png"
            if not cv2.imwrite(str(frame_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(f"failed to write {frame_path}")
            row["frame_path"] = str(frame_path.resolve())
            row["frame_sha256"] = common.sha256_file(frame_path)
            row["supervision"] = "hash_bound_large_model_provisional_silver"
            row["evaluation_eligible"] = False
            row["parent_source_isolation_required"] = True
            all_rows.append(row)
    manifest = args.output_root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": common.sha256_file(args.contract),
        "pair_id": pair_id,
        "parent_source_id": source_id,
        "source_video_sha256": video_sha,
        "episode_counts": {role: sum(row["episode_role"] == role for row in all_rows) for role in specifications},
        "frame_count": len(all_rows),
        "unique_frame_sha256_count": len({row["frame_sha256"] for row in all_rows}),
        "manifest_sha256": common.sha256_file(manifest),
        "pair_semantics": contract["pair_semantics"],
        "authorization": contract["authorization"],
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(manifest) + ".sha256").write_text(common.sha256_file(manifest) + "\n", encoding="ascii")
    Path(str(report_path) + ".sha256").write_text(common.sha256_file(report_path) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(json.dumps({"ok": True, "frame_count": result["frame_count"],
                      "manifest_sha256": result["manifest_sha256"]}))
