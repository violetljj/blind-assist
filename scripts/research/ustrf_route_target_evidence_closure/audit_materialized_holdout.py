#!/usr/bin/env python3
"""Fail-closed integrity and cross-source duplicate audit for materialized holdout RGB-D."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dhash64(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def band_keys(value: int) -> Iterable[tuple[int, int]]:
    # Five disjoint bands guarantee a shared band whenever 64-bit Hamming distance <= 4.
    shifts_and_widths = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))
    for band, (shift, width) in enumerate(shifts_and_widths):
        yield band, (value >> shift) & ((1 << width) - 1)


def verify_source(dataset_root: Path, source_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_root = dataset_root / source_id / "sequences"
    if not source_root.is_dir():
        raise RuntimeError(f"source sequence directory missing: {source_root}")
    sequence_rows: list[dict[str, Any]] = []
    rgb_rows: list[dict[str, Any]] = []
    for sequence_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        bundle_path = sequence_dir / "bundle.json"
        frames_path = sequence_dir / "frames.jsonl"
        tf_inventory_path = sequence_dir / "tf-frame-inventory.json"
        cleanup_path = sequence_dir / "raw-cleanup-receipt.json"
        if (
            not bundle_path.is_file()
            or not frames_path.is_file()
            or not tf_inventory_path.is_file()
            or not cleanup_path.is_file()
        ):
            raise RuntimeError(f"incomplete sequence evidence: {sequence_dir}")
        bundle = load_json(bundle_path)
        cleanup = load_json(cleanup_path)
        if bundle.get("candidate_outputs_executed") is not False:
            raise RuntimeError(f"candidate output leak in {bundle_path}")
        frames_sha = sha256_file(frames_path)
        if bundle.get("frames_sha256") != frames_sha:
            raise RuntimeError(f"frames hash mismatch: {frames_path}")
        tf_inventory_sha = sha256_file(tf_inventory_path)
        if bundle.get("tf_frame_inventory_sha256") != tf_inventory_sha:
            raise RuntimeError(f"TF inventory hash mismatch: {tf_inventory_path}")
        tf_inventory = load_json(tf_inventory_path)
        if tf_inventory.get("candidate_outputs_executed") is not False or not tf_inventory.get("frame_pairs"):
            raise RuntimeError(f"invalid TF inventory: {tf_inventory_path}")
        if bundle.get("tf_frame_pair_count") != len(tf_inventory["frame_pairs"]):
            raise RuntimeError(f"TF frame-pair count mismatch: {tf_inventory_path}")
        frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines() if line]
        if len(frames) != bundle.get("rgb_frame_count"):
            raise RuntimeError(f"RGB frame count mismatch: {sequence_dir}")
        seen_rgb_paths: set[Path] = set()
        seen_exact_depth_paths: set[Path] = set()
        exact_count = 0
        for row in frames:
            if row.get("source_id") != source_id or row.get("sequence_id") != sequence_dir.name:
                raise RuntimeError(f"frame provenance mismatch: {frames_path}")
            rgb_path = sequence_dir / row["rgb_path"]
            if not rgb_path.is_file() or sha256_file(rgb_path) != row.get("rgb_sha256"):
                raise RuntimeError(f"RGB hash mismatch: {rgb_path}")
            seen_rgb_paths.add(rgb_path.resolve())
            rgb_rows.append({
                "source_id": source_id,
                "sequence_id": sequence_dir.name,
                "frame_id": row["frame_id"],
                "path": rgb_path,
                "sha256": row["rgb_sha256"],
                "dhash64": dhash64(rgb_path),
            })
            if row.get("exact_aligned_depth"):
                exact_count += 1
                depth_path = sequence_dir / row["aligned_depth_path"]
                if not depth_path.is_file() or sha256_file(depth_path) != row.get("aligned_depth_sha256"):
                    raise RuntimeError(f"exact depth hash mismatch: {depth_path}")
                seen_exact_depth_paths.add(depth_path.resolve())
            elif row.get("aligned_depth_path") is not None or row.get("aligned_depth_sha256") is not None:
                raise RuntimeError(f"non-exact RGB frame carries depth authority: {frames_path}")
        all_rgb_paths = {path.resolve() for path in (sequence_dir / "rgb").glob("*.png")}
        all_depth_paths = {path.resolve() for path in (sequence_dir / "aligned_depth").glob("*.png")}
        if all_rgb_paths != seen_rgb_paths:
            raise RuntimeError(f"RGB inventory mismatch: {sequence_dir}")
        if len(all_depth_paths) != bundle.get("aligned_depth_frame_count"):
            raise RuntimeError(f"depth inventory mismatch: {sequence_dir}")
        if exact_count != bundle.get("exact_rgb_depth_frame_count"):
            raise RuntimeError(f"exact RGB-depth count mismatch: {sequence_dir}")
        # Hash every depth file, including depth timestamps without an exact RGB peer.
        depth_inventory_digest = hashlib.sha256()
        for depth_path in sorted(all_depth_paths, key=lambda path: path.as_posix()):
            depth_inventory_digest.update(depth_path.name.encode("utf-8"))
            depth_inventory_digest.update(sha256_file(depth_path).encode("ascii"))
        if cleanup.get("schema") != "blindassist_crowdbot_raw_cleanup_receipt_r1":
            raise RuntimeError(f"raw cleanup receipt schema mismatch: {cleanup_path}")
        sequence_rows.append({
            "sequence_id": sequence_dir.name,
            "bundle_sha256": sha256_file(bundle_path),
            "frames_sha256": frames_sha,
            "tf_frame_inventory_sha256": tf_inventory_sha,
            "tf_frame_pair_count": len(tf_inventory["frame_pairs"]),
            "cleanup_receipt_sha256": sha256_file(cleanup_path),
            "rgb_frame_count": len(frames),
            "aligned_depth_frame_count": len(all_depth_paths),
            "exact_rgb_depth_frame_count": exact_count,
            "unpaired_depth_frame_count": len(all_depth_paths - seen_exact_depth_paths),
            "depth_inventory_sha256": depth_inventory_digest.hexdigest(),
        })
    return {
        "source_id": source_id,
        "sequence_count": len(sequence_rows),
        "rgb_frame_count": sum(row["rgb_frame_count"] for row in sequence_rows),
        "aligned_depth_frame_count": sum(row["aligned_depth_frame_count"] for row in sequence_rows),
        "exact_rgb_depth_frame_count": sum(row["exact_rgb_depth_frame_count"] for row in sequence_rows),
        "sequences": sequence_rows,
    }, rgb_rows


def cross_source_duplicates(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    sample_limit = 100
    left_exact = defaultdict(list)
    for row in left:
        left_exact[row["sha256"]].append(row)
    exact_pair_count = 0
    exact_pair_samples = []
    for row in right:
        for match in left_exact.get(row["sha256"], []):
            exact_pair_count += 1
            if len(exact_pair_samples) < sample_limit:
                exact_pair_samples.append(
                    [match["sequence_id"], match["frame_id"], row["sequence_id"], row["frame_id"]]
                )

    bands: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(left):
        for key in band_keys(row["dhash64"]):
            bands[key].append(index)
    near_pair_count = 0
    near_pair_samples = []
    for row in right:
        candidates: set[int] = set()
        for key in band_keys(row["dhash64"]):
            candidates.update(bands.get(key, []))
        for index in candidates:
            match = left[index]
            distance = (match["dhash64"] ^ row["dhash64"]).bit_count()
            if distance <= 4:
                near_pair_count += 1
                if len(near_pair_samples) < sample_limit:
                    near_pair_samples.append({
                        "left_sequence_id": match["sequence_id"],
                        "left_frame_id": match["frame_id"],
                        "right_sequence_id": row["sequence_id"],
                        "right_frame_id": row["frame_id"],
                        "hamming_distance": distance,
                    })
    return {
        "sample_limit_per_check": sample_limit,
        "exact_rgb_sha256_pair_count": exact_pair_count,
        "exact_pair_samples": exact_pair_samples,
        "dhash64_hamming_threshold": 4,
        "perceptual_near_duplicate_pair_count": near_pair_count,
        "perceptual_near_duplicate_pair_samples": near_pair_samples,
    }


def expected_sequence_counts(
    source_ids: list[str],
    source_bindings: list[str],
    fallback: int,
) -> dict[str, int]:
    if not source_bindings:
        return {source_id: fallback for source_id in source_ids}
    result: dict[str, int] = {}
    for binding in source_bindings:
        if "=" not in binding:
            raise RuntimeError("--expected-sequences must use SOURCE_ID=COUNT")
        source_id, count_text = binding.split("=", 1)
        if source_id in result or source_id not in source_ids:
            raise RuntimeError(f"unexpected or duplicate expected-sequences source: {source_id}")
        count = int(count_text)
        if count <= 0:
            raise RuntimeError("expected sequence count must be positive")
        result[source_id] = count
    if set(result) != set(source_ids):
        raise RuntimeError("expected sequence bindings must cover both sources")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--source-id", action="append", required=True)
    parser.add_argument("--expected-sequences-per-source", type=int, default=8)
    parser.add_argument("--expected-sequences", action="append", default=[], metavar="SOURCE_ID=COUNT")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--integrity-freeze", type=Path)
    args = parser.parse_args()
    if len(args.source_id) != 2 or len(set(args.source_id)) != 2:
        raise RuntimeError("exactly two distinct source IDs are required")
    state = load_json(args.state)
    if state.get("status") != "complete" or state.get("candidate_outputs_executed") is not False:
        raise RuntimeError("materialization must be complete and candidate-blind before audit")
    expected_counts = expected_sequence_counts(
        args.source_id,
        args.expected_sequences,
        args.expected_sequences_per_source,
    )
    integrity_freeze = load_json(args.integrity_freeze) if args.integrity_freeze else None
    if integrity_freeze:
        if integrity_freeze["candidate_outputs_executed_before_freeze"] is not False:
            raise RuntimeError("integrity protocol followed candidate execution")
        if state.get("replacement_preregistration_sha256") != integrity_freeze[
            "replacement_preregistration_sha256"
        ]:
            raise RuntimeError("integrity protocol replacement binding mismatch")
        frozen_counts = {
            row["source_id"]: int(row["expected_sequence_count"])
            for row in integrity_freeze["sources"]
        }
        if expected_counts != frozen_counts:
            raise RuntimeError("integrity expected source counts differ from freeze")
        if args.output.resolve() != (Path.cwd() / integrity_freeze["planned_output"]).resolve():
            raise RuntimeError("integrity output path differs from freeze")
    summaries = []
    frames_by_source = []
    for source_id in args.source_id:
        summary, frames = verify_source(args.dataset_root, source_id)
        if summary["sequence_count"] != expected_counts[source_id]:
            raise RuntimeError(f"unexpected sequence count for {source_id}: {summary['sequence_count']}")
        summaries.append(summary)
        frames_by_source.append(frames)
    duplicates = cross_source_duplicates(frames_by_source[0], frames_by_source[1])
    payload = {
        "schema": "blindassist_crowdbot_holdout_integrity_audit_r1",
        "candidate_outputs_executed": False,
        "candidate_h2_authority": False,
        "production_authority": False,
        "admitted_source_count": 0,
        "reason_not_admitted": "integrity audit does not replace frozen all-person route-role and lifecycle truth",
        "materialization_state_sha256": sha256_file(args.state),
        "integrity_freeze_sha256": sha256_file(args.integrity_freeze) if args.integrity_freeze else None,
        "sources": summaries,
        "cross_source_duplicate_audit": duplicates,
        "cross_source_duplicate_gate_pass": (
            duplicates["exact_rgb_sha256_pair_count"] == 0
            and duplicates["perceptual_near_duplicate_pair_count"] == 0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS" if payload["cross_source_duplicate_gate_pass"] else "FAIL_CLOSED_DUPLICATES_FOUND",
        "source_count": len(summaries),
        "sequence_count": sum(row["sequence_count"] for row in summaries),
        "rgb_frame_count": sum(row["rgb_frame_count"] for row in summaries),
        "exact_duplicate_pairs": duplicates["exact_rgb_sha256_pair_count"],
        "near_duplicate_pairs": duplicates["perceptual_near_duplicate_pair_count"],
        "output_sha256": sha256_file(args.output),
    }))
    return 0 if payload["cross_source_duplicate_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
