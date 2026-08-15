#!/usr/bin/env python3
"""Run SATOM-R0 on a synthetic canary or a frozen real-data manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import Frame, Intrinsics, evaluate_frames, make_synthetic_frames


MANIFEST_SCHEMA = "blindassist.satom_r0.dataset_manifest.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_manifest(path: Path) -> tuple[list[Frame], dict[str, Any], str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("SATOM dataset manifest schema mismatch")
    provenance = manifest.get("prior_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("prior_provenance is required")
    if provenance.get("family") != "DepthART" or provenance.get("frozen") is not True:
        raise ValueError("real-data evaluation requires an explicitly frozen DepthART prior")
    if provenance.get("truth_derived") is not False:
        raise ValueError("DepthART prior must explicitly declare truth_derived=false")
    evidence_role = str(manifest.get("evidence_role", ""))
    allowed_roles = {"DEVELOPMENT", "CONSUMED_DEVELOPMENT", "BURNED_DEVELOPMENT"}
    if evidence_role not in allowed_roles:
        raise ValueError(f"SATOM-R0 E0 requires an explicit Development role, got: {evidence_role}")
    parent_rows = manifest.get("parents")
    if not isinstance(parent_rows, list) or not parent_rows:
        raise ValueError("manifest requires parents")
    frames: list[Frame] = []
    for parent in parent_rows:
        parent_id = str(parent["parent_id"])
        bundle_path = (path.parent / str(parent["bundle"])).resolve()
        if _sha256(bundle_path) != str(parent["sha256"]).upper():
            raise ValueError(f"bundle hash mismatch: {parent_id}")
        with np.load(bundle_path, allow_pickle=False) as bundle:
            required = {
                "timestamp_s",
                "truth_depth_m",
                "prior_depth_m",
                "intrinsics",
                "world_from_camera",
                "candidate_camera_height_m",
                "truth_camera_height_m",
                "gravity_down_camera",
            }
            missing = required.difference(bundle.files)
            if missing:
                raise ValueError(f"{parent_id}: missing arrays {sorted(missing)}")
            count = len(bundle["timestamp_s"])
            if any(len(bundle[key]) != count for key in required):
                raise ValueError(f"{parent_id}: inconsistent frame counts")
            confidence = bundle["prior_confidence"] if "prior_confidence" in bundle.files else None
            for index in range(count):
                fx, fy, cx, cy = (float(value) for value in bundle["intrinsics"][index])
                frames.append(
                    Frame(
                        parent_id=parent_id,
                        frame_index=index,
                        timestamp_s=float(bundle["timestamp_s"][index]),
                        truth_depth_m=np.asarray(bundle["truth_depth_m"][index], dtype=np.float32),
                        prior_depth_m=np.asarray(bundle["prior_depth_m"][index], dtype=np.float32),
                        prior_confidence=(None if confidence is None else np.asarray(confidence[index], dtype=np.float32)),
                        intrinsics=Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
                        world_from_camera=np.asarray(bundle["world_from_camera"][index], dtype=np.float64),
                        camera_height_m=float(bundle["candidate_camera_height_m"][index]),
                        truth_camera_height_m=float(bundle["truth_camera_height_m"][index]),
                        gravity_down_camera=np.asarray(bundle["gravity_down_camera"][index], dtype=np.float64),
                    )
                )
    return frames, provenance, evidence_role


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic-canary", action="store_true")
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-parents", type=int, default=4)
    parser.add_argument("--synthetic-frames", type=int, default=24)
    args = parser.parse_args()
    if args.synthetic_canary:
        frames = make_synthetic_frames(args.synthetic_parents, args.synthetic_frames)
        provenance = {
            "family": "SYNTHETIC_DEPTHART_LIKE_PRIOR",
            "frozen": True,
            "truth_derived": True,
            "scientific_depthart_evidence": False,
        }
        evidence_role = "SYNTHETIC_MECHANICS_CANARY"
    else:
        frames, provenance, evidence_role = load_manifest(args.manifest)
    result = evaluate_frames(
        frames,
        evidence_role=evidence_role,
        prior_provenance=provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        name: {
            "parent_macro": row["parent_macro"],
            "worst_parent": row["worst_parent"],
        }
        for name, row in result["arms"].items()
    }
    print(json.dumps({"status": result["status"], "arms": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
