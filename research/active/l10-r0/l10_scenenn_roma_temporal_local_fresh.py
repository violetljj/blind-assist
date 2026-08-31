#!/usr/bin/env python3
"""Fresh SceneNN confirmation of temporally local active RoMa support."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


_NVIDIA_ROOT = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "nvidia"
_DLL_DIRECTORY_HANDLES = []
if os.name == "nt" and _NVIDIA_ROOT.is_dir():
    for _dll_dir in sorted(_NVIDIA_ROOT.glob("*/bin")):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(_dll_dir)))

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_scenenn_roma_active_none as base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-scenenn-roma-temporal-local-fresh-protocol-v1"
ADMISSION_SCHEMA = "blindassist-l10-scenenn-roma-temporal-local-fresh-source-admission-v1"
COHORT_SCHEMA = "blindassist-l10-scenenn-roma-temporal-local-fresh-cohort-v1"
RECEIPT_SCHEMA = "blindassist-l10-scenenn-roma-temporal-local-fresh-rgb-receipt-v1"
RESULT_SCHEMA = "blindassist-l10-scenenn-roma-temporal-local-fresh-result-v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    base.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    prior = protocol["predecessor"]
    prior_path = HERE / prior["result_path"]
    base.require(base.sha256(prior_path) == prior["result_sha256"], "PREDECESSOR_HASH")
    base.require(load_json(prior_path)["conclusion"] == prior["required_conclusion"], "PREDECESSOR_CONCLUSION")
    model_root = ROOT / protocol["matcher"]["path"]
    base.require(base.sha256(model_root / "roma_indoor.pth") == protocol["matcher"]["roma_weights_sha256"], "ROMA_WEIGHTS")
    base.require(base.sha256(model_root / "dinov2_vitl14_pretrain.pth") == protocol["matcher"]["dinov2_weights_sha256"], "ROMA_DINOV2_WEIGHTS")
    wheel_path = ROOT / protocol["matcher"]["wheel_path"]
    base.require(base.sha256(wheel_path) == protocol["matcher"]["wheel_sha256"], "ROMA_WHEEL")
    return protocol


def temporal_local_pair(rows: list[dict[str, Any]], minimum_baseline: float, minimum_gap: int):
    eligible = [row for row in rows if row["eligible"]]
    candidates = []
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            frame_gap = abs(int(left["frame"]) - int(right["frame"]))
            if frame_gap < minimum_gap:
                continue
            left_center = np.asarray(left["camera_center_world"], dtype=np.float64)
            right_center = np.asarray(right["camera_center_world"], dtype=np.float64)
            baseline = float(np.linalg.norm(left_center - right_center))
            if baseline < minimum_baseline:
                continue
            first, second = (left, right) if int(left["frame"]) < int(right["frame"]) else (right, left)
            rank = (
                frame_gap,
                baseline,
                -min(int(first["visible_pixels"]), int(second["visible_pixels"])),
                -min(float(first["visible_to_target_raster_ratio"]), float(second["visible_to_target_raster_ratio"])),
                int(first["frame"]),
                int(second["frame"]),
            )
            candidates.append((rank, first, second, baseline))
    if not candidates:
        return None
    _, reference, query, baseline = min(candidates, key=lambda row: row[0])
    return reference, query, baseline


@contextmanager
def patched_base():
    names = (
        "PROTOCOL_SCHEMA",
        "ADMISSION_SCHEMA",
        "COHORT_SCHEMA",
        "RECEIPT_SCHEMA",
        "RESULT_SCHEMA",
        "load_protocol",
        "minimum_eligible_baseline_pair",
        "__file__",
    )
    saved = {name: getattr(base, name) for name in names}
    base.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    base.ADMISSION_SCHEMA = ADMISSION_SCHEMA
    base.COHORT_SCHEMA = COHORT_SCHEMA
    base.RECEIPT_SCHEMA = RECEIPT_SCHEMA
    base.RESULT_SCHEMA = RESULT_SCHEMA
    base.load_protocol = load_protocol
    base.minimum_eligible_baseline_pair = temporal_local_pair
    base.__file__ = __file__
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(base, name, value)


def freeze(protocol_path: Path, admission_path: Path, source_root: Path, output_path: Path) -> None:
    with patched_base():
        base.freeze(protocol_path, admission_path, source_root, output_path)
    cohort = load_json(output_path)
    cohort["authority"] = "FRESH_PRE_RGBD_TEMPORALLY_LOCAL_ROMA_NONE_CONFIRMATION_COHORT"
    cohort["selection_policy"] = "minimum qualifying frame gap, then minimum qualifying camera baseline, visibility and time tie-breaks"
    base.predecessor.parent.write_json(output_path, cohort)


def seal(cohort_path: Path, source_root: Path, extraction_root: Path, extractor_exe: Path, output_path: Path) -> None:
    with patched_base():
        base.seal(cohort_path, source_root, extraction_root, extractor_exe, output_path)
    receipt = load_json(output_path)
    receipt["authority"] = "POST_FRESH_COHORT_FREEZE_TEMPORALLY_LOCAL_SPARSE_RGB_ONLY_RECEIPT"
    base.predecessor.parent.write_json(output_path, receipt)


def replay(protocol_path: Path, cohort_path: Path, receipt_path: Path, source_root: Path, output_path: Path) -> None:
    with patched_base():
        base.replay(protocol_path, cohort_path, receipt_path, source_root, output_path)
    result = load_json(output_path)
    result["authority"] = "FRESH_SOURCE_DISJOINT_TEMPORALLY_LOCAL_ROMA_CYCLE_NONE_DEVELOPMENT_RESULT"
    result["conclusion"] = (
        "L10_SCENENN_ROMA_TEMPORAL_LOCAL_FRESH_DEVELOPMENT_GATE_MET"
        if result["gate_met"]
        else "L10_SCENENN_ROMA_TEMPORAL_LOCAL_FRESH_DEVELOPMENT_GATE_NOT_MET"
    )
    base.predecessor.parent.write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--admission", type=Path, required=True)
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--cohort", type=Path, required=True)
    seal_parser.add_argument("--source-root", type=Path, required=True)
    seal_parser.add_argument("--extraction-root", type=Path, required=True)
    seal_parser.add_argument("--extractor-exe", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, required=True)
    replay_parser.add_argument("--cohort", type=Path, required=True)
    replay_parser.add_argument("--receipt", type=Path, required=True)
    replay_parser.add_argument("--source-root", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.protocol, args.admission, args.source_root, args.output)
    elif args.action == "seal":
        seal(args.cohort, args.source_root, args.extraction_root, args.extractor_exe, args.output)
    else:
        replay(args.protocol, args.cohort, args.receipt, args.source_root, args.output)


if __name__ == "__main__":
    main()
