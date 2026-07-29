"""Independent structural and numeric validator for the P4-A manipulation check.

This file intentionally does not import the producer module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FRAME_POSITIONS = (
    0, 40, 80, 120, 160, 200, 240, 280,
    320, 360, 400, 440, 480, 520, 560, 601,
)
BLOCKS = ("ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17")
MOTIONS = ("STATIC_CAMERA", "PERIODIC_6DOF_SELF_MOTION")
LAPLACIAN_RANGE = (0.35, 0.55)
RMS_MINIMUM = 0.70
GRADIENT_RANGE = (0.35, 0.55)


class InvalidIndependentManipulation(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: list[float]) -> float:
    if len(values) != 16 or not all(math.isfinite(value) for value in values):
        raise InvalidIndependentManipulation("FRAME_METRIC_CARDINALITY")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def validate(output_dir: Path) -> dict[str, Any]:
    receipt_path = output_dir / "formal_manipulation_receipt.json"
    ledger_path = output_dir / "formal_manipulation_ledger.jsonl"
    manifest_path = output_dir / "formal_main_scene_manifest.jsonl"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["evidence"]["formal_manipulation_ledger_sha256"] != _sha(ledger_path):
        raise InvalidIndependentManipulation("LEDGER_HASH")
    if receipt["evidence"]["formal_main_scene_manifest_sha256"] != _sha(manifest_path):
        raise InvalidIndependentManipulation("MANIFEST_HASH")
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scenes = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 160 or len(scenes) != 80:
        raise InvalidIndependentManipulation("CARDINALITY")
    expected_scene_ids = {
        f"{block}__MAIN_{ordinal:02d}"
        for block in BLOCKS for ordinal in range(20)
    }
    if {row["cluster_id"] for row in scenes} != expected_scene_ids:
        raise InvalidIndependentManipulation("SCENE_IDENTITIES")
    seen = set()
    recomputed = []
    for row in rows:
        key = (row["cluster_id"], row["motion"])
        if key in seen:
            raise InvalidIndependentManipulation("DUPLICATE_SEQUENCE_CHECK")
        seen.add(key)
        if row["cluster_id"] not in expected_scene_ids or row["motion"] not in MOTIONS:
            raise InvalidIndependentManipulation("SEQUENCE_IDENTITY")
        if tuple(row["frame_positions"]) != FRAME_POSITIONS:
            raise InvalidIndependentManipulation("FRAME_POSITIONS")
        frame_rows = row["frame_rows"]
        if [item["frame_index"] for item in frame_rows] != list(FRAME_POSITIONS):
            raise InvalidIndependentManipulation("FRAME_ROW_ORDER")
        lap = _median([float(item["laplacian_variance_ratio"]) for item in frame_rows])
        rms = _median([float(item["local_rms_contrast_ratio"]) for item in frame_rows])
        gradient = _median(
            [float(item["multiscale_gradient_density_ratio"]) for item in frame_rows]
        )
        expected_blur = LAPLACIAN_RANGE[0] <= lap <= LAPLACIAN_RANGE[1] and rms >= RMS_MINIMUM
        expected_low = GRADIENT_RANGE[0] <= gradient <= GRADIENT_RANGE[1]
        if row["sequence_medians"] != {
            "laplacian_variance_ratio": lap,
            "local_rms_contrast_ratio": rms,
            "multiscale_gradient_density_ratio": gradient,
        }:
            raise InvalidIndependentManipulation("SEQUENCE_MEDIANS")
        if row["blur_sequence_pass"] is not expected_blur:
            raise InvalidIndependentManipulation("BLUR_SEQUENCE_PASS")
        if row["low_texture_sequence_pass"] is not expected_low:
            raise InvalidIndependentManipulation("LOW_SEQUENCE_PASS")
        recomputed.append((row, expected_blur, expected_low))
    subgroups = []
    for block in BLOCKS:
        for motion in MOTIONS:
            selected = [
                item for item in recomputed
                if item[0]["block"] == block and item[0]["motion"] == motion
            ]
            if len(selected) != 20:
                raise InvalidIndependentManipulation("SUBGROUP_COUNT")
            blur_count = sum(item[1] for item in selected)
            low_count = sum(item[2] for item in selected)
            subgroups.append(
                {
                    "block": block,
                    "motion": motion,
                    "sequence_count": 20,
                    "blur_pass_count": blur_count,
                    "low_texture_pass_count": low_count,
                    "blur_subgroup_pass": blur_count >= 18,
                    "low_texture_subgroup_pass": low_count >= 18,
                }
            )
    if receipt["subgroups"] != subgroups:
        raise InvalidIndependentManipulation("SUBGROUP_SUMMARY")
    expected_terminal = (
        "PASS"
        if all(
            item["blur_subgroup_pass"] and item["low_texture_subgroup_pass"]
            for item in subgroups
        )
        else "INTERVENTION_NOT_EVALUABLE"
    )
    if receipt["terminal"] != expected_terminal:
        raise InvalidIndependentManipulation("TERMINAL")
    return {
        "schema": "rcle.periodic_self_motion_counterfactual.p4_manipulation_independent_validation.v1",
        "validation": "VALID",
        "terminal": expected_terminal,
        "sequence_checks": 160,
        "ledger_sha256": _sha(ledger_path),
        "receipt_sha256": _sha(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path)
    arguments = parser.parse_args()
    result = validate(arguments.output_dir.resolve())
    if arguments.receipt_out:
        path = arguments.receipt_out.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
