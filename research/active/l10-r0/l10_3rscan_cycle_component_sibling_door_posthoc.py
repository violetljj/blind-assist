#!/usr/bin/env python3
"""Run coherent transport on frozen same-scene sibling-door negatives."""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_cycle_component_open_set_posthoc as open_set  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cycle-component-sibling-door-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-cycle-component-sibling-door-posthoc-result-v1"


def load_json(path: Path) -> dict[str, Any]:
    return open_set.base.load_json(path)


def sha256(path: Path) -> str:
    return open_set.base.sha256(path)


def verify_sibling_absence(protocol_path: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    open_set.base.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    cohort = load_json(HERE / protocol["source"]["cohort_path"])
    episodes = {str(row["episode_id"]): row for row in cohort["episodes"]}
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    receipts: dict[str, Any] = {}
    source_rows = {str(row["path"]): row for row in protocol["absence_sources"]}

    for pair in protocol["evaluation"]["pairs"]:
        if pair["label"] != "target_absent":
            continue
        pair_id = str(pair["id"])
        reference = episodes[str(pair["reference_episode"])]
        query = episodes[str(pair["query_episode"])]
        open_set.base.require(
            reference["reference_scan_id"] == query["reference_scan_id"],
            f"NEGATIVE_NOT_SAME_SCENE:{pair_id}",
        )
        open_set.base.require(
            int(reference["target_instance_id"]) != int(query["target_instance_id"]),
            f"NEGATIVE_NOT_SIBLING_TARGET:{pair_id}",
        )
        expected = pair["absence"]
        open_set.base.require(
            str(expected["query_scan_id"]) == str(query["rescan_id"])
            and int(expected["query_frame"]) == int(query["query"]["frame"])
            and int(expected["reference_target_instance_id"])
            == int(reference["target_instance_id"]),
            f"ABSENCE_PAIR_BINDING:{pair_id}",
        )
        ply_relative = f"datasets/3rscan/{query['rescan_id']}/labels.instances.annotated.v2.ply"
        ply_path = artifact_root / ply_relative
        source = source_rows[ply_relative]
        open_set.base.require(ply_path.stat().st_size == int(source["bytes"]), f"PLY_BYTES:{pair_id}")
        open_set.base.require(sha256(ply_path) == source["sha256"], f"PLY_HASH:{pair_id}")
        target_id = int(reference["target_instance_id"])
        points = extent.ply_instance_points(ply_path, {target_id}).get(target_id)
        open_set.base.require(points is not None and len(points) > 0, f"TARGET_POINTS:{pair_id}")

        archive_path = artifact_root / f"datasets/3rscan/{query['rescan_id']}/sequence.zip"
        with zipfile.ZipFile(archive_path) as archive:
            info = pixel.parse_info(archive.read("_info.txt").decode("utf-8"))
            pose = pixel.read_pose(archive, int(query["query"]["frame"]))
        _, _, inside = pixel.project_points(
            points,
            pose,
            info["color_intrinsic"],
            int(info["color_width"]),
            int(info["color_height"]),
        )
        inside_vertices = int(np.count_nonzero(inside))
        open_set.base.require(
            inside_vertices == int(expected["projected_inside_vertices"]) == 0,
            f"TARGET_NOT_ABSENT_OUT_OF_FOV:{pair_id}:{inside_vertices}",
        )
        receipts[pair_id] = {
            "authority": "PROVIDER_REGISTERED_EXACT_TARGET_ZERO_PROJECTED_VERTICES",
            "reference_scan_id": reference["reference_scan_id"],
            "reference_target_instance_id": target_id,
            "sibling_target_instance_id": int(query["target_instance_id"]),
            "query_scan_id": query["rescan_id"],
            "query_frame": int(query["query"]["frame"]),
            "target_mesh_vertices": int(len(points)),
            "projected_inside_vertices": inside_vertices,
            "ply_path": ply_relative,
            "ply_sha256": source["sha256"],
        }
    open_set.base.require(len(receipts) > 0, "NO_SIBLING_NEGATIVES")
    return receipts


@contextmanager
def sibling_surface(receipts: dict[str, Any]):
    saved_protocol_schema = open_set.PROTOCOL_SCHEMA
    saved_result_schema = open_set.RESULT_SCHEMA
    saved_require = open_set.base.require

    def sibling_require(condition: bool, message: str) -> None:
        if not condition and message.startswith("NEGATIVE_NOT_CROSS_SCENE:"):
            pair_id = message.split(":", 1)[1]
            saved_require(pair_id in receipts, f"UNVERIFIED_SIBLING_NEGATIVE:{pair_id}")
            return
        saved_require(condition, message)

    open_set.PROTOCOL_SCHEMA = PROTOCOL_SCHEMA
    open_set.RESULT_SCHEMA = RESULT_SCHEMA
    open_set.base.require = sibling_require
    try:
        yield
    finally:
        open_set.base.require = saved_require
        open_set.RESULT_SCHEMA = saved_result_schema
        open_set.PROTOCOL_SCHEMA = saved_protocol_schema


def replay(protocol_path: Path, output_path: Path) -> None:
    receipts = verify_sibling_absence(protocol_path)
    with sibling_surface(receipts):
        open_set.replay(protocol_path, output_path)
    result = load_json(output_path)
    result["authority"] = "CONSUMED_POSTHOC_SAME_SCENE_SIBLING_DOOR_ABSENCE_DEVELOPMENT_RESULT"
    result["entrypoint"] = {"path": Path(__file__).name, "sha256": sha256(Path(__file__))}
    result["sibling_absence_receipts"] = receipts
    result["conclusion"] = (
        "L10_3RSCAN_CYCLE_COMPONENT_SIBLING_DOOR_POSTHOC_DEVELOPMENT_GATE_MET"
        if result["gate_met"]
        else "L10_3RSCAN_CYCLE_COMPONENT_SIBLING_DOOR_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
    )
    open_set.base.roma_base.predecessor.parent.write_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
