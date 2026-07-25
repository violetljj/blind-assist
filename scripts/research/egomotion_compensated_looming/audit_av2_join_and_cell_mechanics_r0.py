#!/usr/bin/env python3
"""Audit AV2 timestamp joining and rigid-rig cell mechanics without payload GETs."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import statistics
import urllib.request
from pathlib import Path

from probe_av2_official_inventory_r0 import DATASET_PREFIX, list_objects


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY = (
    ROOT
    / "artifacts.local"
    / "evidence"
    / "ustrf"
    / "egomotion_compensated_looming_r0"
    / "source_audit"
    / "av2_official_inventory_r0.json"
)
EXPECTED_INVENTORY_SHA256 = (
    "69cd1a22422dc4a6a1a128399a3b8f268dae6b2378fc45cf2d82add05e1d9e12"
)
COHORT_SIZE = 24
EXPECTED_COHORT_SHA256 = (
    "b7a1645eb67a4901e4b8f5a582fd31bb46d8b91b0347aed232a7c3a0b5bbc09e"
)
JOIN_TOLERANCE_NS = 25_000_000
AV2_API_COMMIT = "b7321d1f71f6ce0ecdd151f4f2b648338c191edd"
SYNC_SOURCE_URL = (
    "https://raw.githubusercontent.com/argoverse/av2-api/"
    f"{AV2_API_COMMIT}/src/av2/utils/synchronization_database.py"
)
EXPECTED_SYNC_SOURCE_SHA256 = (
    "0366b88959bacf8be214ebb64ec19ed636035ea001a4068bee9921d537d961ac"
)
REQUIRED_TABLES = (
    "annotations.feather",
    "city_SE3_egovehicle.feather",
    "calibration/egovehicle_SE3_sensor.feather",
    "calibration/intrinsics.feather",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(key: str, suffix: str) -> int:
    return int(key.rsplit("/", 1)[1].removesuffix(suffix))


def _cohort(inventory: dict) -> tuple[list[tuple[str, str, str]], str]:
    ranked: list[tuple[str, str, str]] = []
    for split in ("train", "val"):
        for log_id in inventory["log_ids"][split]:
            selection_hash = hashlib.sha256(
                f"{split}\t{log_id}".encode("utf-8")
            ).hexdigest()
            ranked.append((selection_hash, split, log_id))
    selected = sorted(ranked)[:COHORT_SIZE]
    identity = "\n".join(
        f"{split}\t{log_id}\t{selection_hash}"
        for selection_hash, split, log_id in selected
    )
    return selected, hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _nearest_delta(query: int, references: list[int]) -> tuple[int | None, bool]:
    index = bisect.bisect_left(references, query)
    candidates = references[max(0, index - 1) : min(len(references), index + 1)]
    if not candidates:
        return None, False
    delta = min(abs(value - query) for value in candidates)
    tied = sum(abs(value - query) == delta for value in candidates) != 1
    return delta, tied


def build_receipt(inventory_path: Path) -> dict:
    if _sha256(inventory_path) != EXPECTED_INVENTORY_SHA256:
        raise AssertionError("AV2 inventory hash drift")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    cohort, cohort_sha256 = _cohort(inventory)
    if cohort_sha256 != EXPECTED_COHORT_SHA256:
        raise AssertionError("AV2 cohort identity drift")

    with urllib.request.urlopen(SYNC_SOURCE_URL, timeout=30) as response:
        sync_source = response.read()
    sync_source_sha256 = hashlib.sha256(sync_source).hexdigest()
    if sync_source_sha256 != EXPECTED_SYNC_SOURCE_SHA256:
        raise AssertionError("AV2 synchronization source drift")

    total_anchors = 0
    matched_anchors = 0
    tie_count = 0
    over_tolerance: list[dict[str, int | str]] = []
    log_median_deltas: list[int] = []
    required_table_bytes = {name: 0 for name in REQUIRED_TABLES}
    selected_logs: list[dict] = []

    for selection_hash, split, log_id in cohort:
        prefix = f"{DATASET_PREFIX}{split}/{log_id}/"
        objects = list_objects(prefix)
        relative = {
            str(item["key"])[len(prefix) :]: item
            for item in objects
        }
        missing = [name for name in REQUIRED_TABLES if name not in relative]
        if missing:
            raise AssertionError(f"missing required AV2 tables: {split}/{log_id}: {missing}")
        for name in REQUIRED_TABLES:
            required_table_bytes[name] += int(relative[name]["bytes"])

        lidar = sorted(
            _timestamp(name, ".feather")
            for name in relative
            if name.startswith("sensors/lidar/") and name.endswith(".feather")
        )
        camera = sorted(
            _timestamp(name, ".jpg")
            for name in relative
            if name.startswith("sensors/cameras/ring_front_center/")
            and name.endswith(".jpg")
        )
        deltas: list[int] = []
        log_matches = 0
        for anchor in lidar:
            total_anchors += 1
            delta, tied = _nearest_delta(anchor, camera)
            if tied:
                tie_count += 1
                continue
            if delta is None:
                continue
            deltas.append(delta)
            if delta <= JOIN_TOLERANCE_NS:
                matched_anchors += 1
                log_matches += 1
            else:
                over_tolerance.append(
                    {
                        "split": split,
                        "log_id": log_id,
                        "lidar_timestamp_ns": anchor,
                        "nearest_delta_ns": delta,
                    }
                )
        if deltas:
            log_median_deltas.append(int(statistics.median(deltas)))
        selected_logs.append(
            {
                "split": split,
                "log_id": log_id,
                "selection_hash": selection_hash,
                "lidar_anchor_count": len(lidar),
                "ring_front_center_timestamp_count": len(camera),
                "unique_join_within_25ms_count": log_matches,
            }
        )

    return {
        "schema_version": "av2_join_and_cell_mechanics_audit_r0",
        "source_id": "ARGOVERSE_2_SENSOR",
        "inventory_path": inventory_path.relative_to(ROOT).as_posix(),
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
        "payload_get_count": 0,
        "rgb_or_sensor_payload_decoded": False,
        "candidate_signal_computed": False,
        "cohort_contract": {
            "eligible_splits": ["train", "val"],
            "selection": "lowest 24 SHA256(split + TAB + log_id)",
            "cohort_size": COHORT_SIZE,
            "cohort_identity_sha256": cohort_sha256,
            "cohort_role": "SOURCE_PRESCREEN_ONLY",
        },
        "selected_logs": selected_logs,
        "official_join_contract": {
            "av2_api_commit": AV2_API_COMMIT,
            "synchronization_source_url": SYNC_SOURCE_URL,
            "synchronization_source_sha256": sync_source_sha256,
            "source_anchor": "10Hz lidar filename timestamp_ns",
            "target_camera": "ring_front_center",
            "method": "unique nearest camera filename timestamp",
            "tolerance_ns": JOIN_TOLERANCE_NS,
            "truth_interpolation_or_duplication_to_20hz": False,
            "annotation_table_timestamp_coverage": "NOT_EVALUATED",
        },
        "join_audit": {
            "scope": "LIDAR_FILENAME_TO_CAMERA_FILENAME_ONLY",
            "lidar_anchor_count": total_anchors,
            "unique_join_within_25ms_count": matched_anchors,
            "tie_count": tie_count,
            "over_tolerance_count": len(over_tolerance),
            "over_tolerance": over_tolerance,
            "median_of_log_median_delta_ns": int(
                statistics.median(log_median_deltas)
            ),
        },
        "geometry_table_preview": {
            "member_count": COHORT_SIZE * len(REQUIRED_TABLES),
            "member_bytes_by_name": required_table_bytes,
            "total_bytes": sum(required_table_bytes.values()),
            "downloaded": False,
            "annotation_to_camera_join_evaluated": False,
        },
        "cell_mechanics": {
            "sensor_rig": "rigid roof-mounted cameras on Ford Fusion Hybrid",
            "independent_head_or_camera_rotation_dof": False,
            "pure_ego_rotation_no_closing_real_cell": "STRUCTURALLY_ABSENT",
            "vehicle_yaw_with_translation_may_not_be_relabelled_as_pure_rotation": True,
            "synthetic_rotation_may_count_toward_real_source_minimum": False,
        },
        "capture_authority": {
            "log_uuid_is_session_id": True,
            "log_uuid_is_parent_capture_cluster_id": False,
            "published_parent_drive_or_burst_id_present": False,
            "may_freeze_role_split": False,
        },
        "source_admission": "HOLD_R0_ADMISSION",
        "terminal": "AV2_REQUIRED_PURE_ROTATION_CELL_STRUCTURALLY_ABSENT",
        "status": "VALID",
        "authority": {
            "may_download_geometry_tables_for_r0_cell_rescue": False,
            "may_download_or_decode_rgb": False,
            "may_run_signal": False,
            "may_count_av2_toward_three_real_sources": False,
            "may_retain_as_vehicle_diagnostic_pressure_source": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(args.inventory.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "terminal": receipt["terminal"],
                "join_audit": receipt["join_audit"],
                "geometry_table_bytes": receipt["geometry_table_preview"][
                    "total_bytes"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
