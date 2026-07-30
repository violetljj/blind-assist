"""Freeze JRDB annotation-conditioned LiDAR centroid rows for kernel replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_SEQUENCES = {
    "meyer-green-2019-03-16_0",
    "gates-basement-elevators-2019-01-17_1",
    "stlc-111-2019-04-19_0",
    "clark-center-2019-02-28_0",
}
EXPECTED_SOURCE_ELIGIBLE_ROWS = 8836
ELIGIBLE = "ELIGIBLE"
NOT_ELIGIBLE = "NOT_ELIGIBLE"
FIELDNAMES = [
    "sequence",
    "frame_index",
    "image_timestamp_ns",
    "available_at_ns",
    "frame_width",
    "frame_height",
    "label_id",
    "left",
    "top",
    "right",
    "bottom",
    "box_clamped",
    "geometry_status",
    "geometry_reason",
    "previous_frame_index",
    "previous_timestamp_ns",
    "signed_approach_rate_per_s",
    "quality",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def centroid_range(row: dict[str, Any]) -> float:
    values = row["sensor_centroid_logical_rgb360_m"]
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError("sensor centroid must have three coordinates")
    coordinates = [float(value) for value in values]
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("sensor centroid is not finite")
    return math.sqrt(sum(value * value for value in coordinates))


def signed_approach_rate(
    previous_object: dict[str, Any],
    current_object: dict[str, Any],
    previous_timestamp_ns: int,
    current_timestamp_ns: int,
) -> float:
    delta_ns = current_timestamp_ns - previous_timestamp_ns
    if delta_ns <= 0:
        raise ValueError("frame timestamps are not strictly increasing")
    rate = (
        centroid_range(previous_object) - centroid_range(current_object)
    ) / (delta_ns / 1_000_000_000.0)
    if not math.isfinite(rate):
        raise ValueError("signed approach rate is not finite")
    return rate


def packet_binding(
    packet_path: Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    actual = sha256_file(packet_path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"observation packet hash mismatch: {packet_path}")
    packet = read_json(packet_path)
    if packet.get("status") != "IMMUTABLE_OBSERVATION_PACKET":
        raise ValueError(f"observation packet is not immutable: {packet_path}")
    return packet


def sequence_rows(
    packet: dict[str, Any],
    ledger: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sequence = str(packet["sequence"])
    if str(ledger["sequence"]) != sequence or ledger.get("status") != "COMPLETE":
        raise ValueError(f"packet/ledger sequence mismatch: {sequence}")
    frames = {int(row["frame_index"]): row for row in packet["frames"]}
    if len(frames) != 120:
        raise ValueError(f"expected 120 frames for {sequence}")
    joined: dict[tuple[int, str], dict[str, Any]] = {}
    for frame_index, frame in frames.items():
        for target in frame["labels"]["joined"]:
            key = (frame_index, str(target["label_id"]))
            if key in joined:
                raise ValueError(f"duplicate exact join: {sequence}/{key}")
            joined[key] = target
    objects = {
        (int(row["frame_index"]), str(row["label_id"])): row
        for row in ledger["object_frames"]
    }
    motions = {
        (int(row["right_frame"]), str(row["label_id"])): row
        for row in ledger["motion_pairs"]
    }
    rows: list[dict[str, Any]] = []
    eligible = 0
    missing_frame_targets = 0
    clamped_boxes = 0
    for frame_index in sorted(frames):
        if frame_index == min(frames):
            continue
        frame = frames[frame_index]
        frame_targets = [
            target
            for (target_frame, _), target in joined.items()
            if target_frame == frame_index
        ]
        if not frame_targets:
            missing_frame_targets += 1
            continue
        image_timestamp_ns = int(frame["time"]["image_timestamp_ns"])
        available_at_ns = max(
            image_timestamp_ns,
            int(frame["time"]["lower_pointcloud_timestamp_ns"]),
            int(frame["time"]["upper_pointcloud_timestamp_ns"]),
        )
        width = int(frame["source"]["image"]["width"])
        height = int(frame["source"]["image"]["height"])
        for target in sorted(frame_targets, key=lambda value: str(value["label_id"])):
            label_id = str(target["label_id"])
            x, y, box_width, box_height = [
                float(value) for value in target["box_2d_xywh"]
            ]
            raw_box = (x, y, x + box_width, y + box_height)
            if not all(math.isfinite(value) for value in raw_box) or box_width <= 0 or box_height <= 0:
                raise ValueError(f"invalid 2D box: {sequence}/{frame_index}/{label_id}")
            box = (
                min(max(raw_box[0], 0.0), float(width)),
                min(max(raw_box[1], 0.0), float(height)),
                min(max(raw_box[2], 0.0), float(width)),
                min(max(raw_box[3], 0.0), float(height)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"empty clamped 2D box: {sequence}/{frame_index}/{label_id}")
            box_clamped = box != raw_box
            if box_clamped:
                clamped_boxes += 1
            motion = motions.get((frame_index, label_id))
            previous_frame_index = (
                int(motion["left_frame"]) if motion is not None else None
            )
            previous_frame = (
                frames.get(previous_frame_index)
                if previous_frame_index is not None
                else None
            )
            previous_timestamp_ns = (
                int(previous_frame["time"]["image_timestamp_ns"])
                if previous_frame is not None
                else None
            )
            previous_object = (
                objects.get((previous_frame_index, label_id))
                if previous_frame_index is not None
                else None
            )
            current_object = objects.get((frame_index, label_id))
            previous_join = (
                joined.get((previous_frame_index, label_id))
                if previous_frame_index is not None
                else None
            )
            geometry_status = NOT_ELIGIBLE
            geometry_reason = "NO_SENSOR_SUPPORTED_EXACT_PAIR"
            rate: float | None = None
            quality: float | None = None
            if (
                motion is not None
                and motion.get("classification") == "sensor-supported"
                and previous_object is not None
                and previous_object.get("classification") == "sensor-supported"
                and current_object is not None
                and current_object.get("classification") == "sensor-supported"
                and previous_join is not None
                and previous_timestamp_ns is not None
            ):
                rate = signed_approach_rate(
                    previous_object,
                    current_object,
                    previous_timestamp_ns,
                    image_timestamp_ns,
                )
                geometry_status = ELIGIBLE
                geometry_reason = "BOTH_ENDPOINTS_SENSOR_SUPPORTED_EXACT_2D_JOIN"
                quality = 1.0
                eligible += 1
            rows.append(
                {
                    "sequence": sequence,
                    "frame_index": frame_index,
                    "image_timestamp_ns": image_timestamp_ns,
                    "available_at_ns": available_at_ns,
                    "frame_width": width,
                    "frame_height": height,
                    "label_id": label_id,
                    "left": format(box[0], ".9g"),
                    "top": format(box[1], ".9g"),
                    "right": format(box[2], ".9g"),
                    "bottom": format(box[3], ".9g"),
                    "box_clamped": str(box_clamped).lower(),
                    "geometry_status": geometry_status,
                    "geometry_reason": geometry_reason,
                    "previous_frame_index": (
                        previous_frame_index if previous_frame_index is not None else ""
                    ),
                    "previous_timestamp_ns": (
                        previous_timestamp_ns if previous_timestamp_ns is not None else ""
                    ),
                    "signed_approach_rate_per_s": (
                        format(rate, ".17g") if rate is not None else ""
                    ),
                    "quality": format(quality, ".1f") if quality is not None else "",
                }
            )
    return rows, {
        "frames": len(frames),
        "detection_rows": len(rows),
        "source_eligible_rows": eligible,
        "frames_without_exact_joined_target": missing_frame_targets,
        "clamped_boxes": clamped_boxes,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=path.parent,
        suffix=".tmp",
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def produce(
    canary_packet_path: Path,
    canary_ledger_path: Path,
    cross_manifest_path: Path,
    cross_ledger_path: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("output namespace already exists")
    canary_ledger = read_json(canary_ledger_path)
    canary_packet = packet_binding(
        canary_packet_path,
        str(canary_ledger["parent_packet_sha256"]),
    )
    cross_manifest = read_json(cross_manifest_path)
    cross_ledger = read_json(cross_ledger_path)
    if cross_manifest.get("status") != "INPUTS_FROZEN_BEFORE_SUPPORT":
        raise ValueError("cross-sequence manifest is not frozen")
    if cross_ledger.get("status") != "COMPLETE":
        raise ValueError("cross-sequence ledger is not complete")
    if cross_ledger.get("input_manifest_sha256") != sha256_file(cross_manifest_path):
        raise ValueError("cross-sequence ledger does not bind the input manifest")
    ledger_by_sequence = {
        str(entry["sequence"]): entry["ledger"]
        for entry in cross_ledger["per_sequences"]
    }
    packets: dict[str, dict[str, Any]] = {
        str(canary_packet["sequence"]): canary_packet
    }
    packet_hash_by_sequence: dict[str, str] = {
        str(canary_packet["sequence"]): sha256_file(canary_packet_path)
    }
    source_hashes: dict[str, str] = {
        canary_packet_path.as_posix(): sha256_file(canary_packet_path),
        canary_ledger_path.as_posix(): sha256_file(canary_ledger_path),
        cross_manifest_path.as_posix(): sha256_file(cross_manifest_path),
        cross_ledger_path.as_posix(): sha256_file(cross_ledger_path),
    }
    for binding in cross_manifest["bindings"]:
        sequence = str(binding["sequence"])
        packet_meta = binding["files"]["observation_packet"]
        packet_path = Path(str(packet_meta["path"]))
        packets[sequence] = packet_binding(packet_path, str(packet_meta["sha256"]))
        packet_hash_by_sequence[sequence] = str(packet_meta["sha256"])
        source_hashes[packet_path.as_posix()] = sha256_file(packet_path)
    if set(packets) != EXPECTED_SEQUENCES:
        raise ValueError("unexpected JRDB sequence set")

    all_rows: list[dict[str, Any]] = []
    per_sequence: dict[str, dict[str, int]] = {}
    for sequence in sorted(packets):
        ledger = (
            canary_ledger
            if sequence == str(canary_packet["sequence"])
            else ledger_by_sequence[sequence]
        )
        if ledger.get("parent_packet_sha256") != packet_hash_by_sequence[sequence]:
            raise ValueError(f"sensor-support ledger does not bind packet: {sequence}")
        rows, summary = sequence_rows(packets[sequence], ledger)
        all_rows.extend(rows)
        per_sequence[sequence] = summary
    all_rows.sort(
        key=lambda row: (
            str(row["sequence"]),
            int(row["frame_index"]),
            str(row["label_id"]),
        )
    )
    eligible = sum(
        1 for row in all_rows if row["geometry_status"] == ELIGIBLE
    )
    if eligible != EXPECTED_SOURCE_ELIGIBLE_ROWS:
        raise ValueError(
            f"expected {EXPECTED_SOURCE_ELIGIBLE_ROWS} source-eligible rows, found {eligible}"
        )
    if any(
        summary["frames_without_exact_joined_target"] != 0
        for summary in per_sequence.values()
    ):
        raise ValueError("at least one replay frame has no exact joined target")
    write_tsv(output_path, all_rows)
    receipt = {
        "schema": "blindassist.dual_loop_jrdb_shadow_producer_receipt.v1",
        "status": "COMPLETE",
        "source_id": "JRDB_ANNOTATION_CONDITIONED_LIDAR_CENTROID_REPLAY_V1",
        "implementation_sha256": sha256_file(Path(__file__)),
        "claim_ceiling": "DIAGNOSTIC_ENGINEERING_ONLY",
        "source_annotation_target_opened": True,
        "lidar_measurements_opened": True,
        "protected_alert_outcomes_opened": False,
        "signed_rate_formula": "(range_previous_m-range_current_m)/dt_seconds",
        "quality_rule": "1.0_when_frozen_binary_sensor_support_contract_passes",
        "availability_rule": "max(rgb,lower_lidar,upper_lidar)_current_frame",
        "sequence_count": len(packets),
        "frame_count": sum(value["frames"] for value in per_sequence.values()),
        "detection_rows": len(all_rows),
        "source_eligible_rows": eligible,
        "per_sequence": per_sequence,
        "source_hashes": source_hashes,
        "output_path": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-packet", type=Path, required=True)
    parser.add_argument("--canary-ledger", type=Path, required=True)
    parser.add_argument("--cross-manifest", type=Path, required=True)
    parser.add_argument("--cross-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            produce(
                args.canary_packet,
                args.canary_ledger,
                args.cross_manifest,
                args.cross_ledger,
                args.output,
                args.receipt,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
