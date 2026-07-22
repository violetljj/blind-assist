"""Run the generic fail-closed GT-only R3 prescreen for a third source.

This tool consumes only RGB timestamps, independent body-pose ground truth,
and static body-to-camera transforms from a ROS 2 bag.  It does not decode
RGB/depth pixels, derive candidate alerts, grant source credit, or run the
evaluator.  A pass therefore has rejection-only authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from contract import quaternion_matrix, read_json, sha256, validate_pose, write_json
from prescreen_lilocbench_sources import route_proxy_stats


SCHEMA = "blindassist_ustrf_sensor_replay_r3_third_source_gt_prescreen_v1"
DISCOVERY_SCHEMA = "blindassist_ustrf_sensor_replay_r3_third_source_discovery_v1"
INPUT_SCHEMA = "blindassist_ustrf_sensor_replay_r3_third_source_sequence_input_v1"
TERMINAL_REJECTION_SCHEMA = "blindassist_ustrf_sensor_replay_r3_third_source_sequence_rejection_v1"


def timestamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def associate_rgb_to_gt(rgb_ns: np.ndarray, gt_ns: np.ndarray, maximum_delta_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rgb_ns.ndim != 1 or gt_ns.ndim != 1 or len(rgb_ns) < 2 or len(gt_ns) < 2:
        raise ValueError("insufficient RGB or GT timestamps")
    if np.any(np.diff(rgb_ns) <= 0) or np.any(np.diff(gt_ns) <= 0):
        raise ValueError("RGB and GT timestamps must be strictly increasing")
    right = np.searchsorted(gt_ns, rgb_ns, side="left")
    right = np.clip(right, 0, len(gt_ns) - 1)
    left = np.clip(right - 1, 0, len(gt_ns) - 1)
    choose_left = np.abs(gt_ns[left] - rgb_ns) <= np.abs(gt_ns[right] - rgb_ns)
    indices = np.where(choose_left, left, right)
    deltas = np.abs(gt_ns[indices] - rgb_ns)
    return indices, deltas <= maximum_delta_ns, deltas


def transform_entry(transform: Any) -> dict[str, Any]:
    value = transform.transform
    return {
        "parent": transform.header.frame_id,
        "child": transform.child_frame_id,
        "translation": {
            "x": float(value.translation.x),
            "y": float(value.translation.y),
            "z": float(value.translation.z),
        },
        "rotation": {
            "x": float(value.rotation.x),
            "y": float(value.rotation.y),
            "z": float(value.rotation.z),
            "w": float(value.rotation.w),
        },
    }


def transform_matrix(entry: dict[str, Any]) -> np.ndarray:
    translation = entry["translation"]
    rotation = entry["rotation"]
    matrix = np.asarray(quaternion_matrix([
        float(translation["x"]),
        float(translation["y"]),
        float(translation["z"]),
        float(rotation["x"]),
        float(rotation["y"]),
        float(rotation["z"]),
        float(rotation["w"]),
    ]), dtype=np.float64)
    validate_pose(matrix.tolist())
    return matrix


def compose_transform_chain(entries: list[dict[str, Any]], frames: list[str]) -> np.ndarray:
    if len(frames) < 2 or len(set(frames)) != len(frames):
        raise ValueError("invalid body-to-camera transform chain")
    by_pair = {(entry["parent"], entry["child"]): entry for entry in entries}
    result = np.eye(4, dtype=np.float64)
    for parent, child in zip(frames, frames[1:]):
        entry = by_pair.get((parent, child))
        if entry is None:
            raise ValueError(f"missing static transform {parent}_T_{child}")
        result = result @ transform_matrix(entry)
    validate_pose(result.tolist())
    return result


def validate_front_color_optical(body_from_color: np.ndarray) -> np.ndarray:
    validate_pose(body_from_color.tolist())
    optical_axis = body_from_color[:3, 2]
    if float(optical_axis[0]) < 0.95 or float(optical_axis[0]) <= abs(float(optical_axis[1])):
        raise ValueError(f"RGB-D color optical axis is not body-forward: {optical_axis.tolist()}")
    return optical_axis


def read_rosbag_gt(bag_root: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int]]:
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as error:
        raise ValueError("rosbags runtime is required for ROS 2 GT extraction") from error

    precheck = config["gt_only_prescreen"]
    rgb_topic = precheck["rgb_topic"]
    gt_parent = precheck["pose_parent_frame"]
    gt_child = precheck["pose_child_frame"]
    required_topics = {rgb_topic, "/tf", "/tf_static"}
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    rgb_stamps: list[int] = []
    gt_rows: list[list[float]] = []
    gt_stamps: list[int] = []
    static_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    topic_counts = {topic: 0 for topic in required_topics}

    with Reader(bag_root) as reader:
        present = {connection.topic for connection in reader.connections}
        missing = required_topics - present
        if missing:
            raise ValueError(f"missing required ROS topics: {sorted(missing)}")
        connections = [connection for connection in reader.connections if connection.topic in required_topics]
        for connection, _, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_cdr(rawdata, connection.msgtype)
            topic_counts[connection.topic] += 1
            if connection.topic == rgb_topic:
                rgb_stamps.append(timestamp_ns(message.header.stamp))
            elif connection.topic == "/tf":
                for transform in message.transforms:
                    if transform.header.frame_id == gt_parent and transform.child_frame_id == gt_child:
                        value = transform.transform
                        gt_stamps.append(timestamp_ns(transform.header.stamp))
                        gt_rows.append([
                            float(value.translation.x),
                            float(value.translation.y),
                            float(value.translation.z),
                            float(value.rotation.x),
                            float(value.rotation.y),
                            float(value.rotation.z),
                            float(value.rotation.w),
                        ])
            else:
                for transform in message.transforms:
                    entry = transform_entry(transform)
                    pair = (entry["parent"], entry["child"])
                    if pair in static_by_pair and static_by_pair[pair] != entry:
                        raise ValueError(f"conflicting static transform: {pair}")
                    static_by_pair[pair] = entry

    if len(rgb_stamps) < 2 or len(gt_rows) < 2:
        raise ValueError("insufficient RGB timestamps or independent body poses")
    rgb_order = np.argsort(np.asarray(rgb_stamps, dtype=np.int64))
    gt_order = np.argsort(np.asarray(gt_stamps, dtype=np.int64))
    rgb_ns = np.asarray(rgb_stamps, dtype=np.int64)[rgb_order]
    gt_ns = np.asarray(gt_stamps, dtype=np.int64)[gt_order]
    poses = np.asarray(gt_rows, dtype=np.float64)[gt_order]
    if np.any(np.diff(rgb_ns) <= 0) or np.any(np.diff(gt_ns) <= 0):
        raise ValueError("duplicate or non-monotonic RGB/GT header timestamps")
    quaternion_norms = np.linalg.norm(poses[:, 3:7], axis=1)
    if np.any(~np.isfinite(poses)) or np.any(np.abs(quaternion_norms - 1.0) > 1e-3):
        raise ValueError("invalid body-pose ground truth")
    rows = np.column_stack((gt_ns.astype(np.float64) / 1e9, poses))
    return rgb_ns, rows, list(static_by_pair.values()), topic_counts


def select_sequence(repo: Path, config: dict[str, Any], config_sha: str, sequence_input_path: Path | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    priorities = config["selected_dataset"]["sequence_priority"]
    if sequence_input_path is None:
        selected = [row for row in priorities if row.get("selected") is True]
        if len(selected) != 1 or "remote_bag_member" not in selected[0]:
            raise ValueError("exactly one primary sequence with input binding must be frozen")
        return selected[0], selected[0]["remote_bag_member"], None
    sequence_input = read_json(sequence_input_path)
    if sequence_input.get("schema") != INPUT_SCHEMA:
        raise ValueError("unexpected third-source sequence-input schema")
    if sequence_input.get("discovery_manifest_sha256") != config_sha:
        raise ValueError("sequence input is not bound to the frozen discovery manifest")
    sequence_id = sequence_input["sequence_id"]
    indices = [index for index, row in enumerate(priorities) if row["sequence_id"] == sequence_id]
    if len(indices) != 1:
        raise ValueError("sequence input is not in the frozen priority list")
    index = indices[0]
    required_prior_ids = [row["sequence_id"] for row in priorities[:index]]
    receipts = sequence_input.get("prior_rejection_receipts", [])
    if [row.get("sequence_id") for row in receipts] != required_prior_ids:
        raise ValueError("backup sequence does not bind every prior rejection in order")
    for receipt in receipts:
        path = (repo / receipt["path"]).resolve()
        if sha256(path) != receipt["sha256"]:
            raise ValueError("prior rejection receipt hash mismatch")
        report = read_json(path)
        if report.get("sequence_id") != receipt["sequence_id"] or report.get("config_sha256") != config_sha:
            raise ValueError("prior rejection receipt identity mismatch")
        if report.get("schema") == SCHEMA:
            rejected = report.get("gt_route_prescreen_passed") is False
        elif report.get("schema") == TERMINAL_REJECTION_SCHEMA:
            rejected = report.get("terminal_rejected") is True
            evidence = report.get("evidence", [])
            if not evidence:
                raise ValueError("terminal sequence rejection has no hash-bound evidence")
            for item in evidence:
                evidence_path = (repo / item["path"]).resolve()
                if sha256(evidence_path) != item["sha256"]:
                    raise ValueError("terminal sequence rejection evidence hash mismatch")
            consensus_items = [item for item in evidence if item.get("role") == "review_consensus"]
            if len(consensus_items) != 1:
                raise ValueError("terminal sequence rejection must bind one review consensus")
            consensus = read_json((repo / consensus_items[0]["path"]).resolve())
            source_rows = consensus.get("sources", [])
            if len(source_rows) != 1 or source_rows[0].get("route_event_admitted") is not False:
                raise ValueError("terminal sequence review consensus did not reject source")
        else:
            raise ValueError("unsupported prior sequence rejection schema")
        if not rejected or report.get("source_count_credit") != 0 or report.get("evaluator_ran") is not False:
            raise ValueError("prior sequence is not a fail-closed terminal rejection")
    return priorities[index], sequence_input["remote_bag_member"], sequence_input


def build_report(repo: Path, config_path: Path, bag_root: Path, sequence_input_path: Path | None = None) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema") != DISCOVERY_SCHEMA:
        raise ValueError("unexpected third-source discovery schema")
    prereg_receipt = config["frozen_r3_prereg"]
    prereg_path = (repo / prereg_receipt["path"]).resolve()
    prereg_sha = sha256(prereg_path)
    if prereg_sha != prereg_receipt["sha256"]:
        raise ValueError("frozen R3 prereg hash mismatch")
    prereg = read_json(prereg_path)
    frozen = config["unchanged_contract"]
    route = prereg["route"]
    expected = {
        "truth_horizon_frames": route["truth_horizon_frames"],
        "causal_history_frames": route["causal_history_frames"],
        "minimum_forward_displacement_m": route["minimum_forward_displacement_m"],
        "maximum_unknown_rate": route["maximum_unknown_rate"],
        "maximum_rgb_pose_delta_ms": prereg["synchronization"]["maximum_rgb_pose_delta_ms"],
        "minimum_source_aligned_fraction": prereg["synchronization"]["minimum_source_aligned_fraction"],
        "minimum_admitted_sources": prereg["minimum_admitted_sources"],
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"discovery manifest changed frozen field: {key}")

    config_sha = sha256(config_path)
    sequence, bag_member, sequence_input = select_sequence(repo, config, config_sha, sequence_input_path)
    bag_path = bag_root / "out_0.db3"
    metadata_path = bag_root / "metadata.yaml"
    if bag_path.stat().st_size != int(bag_member["file_size"]):
        raise ValueError("selected bag size mismatch")
    if sha256(bag_path) != bag_member["raw_db3_sha256"]:
        raise ValueError("selected bag SHA-256 mismatch")
    if sha256(metadata_path) != bag_member["metadata_sha256"]:
        raise ValueError("selected metadata SHA-256 mismatch")

    rgb_ns, gt_rows, static_entries, topic_counts = read_rosbag_gt(bag_root, config)
    chain = config["gt_only_prescreen"]["body_to_color_optical_chain"]
    body_from_color = compose_transform_chain(static_entries, chain)
    optical_axis = validate_front_color_optical(body_from_color)
    gt_ns = np.rint(gt_rows[:, 0] * 1e9).astype(np.int64)
    maximum_delta_ns = int(round(float(expected["maximum_rgb_pose_delta_ms"]) * 1e6))
    pose_indices, aligned, deltas_ns = associate_rgb_to_gt(rgb_ns, gt_ns, maximum_delta_ns)
    truth = route_proxy_stats(gt_rows, pose_indices, aligned, int(route["truth_horizon_frames"]), float(route["minimum_forward_displacement_m"]), "truth_future")
    causal = route_proxy_stats(gt_rows, pose_indices, aligned, int(route["causal_history_frames"]), float(route["minimum_forward_displacement_m"]), "causal_history")
    aligned_fraction = float(np.mean(aligned))
    maximum_unknown = float(route["maximum_unknown_rate"])
    gates = {
        "frozen_r3_prereg_hash_match": True,
        "selected_bag_hash_and_size_match": True,
        "required_rgb_tf_topics_present": all(topic_counts.values()),
        "independent_body_pose_available": len(gt_rows) >= 2,
        "body_to_color_optical_chain_forward": float(optical_axis[0]) >= 0.95,
        "rgb_pose_alignment_passed": aligned_fraction >= float(expected["minimum_source_aligned_fraction"]),
        "truth_route_proxy_unknown_rate_passed": truth["route_proxy_unknown_rate"] <= maximum_unknown,
        "causal_route_proxy_unknown_rate_passed": causal["route_proxy_unknown_rate"] <= maximum_unknown,
    }
    rgb_deltas = np.diff(rgb_ns).astype(np.float64) / 1e9
    gt_steps = np.linalg.norm(np.diff(gt_rows[:, 1:4], axis=0), axis=1)
    return {
        "schema": SCHEMA,
        "authority": "discovery_candidate_only_reject_only",
        "config_sha256": config_sha,
        "frozen_r3_prereg_sha256": prereg_sha,
        "dataset_id": config["selected_dataset"]["dataset_id"],
        "sequence_id": sequence["sequence_id"],
        "input_receipt": {
            "bag_path": str(bag_path),
            "bag_size_bytes": bag_path.stat().st_size,
            "bag_sha256": sha256(bag_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "official_zip_member_crc32": bag_member["zip_crc32"],
            "sequence_input_manifest_sha256": sha256(sequence_input_path) if sequence_input is not None and sequence_input_path is not None else None,
        },
        "ground_truth": {
            "authority": "OptiTrack world_T_chair 6D pose",
            "parent_frame": config["gt_only_prescreen"]["pose_parent_frame"],
            "body_frame": config["gt_only_prescreen"]["pose_child_frame"],
            "sample_count": int(len(gt_rows)),
            "duration_s": float(gt_rows[-1, 0] - gt_rows[0, 0]),
            "median_rate_hz": float(1.0 / np.median(np.diff(gt_rows[:, 0]))),
            "path_length_m": float(gt_steps.sum()),
        },
        "time_semantics": {
            "rgb_topic": config["gt_only_prescreen"]["rgb_topic"],
            "actual_rgb_frame_count": int(len(rgb_ns)),
            "actual_rgb_duration_s": float((rgb_ns[-1] - rgb_ns[0]) / 1e9),
            "actual_rgb_median_rate_hz": float(1.0 / np.median(rgb_deltas)),
            "truth_horizon_frames": int(route["truth_horizon_frames"]),
            "truth_horizon_median_seconds": float(route["truth_horizon_frames"] * np.median(rgb_deltas)),
            "causal_history_frames": int(route["causal_history_frames"]),
            "causal_history_median_seconds": float(route["causal_history_frames"] * np.median(rgb_deltas)),
            "rgb_pose_aligned_fraction": aligned_fraction,
            "rgb_pose_delta_p95_ms": float(np.quantile(deltas_ns, 0.95) / 1e6),
            "rgb_pose_delta_max_ms": float(np.max(deltas_ns) / 1e6),
        },
        "camera_geometry": {
            "body_to_color_optical_chain": chain,
            "color_optical_positive_z_in_body": optical_axis.tolist(),
            "forward_facing": True,
            "pixel_geometry_or_event_lifecycle_evaluated": False,
        },
        "topic_message_counts": topic_counts,
        "truth_route_proxy": truth,
        "causal_route_proxy": causal,
        "gates": gates,
        "gt_route_prescreen_passed": all(gates.values()),
        "complete_rgbd_adaptation_performed": False,
        "candidate_generated": False,
        "complete_sequence_two_model_admitted": False,
        "source_count_credit": 0,
        "admitted_source_count_before_review": 2,
        "minimum_admitted_sources_met": False,
        "evaluator_ran": False,
        "next_gate": (
            f"adapt the complete frozen {sequence_id} RGB-D sequence and freeze candidate before isolated review"
            if all(gates.values())
            else f"reject {sequence_id} without RGB-D adaptation"
        ),
        "hardware_selection_authorized": False,
        "u0_authorized": False,
        "production_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--sequence-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.exists():
            raise ValueError("refusing to overwrite third-source GT prescreen output")
        sequence_input = args.sequence_input.resolve() if args.sequence_input is not None else None
        report = build_report(args.repo.resolve(), args.config.resolve(), args.bag_root.resolve(), sequence_input)
        write_json(output, report)
        print(json.dumps({
            "sequence_id": report["sequence_id"],
            "gt_route_prescreen_passed": report["gt_route_prescreen_passed"],
            "source_count_credit": report["source_count_credit"],
            "evaluator_ran": report["evaluator_ran"],
        }))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
