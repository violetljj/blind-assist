#!/usr/bin/env python3
"""CUDA cross-modal audit of REveL RGB person boxes against Vicon helmet poses.

The Dynamic bag supplies RGB timestamps and Vicon transforms; the companion
archives supply ordered RGB images, YOLO boxes, and the green/yellow class map.
Using the official session calibration, this audit projects the matching helmet
marker into each RGB image.  It measures source-data cross-modal consistency,
not detector accuracy or an assistive safety event.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_revel_dynamic_vicon_trajectories import PERSON_TOPICS, SENSOR_TOPIC, _extract_topic, _nearest_indices, _rotation_matrix


SYNC_MAX_DELTA_MS = 20.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_labels(root: Path) -> tuple[list[str], dict[str, list[tuple[int, float, float, float, float]]]]:
    labels = root / "extracted" / "labels" / "labels"
    images = root / "extracted" / "images" / "images"
    stems = sorted((path.stem for path in images.glob("*.jpg")), key=int)
    output: dict[str, list[tuple[int, float, float, float, float]]] = {}
    for stem in stems:
        path = labels / f"{stem}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing label for {stem}")
        entries: list[tuple[int, float, float, float, float]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_id, cx, cy, width, height = line.split()
                entries.append((int(class_id), float(cx), float(cy), float(width), float(height)))
        output[stem] = entries
    return stems, output


def _project_camera_points(points: Any, intrinsics: Any, distortion: Any) -> Any:
    """Project Nx3 camera points using OpenCV k1,k2,p1,p2 convention."""
    import torch

    x = points[:, 0] / points[:, 2].clamp_min(1e-12); y = points[:, 1] / points[:, 2].clamp_min(1e-12)
    k1, k2, p1, p2 = distortion; r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return torch.stack((intrinsics[0, 0] * xd + intrinsics[0, 2], intrinsics[1, 1] * yd + intrinsics[1, 2]), dim=1)


def _stats(values: Any) -> dict[str, float]:
    import torch

    if values.numel() == 0:
        return {"count": 0}
    return {"count": int(values.numel()), "median": float(values.median().item()), "p95": float(torch.quantile(values, .95).item()), "max": float(values.max().item())}


def audit(bag_root: Path, image_label_root: Path) -> dict[str, Any]:
    import torch
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    from ruamel.yaml import YAML

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for REveL RGB/Vicon reprojection audit")
    bag = bag_root / "dynamic.bag"; calibration_path = bag_root / "calibration.yaml"; classes_path = bag_root / "classes.txt"
    if not all(path.is_file() for path in (bag, calibration_path, classes_path)):
        raise FileNotFoundError("bag_root needs dynamic.bag, calibration.yaml, and classes.txt")
    yaml = YAML(typ="safe"); calibration = yaml.load(calibration_path.read_text(encoding="utf-8"))
    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if class_names != ["green-helmet", "yellow-helmet"]:
        raise ValueError(f"unexpected class mapping: {class_names}")
    stems, labels = _parse_labels(image_label_root)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag) as reader:
        image_connection = reader.topics["/dvs/image_raw"].connections[0]
        image_timestamps = np.asarray([timestamp for _, timestamp, _ in reader.messages(connections=[image_connection])], dtype=np.int64)
        sensor = _extract_topic(reader, typestore, SENSOR_TOPIC)
        people = [_extract_topic(reader, typestore, topic) for topic in PERSON_TOPICS]
    if len(stems) != len(image_timestamps):
        raise ValueError(f"archive image count {len(stems)} != bag image count {len(image_timestamps)}")
    device = torch.device("cuda")
    filename_timestamps = np.asarray([int(stem) for stem in stems], dtype=np.int64)
    archive_bag_delta_ms = np.abs(filename_timestamps - image_timestamps).astype(np.float64) / 1e6
    sensor_index = _nearest_indices(image_timestamps, sensor["timestamps_ns"])
    sensor_delta_ms = np.abs(sensor["timestamps_ns"][sensor_index] - image_timestamps).astype(np.float64) / 1e6
    t_v_c = torch.as_tensor(np.asarray(calibration["T_v_c"], dtype=np.float64), device=device)
    intrinsics = torch.as_tensor(np.asarray(calibration["K"], dtype=np.float64), device=device)
    distortion = torch.as_tensor(np.asarray(calibration["distortion"], dtype=np.float64), device=device)
    sensor_position = torch.as_tensor(sensor["positions"][sensor_index], dtype=torch.float64, device=device)
    sensor_rotation = _rotation_matrix(torch.as_tensor(sensor["quaternions"][sensor_index], dtype=torch.float64, device=device))
    all_results: dict[str, Any] = {}
    for class_id, person in enumerate(people):
        person_index = _nearest_indices(image_timestamps, person["timestamps_ns"])
        person_delta_ms = np.abs(person["timestamps_ns"][person_index] - image_timestamps).astype(np.float64) / 1e6
        person_position = torch.as_tensor(person["positions"][person_index], dtype=torch.float64, device=device)
        local_vicon = torch.bmm(sensor_rotation.transpose(1, 2), (person_position - sensor_position).unsqueeze(2)).squeeze(2)
        camera = (t_v_c[:3, :3].transpose(0, 1) @ (local_vicon - t_v_c[:3, 3]).T).T
        pixel = _project_camera_points(camera, intrinsics, distortion)
        image_indices = [index for index, stem in enumerate(stems) if any(row[0] == class_id for row in labels[stem])]
        label_boxes: list[tuple[float, float, float, float]] = []; box_frame_slots: list[int] = []; ambiguous_frames = 0
        for frame_slot, index in enumerate(image_indices):
            matching = [row for row in labels[stems[index]] if row[0] == class_id]
            ambiguous_frames += int(len(matching) > 1)
            label_boxes.extend(row[1:] for row in matching); box_frame_slots.extend([frame_slot] * len(matching))
        selection = torch.as_tensor(image_indices, dtype=torch.long, device=device)
        boxes = torch.as_tensor(label_boxes, dtype=torch.float64, device=device)
        frame_slots = torch.as_tensor(box_frame_slots, dtype=torch.long, device=device)
        selected_pixel = pixel[selection]; selected_camera = camera[selection]
        valid_sync = torch.as_tensor((sensor_delta_ms[image_indices] <= SYNC_MAX_DELTA_MS) & (person_delta_ms[image_indices] <= SYNC_MAX_DELTA_MS), device=device)
        positive_depth = selected_camera[:, 2] > 0
        width, height = 346.0, 260.0
        inside_image = (selected_pixel[:, 0] >= 0) & (selected_pixel[:, 0] < width) & (selected_pixel[:, 1] >= 0) & (selected_pixel[:, 1] < height)
        centre = boxes[:, :2] * torch.tensor([width, height], dtype=torch.float64, device=device)
        half = boxes[:, 2:] * torch.tensor([width, height], dtype=torch.float64, device=device) / 2
        candidate_pixel = selected_pixel[frame_slots]
        inside_box = (torch.abs(candidate_pixel - centre) <= half).all(dim=1)
        clipped = torch.maximum(torch.abs(candidate_pixel - centre) - half, torch.zeros_like(candidate_pixel))
        outside_distance_px = torch.linalg.vector_norm(clipped, dim=1)
        usable = valid_sync & positive_depth & inside_image
        inside_any = torch.zeros(len(image_indices), dtype=torch.int64, device=device)
        inside_any.scatter_reduce_(0, frame_slots, inside_box.to(torch.int64), reduce="amax", include_self=True)
        closest_outside = torch.full((len(image_indices),), float("inf"), dtype=torch.float64, device=device)
        closest_outside.scatter_reduce_(0, frame_slots, outside_distance_px, reduce="amin", include_self=True)
        all_results[class_names[class_id]] = {
            "labelled_frame_count": len(image_indices), "label_box_count": len(label_boxes), "ambiguous_same_class_frame_count": ambiguous_frames,
            "valid_sensor_and_person_sync_count": int(valid_sync.sum().item()),
            "positive_depth_count": int(positive_depth.sum().item()), "inside_image_count": int(inside_image.sum().item()),
            "inside_any_matching_box_count": int((usable & inside_any.bool()).sum().item()),
            "inside_any_matching_box_fraction_of_usable": float((usable & inside_any.bool()).double().sum().item() / usable.double().sum().clamp_min(1).item()),
            "closest_matching_box_outside_distance_px": _stats(closest_outside[usable]),
            "sensor_sync_delta_ms": _stats(torch.as_tensor(sensor_delta_ms[image_indices], dtype=torch.float64, device=device)[valid_sync]),
            "person_sync_delta_ms": _stats(torch.as_tensor(person_delta_ms[image_indices], dtype=torch.float64, device=device)[valid_sync]),
        }
    report = {
        "format": "blindassist_revel_rgb_vicon_reprojection_audit_v1",
        "source": {"bag_sha256": _sha256(bag), "calibration_sha256": _sha256(calibration_path), "class_map": class_names, "image_size_px": [346, 260]},
        "frame_alignment": {"archive_image_count": len(stems), "bag_image_count": len(image_timestamps), "archive_filename_to_bag_timestamp_delta_ms": _stats(torch.as_tensor(archive_bag_delta_ms, dtype=torch.float64, device=device))},
        "reprojection": all_results,
        "admission": {
            "source_cross_modal_2d_3d_alignment_admitted": True,
            "admitted_for": ["offline RGB-box to Vicon-helmet reprojection consistency"],
            "not_admitted_for": ["metric depth", "physical assistive TTC", "body-local safe corridor", "assistive event truth", "on-device safety"],
            "reason": "the check binds source RGB labels, calibration, and helmet markers; it does not verify a user body or target-device mount",
        },
        "compute_backend": {"name": "torch", "cuda": True, "device": torch.cuda.get_device_name(0)}, "production_authority": False,
    }
    qa = bag_root / "qa"; qa.mkdir(exist_ok=True)
    (qa / "revel_rgb_vicon_reprojection_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--image-label-root", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.bag_root, args.image_label_root)
    print(json.dumps({"device": report["compute_backend"]["device"], "classes": list(report["reprojection"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
